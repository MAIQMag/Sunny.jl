# A batched Hermitian eigensolver in Float32, with one tile of W ≥ n lanes per
# matrix for n ≤ 32, after the tiled kernels of EigenG-Batched (RIKEN) for CUDA
# and HIP, and one SIMD group per matrix beyond, after its warp kernel:
# Householder reduction to tridiagonal form, implicit QL, and
# back-transformation. EigenG is real symmetric. Here the reduction is
# the complex one of LAPACK's zhetd2, whose reflectors leave a real tridiagonal
# for the QL iteration of EISPACK's tql2. The scalar reference, against which
# this kernel is tested, is `HouseholderQL` in test/test_householder_ql.jl.
#
# EigenG-Batched is (C) Copyright 2022-24, RIKEN, and is distributed under a
# BSD-style licence that requires retention of its copyright notice, list of
# conditions, and disclaimer:
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are
#   met:
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS
#   IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#   OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
#   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
#   OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
#   ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# EISPACK, from which EigenG derives, is available from netlib,
# http://www.netlib.org/eispack/.
#
# Lane i of a tile owns row i of the matrix and of its eigenvectors, both in
# device memory, and holds d[i], e[i] and τ[i] of the tridiagonal and
# reflectors in registers. Other lanes' values arrive only by shuffles within
# the tile, so no barrier is needed, and the tiles of a SIMD group may iterate
# independently. The scalar recurrences of the reflectors and of QL run
# redundantly on every lane, on identical values. Lanes beyond n contribute
# zeros and touch no memory.

# Status codes of KAExt's batched solvers
const STATUS_OK = Int32(0)
const STATUS_NOCONVERGE = Int32(-3)

# Base's hypot of Float32 computes in Float64, which Apple GPUs lack
@inline function hypot32(x::Float32, y::Float32)
    (a, b) = (abs(x), abs(y))
    (a, b) = (max(a, b), min(a, b))
    a == 0 && return 0f0
    r = b / a
    return a * sqrt(1f0 + r * r)
end
@inline function hypot32(x::Float32, y::Float32, z::Float32)
    m = max(abs(x), abs(y), abs(z))
    m == 0 && return 0f0
    (x, y, z) = (x / m, y / m, z / m)
    return m * sqrt(x * x + y * y + z * z)
end
@inline cinv(z::ComplexF32) = conj(z) / abs2(z)

# The tile width for a matrix of size n ≤ 32
tile_width(n) = n <= 4 ? 4 : n <= 8 ? 8 : n <= 16 ? 16 : n <= 32 ? 32 :
                error("Tiled eigensolver supports n ≤ 32, got $n")

# Eigenvalues λ[:, b] ascending and orthonormal eigenvectors Y[:, :, b] of the
# Hermitian A[:, :, b], both triangles stored, for each b. A is overwritten by
# the reflectors. The status is STATUS_NOCONVERGE where QL fails to converge.
function hermitian_eigen_tiled_kernel!(λ, Y, A, status, ::Val{W}) where W
    n = size(A, 1)
    b = (Int(Metal.thread_position_in_grid_1d()) - 1) ÷ W + 1
    b > size(A, 3) && return
    lane = tile_lane(Val(W))
    converged = eigen_tile!(λ, Y, A, b, lane <= n, Val(W))
    lane == 1 && (status[b] = converged ? STATUS_OK : STATUS_NOCONVERGE)
    return
end

# The work of one tile on matrix b, called by every lane of the tile, where
# `row` is whether the lane owns a row, i.e., lane ≤ n. Contains no barrier, so
# tiles may call it under different control flow. Returns whether QL converged,
# and stores λ only then.
@inline function eigen_tile!(λ, Y, A, b, row, ::Val{W}) where W
    n = size(A, 1)
    lane = tile_lane(Val(W))
    i = min(lane, n)   # A valid row for addressing; stores are predicated on `row`
    bc(x, root) = tile_bcast(x, root, Val(W))

    # Householder reduction, T = Q† A Q with Q = H₁ ⋯ Hₙ₋₁, and Hₖ = I - τₖ vₖ vₖ†
    # acting on rows k+1:n. vₖ is stored in A[k+1:n, k].
    dl = 0f0
    el = 0f0
    τl = zero(ComplexF32)
    for k in 1:n-1
        aik = row ? A[i, k, b] : zero(ComplexF32)
        below = row & (lane >= k + 2)
        xnorm = sqrt(tile_sum(below ? abs2(aik) : 0f0, Val(W)))
        α = bc(aik, k + 1)
        trivial = (xnorm == 0) & (imag(α) == 0)
        β = trivial ? real(α) : -copysign(hypot32(real(α), imag(α), xnorm), real(α))
        τ = trivial ? zero(ComplexF32) : ComplexF32((β - real(α)) / β, -imag(α) / β)
        scale = trivial ? one(ComplexF32) : cinv(α - β)
        v = lane == k + 1 ? one(ComplexF32) : (below ? aik * scale : zero(ComplexF32))
        (row & (lane >= k + 1)) && (A[i, k, b] = v)
        if lane == k
            el = β
            τl = τ
        end

        # w = τ A₂₂ v, then z = w - ½|τ|² (v† A₂₂ v) v
        x = zero(ComplexF32)
        for j in k+1:n
            aij = row ? A[i, j, b] : zero(ComplexF32)
            x += aij * bc(v, j)
        end
        w = (row & (lane >= k + 1)) ? τ * x : zero(ComplexF32)
        z = w - (τ / 2) * tile_sum(conj(w) * v, Val(W)) * v

        # A₂₂ ← A₂₂ - v z† - z v†, by rows
        for j in k+1:n
            (vj, zj) = (bc(v, j), bc(z, j))
            (row & (lane >= k + 1)) && (A[i, j, b] -= v * conj(zj) + z * conj(vj))
        end
    end
    row && (dl = real(A[i, i, b]))

    # Implicit QL on (d, e), with e[n] = 0, of EISPACK tql2. The rotations
    # accumulate in the rows of Y, starting from the identity.
    if row
        for j in 1:n
            Y[i, j, b] = i == j
        end
    end
    f = 0f0
    tst1 = 0f0
    converged = true
    for l in 1:n
        tst1 = max(tst1, abs(bc(dl, l)) + abs(bc(el, l)))
        m = l
        while m < n && abs(bc(el, m)) > eps(Float32) * tst1
            m += 1
        end
        if m > l
            iter = 0
            while true
                if (iter += 1) > 30
                    converged = false
                    break
                end
                # Shift from the leading 2×2 block
                g = bc(dl, l)
                e_l = bc(el, l)
                p = (bc(dl, l + 1) - g) / 2e_l
                r = hypot32(p, 1f0)
                r = p < 0 ? -r : r
                d_l = e_l / (p + r)
                dl1 = e_l * (p + r)
                dl = lane == l ? d_l : (lane == l + 1 ? dl1 : dl)
                h = g - d_l
                ((lane >= l + 2) & row) && (dl -= h)
                f += h
                # Implicit QL transformation, chasing the bulge from m to l
                p = bc(dl, m)
                (c, c2, c3) = (1f0, 1f0, 1f0)
                el1 = bc(el, l + 1)
                (s, s2) = (0f0, 0f0)
                for ii in m-1:-1:l
                    c3 = c2
                    c2 = c
                    s2 = s
                    (e_i, d_i) = (bc(el, ii), bc(dl, ii))
                    g = c * e_i
                    h = c * p
                    r = hypot32(p, e_i)
                    lane == ii + 1 && (el = s * r)
                    s = e_i / r
                    c = p / r
                    p = c * d_i - s * g
                    lane == ii + 1 && (dl = h + s * (c * g + s * d_i))
                    if row
                        (y0, y1) = (Y[i, ii, b], Y[i, ii+1, b])
                        Y[i, ii+1, b] = s * y0 + c * y1
                        Y[i, ii, b] = c * y0 - s * y1
                    end
                end
                p = -s * s2 * c3 * el1 * e_l / dl1
                if lane == l
                    el = s * p
                    dl = c * p
                end
                abs(s * p) > eps(Float32) * tst1 || break
            end
            converged || break
        end
        if lane == l
            dl += f
            el = 0f0
        end
    end

    if converged
        # Selection sort, ascending, by the first minimal index
        for ii in 1:n-1
            cand = (row & (lane >= ii)) ? dl : Inf32
            dmin = tile_max(-cand, Val(W))
            kmin = -tile_max(cand == -dmin ? -Int32(lane) : -Int32(W + 1), Val(W))
            if kmin != ii
                d_i = bc(dl, ii)
                dl = lane == ii ? -dmin : (lane == kmin ? d_i : dl)
                if row
                    (Y[i, ii, b], Y[i, kmin, b]) = (Y[i, kmin, b], Y[i, ii, b])
                end
            end
        end

        # Y ← Hₖ Y for k = n-1, …, 1, each column a tile sum over rows
        for k in n-1:-1:1
            τk = bc(τl, k)
            v = (row & (lane >= k + 1)) ? A[i, k, b] : zero(ComplexF32)
            for j in 1:n
                yij = row ? Y[i, j, b] : zero(ComplexF32)
                x = τk * tile_sum(conj(v) * yij, Val(W))
                (row & (lane >= k + 1)) && (Y[i, j, b] = yij - v * x)
            end
        end
        row && (λ[i, b] = dl)
    end
    return converged
end

# Host entry, for the first nb = size(A, 3) matrices
function hermitian_eigen_tiled!(λ::Metal.MtlMatrix{Float32}, Y::Metal.MtlArray{ComplexF32, 3},
                                A::Metal.MtlArray{ComplexF32, 3}, status::Metal.MtlVector{Int32})
    n = size(A, 1)
    nb = size(A, 3)
    W = tile_width(n)
    threads = 256
    Metal.@metal threads groups=cld(nb * W, threads) hermitian_eigen_tiled_kernel!(λ, Y, A, status, Val(W))
    return
end

# For n > 32, one SIMD group per matrix, after the warp kernel of EigenG. Lane ℓ
# owns rows (or columns) ℓ, ℓ+32, ℓ+64, …, and the tridiagonal (d, e), the
# reflector scalars τ, and the vectors v and z of each reflection are in
# threadgroup memory, one slice per SIMD group. A SIMD group then holds a
# single matrix, so its control flow is uniform, and barriers may appear
# anywhere. The scalar recurrences run redundantly on every lane, and lane 1
# stores their results, between barriers.

const SIMD_WIDTH = 32

# Orders device and threadgroup memory within the SIMD group
@inline group_sync() = simdgroup_barrier(MemoryFlagDevice | MemoryFlagThreadGroup)

# Workspace in threadgroup memory per SIMD group, for n ≤ NP, and the number G
# of SIMD groups per threadgroup that keeps it within 16 KiB
group_padded(n) = cld(n, SIMD_WIDTH) * SIMD_WIDTH
group_bytes(NP) = NP * (2sizeof(Float32) + 3sizeof(ComplexF32))
group_count(NP) = clamp(16384 ÷ group_bytes(NP), 1, 8)

@inline function group_workspace(::Val{NP}, ::Val{G}) where {NP, G}
    return (Metal.MtlThreadGroupArray(Float32, (NP, G)), Metal.MtlThreadGroupArray(Float32, (NP, G)),
            Metal.MtlThreadGroupArray(ComplexF32, (NP, G)), Metal.MtlThreadGroupArray(ComplexF32, (NP, G)),
            Metal.MtlThreadGroupArray(ComplexF32, (NP, G)))
end

# As `eigen_tile!`, for the matrix b of a SIMD group with slice g of the
# workspace `ws = (d, e, τ, v, z)`, and Z[:, :, b] for the real eigenvectors of
# the tridiagonal. Called by every lane of the SIMD group.
@inline function eigen_group!(λ, Y, A, Z, b, ws, g)
    (td, te, tτ, tv, tz) = ws
    n = size(A, 1)
    lane = Int(thread_index_in_simdgroup())
    rows = lane:SIMD_WIDTH:n
    sumg(x) = tile_sum(x, Val(SIMD_WIDTH))

    # Householder reduction, as `eigen_tile!`
    for k in 1:n-1
        s2 = 0f0
        for i in rows
            i >= k + 2 && (s2 += abs2(A[i, k, b]))
        end
        xnorm = sqrt(sumg(s2))
        α = A[k+1, k, b]
        trivial = (xnorm == 0) & (imag(α) == 0)
        β = trivial ? real(α) : -copysign(hypot32(real(α), imag(α), xnorm), real(α))
        τ = trivial ? zero(ComplexF32) : ComplexF32((β - real(α)) / β, -imag(α) / β)
        scale = trivial ? one(ComplexF32) : cinv(α - β)
        group_sync()
        for i in rows
            if i >= k + 1
                v = i == k + 1 ? one(ComplexF32) : A[i, k, b] * scale
                A[i, k, b] = v
                tv[i, g] = v
            end
        end
        if lane == 1
            te[k, g] = β
            tτ[k, g] = τ
        end
        group_sync()

        # w = τ A₂₂ v, then z = w - ½|τ|² (v† A₂₂ v) v
        dot = zero(ComplexF32)
        for i in rows
            if i >= k + 1
                x = zero(ComplexF32)
                for j in k+1:n
                    x += A[i, j, b] * tv[j, g]
                end
                w = τ * x
                tz[i, g] = w
                dot += conj(w) * tv[i, g]
            end
        end
        α2 = -(τ / 2) * sumg(dot)
        for i in rows
            i >= k + 1 && (tz[i, g] += α2 * tv[i, g])
        end
        group_sync()

        # A₂₂ ← A₂₂ - v z† - z v†, by rows
        for i in rows
            if i >= k + 1
                (vi, zi) = (tv[i, g], tz[i, g])
                for j in k+1:n
                    A[i, j, b] -= vi * conj(tz[j, g]) + zi * conj(tv[j, g])
                end
            end
        end
        group_sync()
    end
    for i in rows
        td[i, g] = real(A[i, i, b])
        for j in 1:n
            Z[i, j, b] = i == j
        end
    end
    lane == 1 && (te[n, g] = 0f0)
    group_sync()

    # Implicit QL, as `eigen_tile!`, on (d, e) in threadgroup memory. QL's
    # rotations are real, so they accumulate in the real Z. Each sweep stores
    # its rotations (c, s) in v and its new (d, e) in z, so the recurrence needs
    # no barrier, and then each lane applies the chain of rotations to its rows
    # of Z, carrying one entry between rotations. This was 1.1× faster than
    # rotating the complex Y step by step for n ≤ 64, and 1.2-1.3× for n = 128
    # and 256.
    f = 0f0
    tst1 = 0f0
    converged = true
    for l in 1:n
        tst1 = max(tst1, abs(td[l, g]) + abs(te[l, g]))
        m = l
        while m < n && abs(te[m, g]) > eps(Float32) * tst1
            m += 1
        end
        if m > l
            iter = 0
            while true
                if (iter += 1) > 30
                    converged = false
                    break
                end
                g0 = td[l, g]
                e_l = te[l, g]
                p = (td[l+1, g] - g0) / 2e_l
                r = hypot32(p, 1f0)
                r = p < 0 ? -r : r
                d_l = e_l / (p + r)
                dl1 = e_l * (p + r)
                h = g0 - d_l
                group_sync()
                if lane == 1
                    td[l, g] = d_l
                    td[l+1, g] = dl1
                end
                for i in rows
                    i >= l + 2 && (td[i, g] -= h)
                end
                f += h
                group_sync()

                p = td[m, g]
                (c, c2, c3) = (1f0, 1f0, 1f0)
                el1 = te[l+1, g]
                (s, s2) = (0f0, 0f0)
                for ii in m-1:-1:l
                    c3 = c2
                    c2 = c
                    s2 = s
                    (e_i, d_i) = (te[ii, g], td[ii, g])
                    g1 = c * e_i
                    h1 = c * p
                    r = hypot32(p, e_i)
                    e_next = s * r
                    s = e_i / r
                    c = p / r
                    p = c * d_i - s * g1
                    d_next = h1 + s * (c * g1 + s * d_i)
                    if lane == 1
                        tz[ii+1, g] = ComplexF32(d_next, e_next)
                        tv[ii, g] = ComplexF32(c, s)
                    end
                end
                p = -s * s2 * c3 * el1 * e_l / dl1
                group_sync()
                for i in rows
                    if l + 1 <= i <= m
                        td[i, g] = real(tz[i, g])
                        te[i, g] = imag(tz[i, g])
                    end
                end
                if lane == 1
                    te[l, g] = s * p
                    td[l, g] = c * p
                end
                for i in rows
                    carry = Z[i, m, b]
                    for ii in m-1:-1:l
                        cs = tv[ii, g]
                        z0 = Z[i, ii, b]
                        Z[i, ii+1, b] = imag(cs) * z0 + real(cs) * carry
                        carry = real(cs) * z0 - imag(cs) * carry
                    end
                    Z[i, l, b] = carry
                end
                group_sync()
                abs(s * p) > eps(Float32) * tst1 || break
            end
            converged || break
        end
        group_sync()
        if lane == 1
            td[l, g] += f
            te[l, g] = 0f0
        end
        group_sync()
    end

    if converged
        # Selection sort, ascending, by the first minimal index
        for ii in 1:n-1
            kmin = ii
            dmin = td[ii, g]
            for j in ii+1:n
                dj = td[j, g]
                if dj < dmin
                    dmin = dj
                    kmin = j
                end
            end
            if kmin != ii
                d_i = td[ii, g]
                group_sync()
                if lane == 1
                    td[ii, g] = dmin
                    td[kmin, g] = d_i
                end
                for i in rows
                    (Z[i, ii, b], Z[i, kmin, b]) = (Z[i, kmin, b], Z[i, ii, b])
                end
                group_sync()
            end
        end

        # Y ← Hₖ Z for k = n-1, …, 1, by columns
        for i in rows, j in 1:n
            Y[i, j, b] = Z[i, j, b]
        end
        group_sync()
        for j in rows
            for k in n-1:-1:1
                x = zero(ComplexF32)
                for i in k+1:n
                    x += conj(A[i, k, b]) * Y[i, j, b]
                end
                x *= tτ[k, g]
                for i in k+1:n
                    Y[i, j, b] -= A[i, k, b] * x
                end
            end
        end
        for i in rows
            λ[i, b] = td[i, g]
        end
    end
    group_sync()
    return converged
end

function hermitian_eigen_group_kernel!(λ, Y, A, Z, status, ::Val{NP}, ::Val{G}) where {NP, G}
    b = (Int(Metal.thread_position_in_grid_1d()) - 1) ÷ SIMD_WIDTH + 1
    ws = group_workspace(Val(NP), Val(G))
    b > size(A, 3) && return
    g = Int(Metal.simdgroup_index_in_threadgroup())
    converged = eigen_group!(λ, Y, A, Z, b, ws, g)
    thread_index_in_simdgroup() == 1 && (status[b] = converged ? STATUS_OK : STATUS_NOCONVERGE)
    return
end

function hermitian_eigen_group!(λ::Metal.MtlMatrix{Float32}, Y::Metal.MtlArray{ComplexF32, 3},
                                A::Metal.MtlArray{ComplexF32, 3}, status::Metal.MtlVector{Int32})
    n = size(A, 1)
    nb = size(A, 3)
    NP = group_padded(n)
    G = group_count(NP)
    Z = similar(λ, n, n, nb)
    Metal.@metal threads=SIMD_WIDTH*G groups=cld(nb, G) hermitian_eigen_group_kernel!(λ, Y, A, Z, status, Val(NP), Val(G))
    Metal.unsafe_free!(Z)
    return
end
