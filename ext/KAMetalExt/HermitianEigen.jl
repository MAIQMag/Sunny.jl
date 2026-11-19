# A batched Hermitian eigensolver in Float32 for matrices of size n ≤ 32, with
# one tile of W ≥ n lanes per matrix, after the tiled kernels of EigenG-Batched
# (RIKEN) for CUDA and HIP: Householder reduction to tridiagonal form, implicit
# QL, and back-transformation. EigenG is real symmetric. Here the reduction is
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
    row = lane <= n
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
    lane == 1 && (status[b] = converged ? STATUS_OK : STATUS_NOCONVERGE)
    return
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
