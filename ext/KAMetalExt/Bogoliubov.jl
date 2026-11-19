# The batched Bogoliubov transformation of `Sunny.vendor_bogoliubov!` on Metal,
# in Float32 for n ≤ 256, with one tile per Hamiltonian for n ≤ 32, and one
# SIMD group per Hamiltonian beyond, as the scalar
# `bogoliubov_householder!` of test/test_householder_ql.jl. The generalized
# problem Ĩ t = λ H t is reduced by the Cholesky factor H = K K† to the
# Hermitian A = K⁻¹ Ĩ K⁻†, whose eigenvectors y give t = K⁻† y, with t† H t = 1.
#
# Cholesky and the eigensolver work by rows, with one lane per row, and the
# triangular solves by columns, with one lane per column. Between them, a
# barrier makes each lane's stores visible to the others. A barrier must be
# reached by the whole SIMD group, so every one is at the top level of the
# kernel, and a tile without a Hamiltonian, or whose Hamiltonian has failed,
# skips the work between barriers but not the barriers.

const STATUS_NONFINITE = Int32(-1)

# The largest n, with the workspace of a single SIMD group, 8 KiB of threadgroup
# memory, within the 32 KiB of a threadgroup
const MAX_N = 256
Sunny.vendor_bogoliubov_max_size(::Metal.MetalBackend) = MAX_N

# On entry H[:, :, b] for b ≤ nb. On exit, T[:, :, b] holds the t and E[:, b]
# the λ ascending, as `Sunny.finish_vendor_bogoliubov!` expects, where the
# status is STATUS_OK. H is overwritten by K in its lower triangle, and the
# workspace A by the reflectors.
function bogoliubov_tiled_kernel!(T, E, H, A, status, nb, ::Val{W}) where W
    n = size(H, 1)
    L = n ÷ 2
    b0 = (Int(Metal.thread_position_in_grid_1d()) - 1) ÷ W + 1
    active = b0 <= nb
    b = min(b0, nb)        # A valid matrix for addressing; stores need `active`
    lane = tile_lane(Val(W))
    own = active & (lane <= n)
    i = min(lane, n)       # The row or column of the lane, if `own`
    bc(x, root) = tile_bcast(x, root, Val(W))
    st = active ? STATUS_OK : Int32(1)

    # A non-finite H would not terminate QL
    bad = false
    if own
        for j in 1:n
            bad |= !isfinite(H[i, j, b])
        end
    end
    tile_max(Int32(bad), Val(W)) > 0 && (st = STATUS_NONFINITE)

    # Lower Cholesky factor K in H, by rows. The entries H[j, k] of the pivot
    # row arrive by broadcast from lane j, which owns them.
    if st == STATUS_OK
        for j in 1:n
            x = own ? H[i, j, b] : zero(ComplexF32)
            for k in 1:j-1
                hik = own ? H[i, k, b] : zero(ComplexF32)
                x -= hik * conj(bc(hik, j))
            end
            dj = real(bc(x, j))
            if !(dj > 0)
                st = Int32(j)
                break
            end
            dj = sqrt(dj)
            if own
                lane == j && (H[i, j, b] = dj)
                lane > j && (H[i, j, b] = x / dj)
            end
        end
    end
    tile_sync()

    # B = K⁻¹ Ĩ in T, by columns, by forward substitution. Column c of B is
    # zero above row c.
    ok = own & (st == STATUS_OK)
    if ok
        c = i
        for r in 1:c-1
            T[r, c, b] = 0
        end
        for r in c:n
            x = r == c ? ComplexF32(c <= L ? 1 : -1) : zero(ComplexF32)
            for k in c:r-1
                x -= H[r, k, b] * T[k, c, b]
            end
            T[r, c, b] = x / real(H[r, r, b])
        end
    end
    tile_sync()

    # A = K⁻¹ B†, by columns, which is A itself since A is Hermitian
    if ok
        c = i
        for r in 1:n
            x = conj(T[c, r, b])
            for k in 1:r-1
                x -= H[r, k, b] * A[k, c, b]
            end
            A[r, c, b] = x / real(H[r, r, b])
        end
    end
    tile_sync()

    # Exactly Hermitian, from the lower triangle, by rows. Each lane stores
    # into the upper triangle of its row and loads from the lower triangle of
    # others, which no lane stores into.
    if ok
        for j in i+1:n
            A[i, j, b] = conj(A[j, i, b])
        end
        A[i, i, b] = real(A[i, i, b])
    end
    tile_sync()

    # Eigenvalues λ into E and eigenvectors y into T, by rows. A failed tile
    # skips this uniformly, since `st` agrees on every lane of a tile.
    if active & (st == STATUS_OK)
        eigen_tile!(E, T, A, b, own, Val(W)) || (st = STATUS_NOCONVERGE)
    end
    tile_sync()

    # t = K⁻† y in T, by columns, by back substitution
    if own & (st == STATUS_OK)
        c = i
        for r in n:-1:1
            x = T[r, c, b]
            for k in r+1:n
                x -= conj(H[k, r, b]) * T[k, c, b]
            end
            T[r, c, b] = x / real(H[r, r, b])
        end
    end

    (active & (lane == 1)) && (status[b] = st)
    return
end

# As `bogoliubov_tiled_kernel!`, for n > 32 with one SIMD group per
# Hamiltonian, and the workspaces of `eigen_group!`, including the real Z. Lane
# ℓ owns rows or columns ℓ, ℓ+32, …. Its control flow is uniform over the SIMD
# group, so barriers may appear anywhere, and the pivots of Cholesky are
# computed redundantly by every lane from rows in device memory.
function bogoliubov_group_kernel!(T, E, H, A, Z, status, nb, ::Val{NP}, ::Val{G}) where {NP, G}
    n = size(H, 1)
    L = n ÷ 2
    b = (Int(Metal.thread_position_in_grid_1d()) - 1) ÷ SIMD_WIDTH + 1
    ws = group_workspace(Val(NP), Val(G))
    b > nb && return
    g = Int(Metal.simdgroup_index_in_threadgroup())
    lane = Int(thread_index_in_simdgroup())
    own = lane:SIMD_WIDTH:n
    st = STATUS_OK

    bad = false
    for i in own, j in 1:n
        bad |= !isfinite(H[i, j, b])
    end
    tile_max(Int32(bad), Val(SIMD_WIDTH)) > 0 && (st = STATUS_NONFINITE)

    # Lower Cholesky factor K in H, by rows
    if st == STATUS_OK
        for j in 1:n
            dj = real(H[j, j, b])
            for k in 1:j-1
                dj -= abs2(H[j, k, b])
            end
            if !(dj > 0)
                st = Int32(j)
                break
            end
            dj = sqrt(dj)
            for i in own
                if i > j
                    x = H[i, j, b]
                    for k in 1:j-1
                        x -= H[i, k, b] * conj(H[j, k, b])
                    end
                    H[i, j, b] = x / dj
                end
            end
            group_sync()
            lane == 1 && (H[j, j, b] = dj)
            group_sync()
        end
    end

    if st == STATUS_OK
        # B = K⁻¹ Ĩ in T, by columns
        for c in own
            for r in 1:c-1
                T[r, c, b] = 0
            end
            for r in c:n
                x = r == c ? ComplexF32(c <= L ? 1 : -1) : zero(ComplexF32)
                for k in c:r-1
                    x -= H[r, k, b] * T[k, c, b]
                end
                T[r, c, b] = x / real(H[r, r, b])
            end
        end
        group_sync()

        # A = K⁻¹ B†, by columns
        for c in own
            for r in 1:n
                x = conj(T[c, r, b])
                for k in 1:r-1
                    x -= H[r, k, b] * A[k, c, b]
                end
                A[r, c, b] = x / real(H[r, r, b])
            end
        end
        group_sync()

        # Exactly Hermitian, from the lower triangle, by rows
        for i in own
            for j in i+1:n
                A[i, j, b] = conj(A[j, i, b])
            end
            A[i, i, b] = real(A[i, i, b])
        end
        group_sync()

        eigen_group!(E, T, A, Z, b, ws, g) || (st = STATUS_NOCONVERGE)
    end

    # t = K⁻† y in T, by columns
    if st == STATUS_OK
        for c in own
            for r in n:-1:1
                x = T[r, c, b]
                for k in r+1:n
                    x -= conj(H[k, r, b]) * T[k, c, b]
                end
                T[r, c, b] = x / real(H[r, r, b])
            end
        end
    end

    lane == 1 && (status[b] = st)
    return
end

# The energies of the finished T, recomputed as Rayleigh quotients of the
# original H, ε = (t† H t) / (t† Ĩ t) for each column t, with one thread per
# column. The quotient is stationary at an eigenvector, so the O(δ) error of a
# t gives an O(δ²) error in ε. In Float32, for the √3×√3 kagome model at the
# device regularization, the relative error of energies above 5% of the largest
# fell from 3.0e-3 to 1.9e-4. Energies near zero remain limited by the
# cancellation in t† H t.
function rayleigh_energies_kernel!(E, T, H, status, nb)
    n = size(T, 1)
    L = n ÷ 2
    idx = Int(Metal.thread_position_in_grid_1d())
    (j, b) = (mod1(idx, n), cld(idx, n))
    (b > nb || status[b] != STATUS_OK) && return
    num = zero(ComplexF32)
    den = 0f0
    for k in 1:n
        x = zero(ComplexF32)
        for i in 1:n
            x += H[k, i, b] * T[i, j, b]
        end
        tk = T[k, j, b]
        num += conj(tk) * x
        den += ifelse(k <= L, 1f0, -1f0) * abs2(tk)
    end
    E[j, b] = real(num) / den
    return
end

function Sunny.vendor_bogoliubov!(T::Metal.MtlArray{ComplexF32, 3}, E::Metal.MtlArray{Float32, 2},
                                  H::Metal.MtlArray{ComplexF32, 3}, nb, status::Metal.MtlArray{Int32, 1})
    n = size(H, 1)
    n <= MAX_N || error("Metal eigensolver supports n ≤ $MAX_N, got $n; use `to_device(swt; eigensolver=:host)`")
    # The kernels overwrite H with its Cholesky factor
    H0 = H[:, :, 1:nb]
    A = similar(H, n, n, nb)
    if n <= SIMD_WIDTH
        W = tile_width(n)
        threads = 256
        Metal.@metal threads groups=cld(nb * W, threads) bogoliubov_tiled_kernel!(T, E, H, A, status, nb, Val(W))
    else
        NP = group_padded(n)
        G = group_count(NP)
        Z = similar(E, n, n, nb)
        Metal.@metal threads=SIMD_WIDTH*G groups=cld(nb, G) bogoliubov_group_kernel!(T, E, H, A, Z, status, nb, Val(NP), Val(G))
        Metal.unsafe_free!(Z)
    end
    # The λ in E are copied before E is overwritten by the energies
    Sunny.finish_vendor_bogoliubov!(T, E, E, nb, status)
    threads = 256
    Metal.@metal threads groups=cld(n * nb, threads) rayleigh_energies_kernel!(E, T, H0, status, nb)
    Metal.unsafe_free!(A)
    Metal.unsafe_free!(H0)
    return
end
