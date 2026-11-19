# The batched Bogoliubov transformation of `Sunny.vendor_bogoliubov!` on Metal,
# in Float32 for n ≤ 32, with one tile per Hamiltonian, as the scalar
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

# The largest n of the tiled kernels
const MAX_N = 32
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

function Sunny.vendor_bogoliubov!(T::Metal.MtlArray{ComplexF32, 3}, E::Metal.MtlArray{Float32, 2},
                                  H::Metal.MtlArray{ComplexF32, 3}, nb, status::Metal.MtlArray{Int32, 1})
    n = size(H, 1)
    n <= MAX_N || error("Metal eigensolver supports n ≤ $MAX_N, got $n; use `to_device(swt; eigensolver=:host)`")
    W = tile_width(n)
    A = similar(H, n, n, nb)
    threads = 256
    Metal.@metal threads groups=cld(nb * W, threads) bogoliubov_tiled_kernel!(T, E, H, A, status, nb, Val(W))
    # The λ in E are copied before E is overwritten by the energies
    Sunny.finish_vendor_bogoliubov!(T, E, E, nb, status)
    Metal.unsafe_free!(A)
    return
end
