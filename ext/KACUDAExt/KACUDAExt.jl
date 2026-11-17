# Vendor-specific batched linear algebra (cuSOLVER, cuBLAS) for KAExt
module KACUDAExt

using CUDA: CUDA, CuArray, CuMatrix, CUBLAS, CUSOLVER
using Sunny: Sunny

Sunny.device_free_memory(::CUDA.CUDABackend) = CUDA.free_memory()

# Batched Cholesky factorization and triangular solves, per precision
potrf_batched(::Type{Float64}) = CUSOLVER.cusolverDnZpotrfBatched
potrf_batched(::Type{Float32}) = CUSOLVER.cusolverDnCpotrfBatched
trsm_batched(::Type{Float64}) = CUBLAS.cublasZtrsmBatched_64
trsm_batched(::Type{Float32}) = CUBLAS.cublasCtrsmBatched_64

# Workspace in bytes of `CUSOLVER.XsyevBatched!` per n×n matrix in precision F,
# which is roughly independent of the batch size
function syev_workspace_per_matrix(n, F)
    A = CuArray{Complex{F}}(undef, n, n, 1)
    W = CuMatrix{F}(undef, n, 1)
    (gpu, cpu) = (Ref{Csize_t}(0), Ref{Csize_t}(0))
    CUSOLVER.cusolverDnXsyevBatched_bufferSize(CUSOLVER.dense_handle(), CUSOLVER.CuSolverParameters(), 'V', 'L', n,
                                               Complex{F}, A, n, F, W, Complex{F}, gpu, cpu, 1)
    return Int(gpu[])
end

# Batched variant of `Sunny.bogoliubov!` for the first `nb` Hamiltonians of
# `H`, in Float64 or Float32, by the same generalized eigenproblem Ĩ t = λ H t
# that LAPACK's zhegvd solves. The Cholesky factor H = K K† reduces it to the
# Hermitian A = K⁻¹ Ĩ K⁻†, whose eigenvectors y give t = K⁻† y. The status
# codes per wavevector are those of KAExt: 0 on success, -1 for non-finite H,
# -2 for unexpected inertia, and j > 0 if the Cholesky factorization fails at
# column j. On exit, H is overwritten.
function Sunny.vendor_bogoliubov!(T::CuArray{Complex{F}, 3}, E::CuArray{F, 2}, H::CuArray{Complex{F}, 3}, nb,
                                  status::CuArray{Int32, 1}) where F <: Union{Float32, Float64}
    n = size(H, 1)
    L = n ÷ 2
    if CUSOLVER.version() < v"11.7.1"
        error("cuSOLVER $(CUSOLVER.version()) lacks XsyevBatched; use `to_device(swt; eigensolver=:jacobi)`")
    end
    Hb = view(H, :, :, 1:nb)
    Tb = view(T, :, :, 1:nb)

    finite = vec(all(isfinite, Hb; dims=(1, 2)))
    if !all(finite)
        status .= ifelse.(finite, Int32(0), Int32(-1))
        return
    end

    # Pointers to the matrices of H and T, shared by the batched calls below,
    # which act on the first nb
    Hp = CUBLAS.unsafe_strided_batch(H)
    Tp = CUBLAS.unsafe_strided_batch(T)
    try
        dh = CUSOLVER.dense_handle()
        resize!(dh.info, nb)
        potrf_batched(F)(dh, 'L', n, Hp, n, dh.info, nb)
        copyto!(status, view(dh.info, 1:nb))
        any(!iszero, status) && return

        # Ĩ = diag(1, …, 1, -1, …, -1), then A = K⁻¹ Ĩ K⁻†
        trsm!(side, trans) = trsm_batched(F)(CUBLAS.handle(), side, 'L', trans, 'N', n, n, one(Complex{F}), Hp, n, Tp, n, nb)
        fill!(Tb, 0)
        view(reshape(Tb, n^2, nb), 1:n+1:n^2, :) .= CuArray(Complex{F}[ones(L); -ones(L)])
        trsm!('R', 'C')
        trsm!('L', 'N')

        # Eigenvalues ascending and eigenvectors y in T. The workspace of
        # XsyevBatched, about a megabyte per matrix, bounds the sub-batches.
        λ = CuMatrix{F}(undef, n, nb)
        chunk = clamp(CUDA.free_memory() ÷ 2syev_workspace_per_matrix(n, F), 1, nb)
        for b0 in 0:chunk:nb-1
            bs = b0+1:min(b0+chunk, nb)
            (λs, _) = CUSOLVER.XsyevBatched!('V', 'L', view(T, :, :, bs))
            view(λ, :, bs) .= λs
            CUDA.unsafe_free!(λs)
        end

        # t = K⁻† y, then the inertia check, order, normalization and energies
        # shared with other vendors
        trsm!('L', 'C')
        Sunny.finish_vendor_bogoliubov!(T, E, λ, nb, status)
    finally
        CUDA.unsafe_free!(Hp)
        CUDA.unsafe_free!(Tp)
    end
    return
end

end
