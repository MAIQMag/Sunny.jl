# Vendor-specific batched linear algebra (cuSOLVER, cuBLAS) for KAExt
module KACUDAExt

using CUDA: CUDA, CuArray, CuMatrix, CUBLAS, CUSOLVER
using Sunny: Sunny

Sunny.device_free_memory(::CUDA.CUDABackend) = CUDA.free_memory()

# Workspace in bytes of `CUSOLVER.XsyevBatched!` per n×n matrix, which is
# roughly independent of the batch size
function syev_workspace_per_matrix(n)
    A = CuArray{ComplexF64}(undef, n, n, 1)
    W = CuMatrix{Float64}(undef, n, 1)
    (gpu, cpu) = (Ref{Csize_t}(0), Ref{Csize_t}(0))
    CUSOLVER.cusolverDnXsyevBatched_bufferSize(CUSOLVER.dense_handle(), CUSOLVER.CuSolverParameters(), 'V', 'L', n,
                                               ComplexF64, A, n, Float64, W, ComplexF64, gpu, cpu, 1)
    return Int(gpu[])
end

# Batched variant of `Sunny.bogoliubov!` for the first `nb` Hamiltonians of
# `H`, by the same generalized eigenproblem Ĩ t = λ H t that LAPACK's zhegvd
# solves. The Cholesky factor H = K K† reduces it to the Hermitian A = K⁻¹ Ĩ
# K⁻†, whose eigenvectors y give t = K⁻† y. The status codes per wavevector are
# those of KAExt: 0 on success, -1 for non-finite H, -2 for unexpected inertia,
# and j > 0 if the Cholesky factorization fails at column j. On exit, H is
# overwritten.
function Sunny.vendor_bogoliubov!(T::CuArray{ComplexF64, 3}, E::CuArray{Float64, 2}, H::CuArray{ComplexF64, 3}, nb, status::CuArray{Int32, 1})
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
        CUSOLVER.cusolverDnZpotrfBatched(dh, 'L', n, Hp, n, dh.info, nb)
        copyto!(status, view(dh.info, 1:nb))
        any(!iszero, status) && return

        # Ĩ = diag(1, …, 1, -1, …, -1), then A = K⁻¹ Ĩ K⁻†
        trsm!(side, trans) = CUBLAS.cublasZtrsmBatched_64(CUBLAS.handle(), side, 'L', trans, 'N', n, n, one(ComplexF64), Hp, n, Tp, n, nb)
        fill!(Tb, 0)
        view(reshape(Tb, n^2, nb), 1:n+1:n^2, :) .= CuArray([ones(ComplexF64, L); -ones(ComplexF64, L)])
        trsm!('R', 'C')
        trsm!('L', 'N')

        # Eigenvalues ascending and eigenvectors y in T. The workspace of
        # XsyevBatched, about a megabyte per matrix, bounds the sub-batches.
        λ = CuMatrix{Float64}(undef, n, nb)
        chunk = clamp(CUDA.free_memory() ÷ 2syev_workspace_per_matrix(n), 1, nb)
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
