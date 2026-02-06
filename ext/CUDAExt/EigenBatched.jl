using LinearAlgebra: BlasFloat, checksquare, BlasInt
using LinearAlgebra.LAPACK: chkuplo, chkargsok

using CUDA: unsafe_free!, @allowscalar, with_workspaces
using CUDA.CUBLAS: StridedCuMatrix, unsafe_strided_batch, handle, cublasZtrsmBatched_64, cublasCtrsmBatched_64
using CUDA.CUSOLVER: dense_handle, CuSolverParameters, cusolverDnZpotrfBatched, cusolverDnCpotrfBatched, cusolverDnXsyevBatched_bufferSize, cusolverDnXsyevBatched
## (TR) triangular triangular matrix solution batched
for (fname, elty) in ((:cublasZtrsmBatched_64, :ComplexF64),
                      (:cublasCtrsmBatched_64, :ComplexF32))
    @eval begin
        function trsm_batched!(side::Char,
                               uplo::Char,
                               transa::Char,
                               diag::Char,
                               alpha,
                               m,
                               n,
                               lda,
                               ldb,
                               A::CuArray{CuPtr{$elty}, 1},
                               B::CuArray{CuPtr{$elty}, 1})
            if length(A) != length(B)
                throw(DimensionMismatch(""))
            end

            $fname(handle(), side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb, length(A))
            B
        end
    end
end

for (fname, elty) in ((:cusolverDnCpotrfBatched, :ComplexF32),
                      (:cusolverDnZpotrfBatched, :ComplexF64))
    @eval begin
        function potrfBatched!(uplo::Char, n, lda, A::CuArray{CuPtr{$elty}, 1})

            # Set up information for the solver arguments
            chkuplo(uplo)
            batchSize = length(A)

            dh = dense_handle()
            resize!(dh.info, batchSize)

            # Run the solver
            $fname(dh, uplo, n, A, lda, dh.info, batchSize)

            # Copy the solver info and delete the device memory
            info = @allowscalar collect(dh.info)

            # Double check the solver's exit status
            for i = 1:batchSize
                chkargsok(BlasInt(info[i]))
            end

            # info[i] > 0 means the leading minor of order info[i] is not positive definite
            # LinearAlgebra.LAPACK does not throw Exception here
            # to simplify calls to isposdef! and factorize
            return A, info
        end
    end
end

# XsyevBatched
function XsyevBatched!(jobz::Char, uplo::Char, A::StridedCuArray{T, 3}) where {T <: BlasFloat}
    minimum_version = v"11.7.1"
    CUSOLVER.version() < minimum_version && throw(ErrorException("This operation requires cuSOLVER
        $(minimum_version) or later. Current cuSOLVER version: $(CUSOLVER.version())."))
    chkuplo(uplo)
    n = checksquare(A)
    batch_size = size(A, 3)
    R = real(T)
    lda = max(1, stride(A, 2))
    W = CuMatrix{R}(undef, n, batch_size)
    params = CuSolverParameters()
    dh = dense_handle()
    resize!(dh.info, batch_size)

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsyevBatched_bufferSize(
            dh, params, jobz, uplo, n,
            T, A, lda, R, W, T, out_gpu, out_cpu, batch_size
        )
        return out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsyevBatched(
            dh, params, jobz, uplo, n, T, A,
            lda, R, W, T, buffer_gpu, sizeof(buffer_gpu),
            buffer_cpu, sizeof(buffer_cpu), dh.info, batch_size
        )
    end

    info = @allowscalar collect(dh.info)
    for i in 1:batch_size
        chkargsok(info[i] |> BlasInt)
    end

    if jobz == 'N'
        return W
    elseif jobz == 'V'
        return W, A
    end
end

function eigenbatched!(H_d, I_d)
    m = size(H_d, 1)
    n = size(H_d, 2)
    lda = max(1,stride(H_d, 2))
    ldb = max(1,stride(I_d, 2))
    H_dp = unsafe_strided_batch(H_d)
    I_dp = unsafe_strided_batch(I_d)
    potrfBatched!('L', n, lda, H_dp)
    trsm_batched!('R', 'L', 'C', 'N', ComplexF64(1.), m, n, lda, ldb, H_dp, I_dp)
    trsm_batched!('L', 'L', 'N', 'N', ComplexF64(1.), m, n, lda, ldb, H_dp, I_dp)
    #evalues_d , _ = CUSOLVER.heevjBatched!('V', 'L', I_d)
    evalues_d , _ = XsyevBatched!('V', 'L', I_d)
    trsm_batched!('L', 'L', 'C', 'N', ComplexF64(1.), m, n, lda, ldb, H_dp, I_dp)
    unsafe_free!(H_dp)
    unsafe_free!(I_dp)
    return evalues_d
end
