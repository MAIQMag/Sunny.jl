# AMDGPU-specific hooks for KAExt, with batched linear algebra from rocSOLVER
module KAAMDGPUExt

using AMDGPU: AMDGPU, ROCArray, ROCMatrix, ROCVector, rocBLAS, rocSOLVER
using Sunny: Sunny

Sunny.device_free_memory(::AMDGPU.ROCBackend) = AMDGPU.free()

# Batched generalized Hermitian eigensolvers of rocSOLVER, per precision: hegvd,
# with divide and conquer, and hegvj, with Jacobi rotations
hegvd_strided_batched(::Type{Float64}) = rocSOLVER.rocsolver_zhegvd_strided_batched
hegvd_strided_batched(::Type{Float32}) = rocSOLVER.rocsolver_chegvd_strided_batched
hegvj_strided_batched(::Type{Float64}) = rocSOLVER.rocsolver_zhegvj_strided_batched
hegvj_strided_batched(::Type{Float32}) = rocSOLVER.rocsolver_chegvj_strided_batched

# The largest n for which hegvj is used in place of hegvd. For batches of 100k
# matrices in Float64, hegvj was faster than hegvd by 4.1×, 2.9× and 1.3× for
# n = 8, 16 and 40 on an MI100, and by 3.9×, 2.7× and 1.8× on an MI300A. For
# n = 64, it was 1.3× slower on the MI100, and as fast on the MI300A. Its errors
# in the energies were larger, but small: about 1e-14 vs 2e-15 in Float64, and
# 6e-6 vs 1e-6 in Float32.
const HEGVJ_MAX_N = 40

# Sweeps of hegvj before it reports no convergence
const HEGVJ_MAX_SWEEPS = Int32(100)

# Batched variant of `Sunny.bogoliubov!` for the first `nb` Hamiltonians of
# `H`. rocSOLVER's hegvd or hegvj solves the generalized eigenproblem
# Ĩ t = λ H t in one call, as LAPACK's zhegvd does for `Sunny.bogoliubov!`, with
# eigenvalues in ascending order and t† H t = 1. Its `info` per matrix is n + j
# if H is not positive definite at the leading minor j, and in 1:n if the
# eigensolver fails to converge. These become the status codes of KAExt: 0 on
# success, -1 for non-finite H, -2 for unexpected inertia, -3 for no
# convergence, and j > 0 if the Cholesky factorization fails at column j. On
# exit, H is overwritten.
function Sunny.vendor_bogoliubov!(T::ROCArray{Complex{F}, 3}, E::ROCArray{F, 2}, H::ROCArray{Complex{F}, 3}, nb,
                                  status::ROCArray{Int32, 1}) where F <: Union{Float32, Float64}
    n = size(H, 1)
    L = n ÷ 2
    if !AMDGPU.functional(:rocsolver)
        error("rocSOLVER is unavailable; use `to_device(swt; eigensolver=:jacobi)`")
    end
    Hb = view(H, :, :, 1:nb)
    Tb = view(T, :, :, 1:nb)
    sb = view(status, 1:nb)

    finite = vec(all(isfinite, Hb; dims=(1, 2)))
    if !all(finite)
        sb .= ifelse.(finite, Int32(0), Int32(-1))
        return
    end

    # Ĩ = diag(1, …, 1, -1, …, -1), overwritten by the eigenvectors t
    fill!(Tb, 0)
    view(reshape(Tb, n^2, nb), 1:n+1:n^2, :) .= ROCArray(Complex{F}[ones(L); -ones(L)])

    # The `info` of rocSOLVER is an rocblas_int, i.e. Int32, as is the status,
    # whose first nb entries it fills
    λ = ROCMatrix{F}(undef, n, nb)
    try
        if n <= HEGVJ_MAX_N
            hegvj!(T, H, λ, status, nb)
        else
            hegvd!(T, H, λ, status, nb)
        end
        sb .= ifelse.(sb .> n, sb .- Int32(n), ifelse.(sb .> 0, Int32(-3), sb))
        Sunny.finish_vendor_bogoliubov!(T, E, λ, nb, status)
    finally
        AMDGPU.unsafe_free!(λ)
    end
    return
end

function hegvd!(T::ROCArray{Complex{F}, 3}, H, λ, status, nb) where F
    n = size(H, 1)
    # The off-diagonal of the intermediate tridiagonal form
    offdiag = ROCMatrix{F}(undef, n, nb)
    try
        hegvd_strided_batched(F)(rocBLAS.handle(), rocSOLVER.rocblas_eform_ax, rocSOLVER.rocblas_evect_original,
                                 rocBLAS.rocblas_fill_lower, n, T, n, n^2, H, n, n^2, λ, n, offdiag, n, status, nb)
    finally
        AMDGPU.unsafe_free!(offdiag)
    end
end

function hegvj!(T::ROCArray{Complex{F}, 3}, H, λ, status, nb) where F
    n = size(H, 1)
    # The residual and number of sweeps per matrix, which are not used
    residual = ROCVector{F}(undef, nb)
    n_sweeps = ROCVector{Int32}(undef, nb)
    try
        # abstol = 0 selects rocSOLVER's default tolerance
        hegvj_strided_batched(F)(rocBLAS.handle(), rocSOLVER.rocblas_eform_ax, rocSOLVER.rocblas_evect_original,
                                 rocBLAS.rocblas_fill_lower, n, T, n, n^2, H, n, n^2, zero(F), residual,
                                 HEGVJ_MAX_SWEEPS, n_sweeps, λ, n, status, nb)
    finally
        AMDGPU.unsafe_free!(residual)
        AMDGPU.unsafe_free!(n_sweeps)
    end
end

end
