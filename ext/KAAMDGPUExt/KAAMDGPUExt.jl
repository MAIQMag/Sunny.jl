# AMDGPU-specific hooks for KAExt, with batched linear algebra from rocSOLVER
module KAAMDGPUExt

using AMDGPU: AMDGPU, ROCArray, ROCMatrix, rocBLAS, rocSOLVER
using Sunny: Sunny

Sunny.device_free_memory(::AMDGPU.ROCBackend) = AMDGPU.free()

# Batched generalized Hermitian eigensolvers of rocSOLVER, per precision
hegvd_strided_batched(::Type{Float64}) = rocSOLVER.rocsolver_zhegvd_strided_batched
hegvd_strided_batched(::Type{Float32}) = rocSOLVER.rocsolver_chegvd_strided_batched

# Batched variant of `Sunny.bogoliubov!` for the first `nb` Hamiltonians of
# `H`. rocSOLVER's hegvd solves the generalized eigenproblem Ĩ t = λ H t in one
# call, as LAPACK's zhegvd does for `Sunny.bogoliubov!`, with eigenvalues in
# ascending order and t† H t = 1. Its `info` per matrix is n + j if H is not
# positive definite at the leading minor j, and in 1:n if the eigensolver fails
# to converge. These become the status codes of KAExt: 0 on success, -1 for
# non-finite H, -2 for unexpected inertia, -3 for no convergence, and j > 0 if
# the Cholesky factorization fails at column j. On exit, H is overwritten.
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

    # Eigenvalues λ, and the off-diagonal of the intermediate tridiagonal form.
    # The `info` of rocSOLVER is an rocblas_int, i.e. Int32, as is the status,
    # whose first nb entries it fills.
    λ = ROCMatrix{F}(undef, n, nb)
    offdiag = ROCMatrix{F}(undef, n, nb)
    try
        hegvd_strided_batched(F)(rocBLAS.handle(), rocSOLVER.rocblas_eform_ax, rocSOLVER.rocblas_evect_original,
                                 rocBLAS.rocblas_fill_lower, n, T, n, n^2, H, n, n^2, λ, n, offdiag, n, status, nb)
        sb .= ifelse.(sb .> n, sb .- Int32(n), ifelse.(sb .> 0, Int32(-3), sb))
        Sunny.finish_vendor_bogoliubov!(T, E, λ, nb, status)
    finally
        AMDGPU.unsafe_free!(λ)
        AMDGPU.unsafe_free!(offdiag)
    end
    return
end

end
