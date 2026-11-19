# A scalar reference for a batched Hermitian eigensolver on Metal, after the
# real symmetric EigenG-Batched (RIKEN) of CUDA and HIP, which reduces to
# tridiagonal form by Householder reflections (EISPACK tred1), diagonalizes the
# tridiagonal by implicit QL (tql2), and back-transforms the eigenvectors
# (trbak1). Sunny's Hamiltonians are complex, so the reduction here is the
# complex one of LAPACK's zhetd2, with reflectors of zlarfg that leave a real
# tridiagonal for an unchanged tql2. The loops mirror the work of a SIMD tile,
# with one lane per row or column where marked "lanes", so that this module can
# serve as the oracle of the device kernels.
@testmodule HouseholderQL begin
    using Sunny, LinearAlgebra
    # Loads KAExt, for `Sunny.finish_vendor_bogoliubov!`
    import KernelAbstractions

    # The reflector H = I - τ v v†, with v[1] = 1, for which H† x = β e₁ with
    # β real, as LAPACK's zlarfg. Overwrites x[2:end] with v[2:end] and returns
    # (β, τ). A real x[1] and zero x[2:end] give τ = 0, i.e., H = I.
    function reflector!(x::AbstractVector{C}) where C <: Complex
        α = x[1]
        xnorm = zero(real(C))
        for i in 2:length(x)           # lanes, then a tile sum
            xnorm += abs2(x[i])
        end
        xnorm = sqrt(xnorm)
        (xnorm == 0 && imag(α) == 0) && return (real(α), zero(C))
        β = -copysign(hypot(real(α), imag(α), xnorm), real(α))
        τ = complex((β - real(α)) / β, -imag(α) / β)
        s = 1 / (α - β)
        for i in 2:length(x)           # lanes
            x[i] *= s
        end
        return (β, τ)
    end

    # Reduces the Hermitian A, both triangles stored, to the real tridiagonal T
    # = Q† A Q, with diagonal d and subdiagonal e[1:n-1], and e[n] = 0. The
    # reflector k, acting on rows k+1:n, is stored as v in A[k+1:n, k] and τ[k],
    # so that Q = H₁ H₂ ⋯ Hₙ₋₁.
    function tridiagonalize!(A::AbstractMatrix{C}, d, e, τ) where C <: Complex
        n = size(A, 1)
        w = zeros(C, n)
        for k in 1:n-1
            r = k+1:n
            (β, t) = reflector!(view(A, r, k))
            A[k+1, k] = 1
            e[k] = β
            τ[k] = t
            # w = τ A₂₂ v, then z = w - ½|τ|² (v† A₂₂ v) v, and the rank-2
            # update A₂₂ ← H† A₂₂ H = A₂₂ - v z† - z v†
            for i in r                 # lanes over rows
                x = zero(C)
                for j in r
                    x += A[i, j] * A[j, k]
                end
                w[i] = t * x
            end
            α = zero(C)
            for i in r                 # lanes, then a tile sum
                α += conj(w[i]) * A[i, k]
            end
            α *= -t / 2
            for i in r                 # lanes
                w[i] += α * A[i, k]
            end
            for j in r, i in r         # lanes over rows
                A[i, j] -= A[i, k] * conj(w[j]) + w[i] * conj(A[j, k])
            end
        end
        for k in 1:n
            d[k] = real(A[k, k])
        end
        e[n] = 0
        return
    end

    # Implicit QL iteration on the symmetric tridiagonal (d, e), with e[n] = 0,
    # of EISPACK tql2 as in JAMA. Eigenvalues overwrite d in ascending order,
    # and the rotations accumulate in the columns of Z, which are permuted to
    # match. Returns false if some eigenvalue fails to converge in `maxiter`
    # iterations. The scalar recurrence runs redundantly on every lane, and
    # only the rotation of Z distributes over rows.
    function tql2!(d::AbstractVector{R}, e::AbstractVector{R}, Z::AbstractMatrix{R}; maxiter=30) where R <: Real
        n = length(d)
        f = zero(R)
        tst1 = zero(R)
        for l in 1:n
            tst1 = max(tst1, abs(d[l]) + abs(e[l]))
            m = l
            while m < n && abs(e[m]) > eps(R) * tst1
                m += 1
            end
            if m > l
                iter = 0
                while true
                    (iter += 1) > maxiter && return false
                    # Shift from the leading 2×2 block
                    g = d[l]
                    p = (d[l+1] - g) / 2e[l]
                    r = hypot(p, one(R))
                    r = p < 0 ? -r : r
                    d[l] = e[l] / (p + r)
                    d[l+1] = e[l] * (p + r)
                    dl1 = d[l+1]
                    h = g - d[l]
                    for i in l+2:n
                        d[i] -= h
                    end
                    f += h
                    # Implicit QL transformation, chasing the bulge from m to l
                    p = d[m]
                    (c, c2, c3) = (one(R), one(R), one(R))
                    el1 = e[l+1]
                    (s, s2) = (zero(R), zero(R))
                    for i in m-1:-1:l
                        c3 = c2
                        c2 = c
                        s2 = s
                        g = c * e[i]
                        h = c * p
                        r = hypot(p, e[i])
                        e[i+1] = s * r
                        s = e[i] / r
                        c = p / r
                        p = c * d[i] - s * g
                        d[i+1] = h + s * (c * g + s * d[i])
                        for k in axes(Z, 1)    # lanes over rows
                            h = Z[k, i+1]
                            Z[k, i+1] = s * Z[k, i] + c * h
                            Z[k, i] = c * Z[k, i] - s * h
                        end
                    end
                    p = -s * s2 * c3 * el1 * e[l] / dl1
                    e[l] = s * p
                    d[l] = c * p
                    abs(e[l]) > eps(R) * tst1 || break
                end
            end
            d[l] += f
            e[l] = 0
        end
        # Selection sort, ascending, as EigenG's DO_SORT
        for i in 1:n-1
            k = i
            for j in i+1:n
                d[j] < d[k] && (k = j)
            end
            if k != i
                (d[i], d[k]) = (d[k], d[i])
                for r in axes(Z, 1)            # lanes over rows
                    (Z[r, i], Z[r, k]) = (Z[r, k], Z[r, i])
                end
            end
        end
        return true
    end

    # Y ← Q Y = H₁ H₂ ⋯ Hₙ₋₁ Y, for the reflectors stored by `tridiagonalize!`
    function backtransform!(Y::AbstractMatrix{C}, A, τ) where C <: Complex
        n = size(A, 1)
        for k in n-1:-1:1
            r = k+1:n
            for j in axes(Y, 2)        # lanes over columns
                x = zero(C)
                for i in r
                    x += conj(A[i, k]) * Y[i, j]
                end
                x *= τ[k]
                for i in r
                    Y[i, j] -= A[i, k] * x
                end
            end
        end
        return
    end

    # Eigenvalues λ ascending and orthonormal eigenvectors Y of the Hermitian A,
    # which is overwritten. Returns (λ, Y, converged).
    function hermitian_eigen!(A::AbstractMatrix{Complex{R}}) where R
        n = size(A, 1)
        (d, e, τ) = (zeros(R, n), zeros(R, n), zeros(Complex{R}, n))
        tridiagonalize!(A, d, e, τ)
        Z = Matrix{R}(I, n, n)
        converged = tql2!(d, e, Z)
        Y = Complex{R}.(Z)
        backtransform!(Y, A, τ)
        return (d, Y, converged)
    end

    # Status codes of KAExt's batched Bogoliubov solvers
    const STATUS_OK = Int32(0)
    const STATUS_NONFINITE = Int32(-1)
    const STATUS_NOCONVERGE = Int32(-3)

    # Solves Ĩ t = λ H t for one Hamiltonian H, overwritten by its Cholesky
    # factor H = K K†, as the batched solvers of KAExt before
    # `Sunny.finish_vendor_bogoliubov!`. The Hermitian A = K⁻¹ Ĩ K⁻† has
    # eigenvectors y that give t = K⁻† y, with t† H t = 1. On return, T holds the
    # t and λ the eigenvalues, ascending. Returns a status code of KAExt.
    function bogoliubov_householder!(T::AbstractMatrix{C}, λ, H::AbstractMatrix{C}) where C <: Complex
        n = size(H, 1)
        L = n ÷ 2
        all(isfinite, H) || return STATUS_NONFINITE

        # Lower Cholesky factor K, in place. Lanes over rows below the pivot.
        for j in 1:n
            dj = real(H[j, j])
            for k in 1:j-1
                dj -= abs2(H[j, k])
            end
            dj > 0 || return Int32(j)
            dj = sqrt(dj)
            H[j, j] = dj
            for i in j+1:n
                x = H[i, j]
                for k in 1:j-1
                    x -= H[i, k] * conj(H[j, k])
                end
                H[i, j] = x / dj
            end
        end

        # B = K⁻¹ Ĩ by forward substitution, then A = K⁻¹ B†, which is A since
        # A is Hermitian. Lanes over columns, which are independent.
        B = zeros(C, n, n)
        for j in 1:n
            B[j, j] = j <= L ? 1 : -1
            forward_substitute!(view(B, :, j), H)
        end
        A = copy(B')
        for j in 1:n
            forward_substitute!(view(A, :, j), H)
        end
        hermitianpart!(A)

        (λs, Y, converged) = hermitian_eigen!(A)
        converged || return STATUS_NOCONVERGE

        # t = K⁻† y by back substitution, lanes over columns
        for j in 1:n
            back_substitute_adjoint!(view(Y, :, j), H)
        end
        T .= Y
        λ .= λs
        return STATUS_OK
    end

    # x ← K⁻¹ x for the lower triangle K of H
    function forward_substitute!(x, H)
        n = length(x)
        for i in 1:n
            for k in 1:i-1
                x[i] -= H[i, k] * x[k]
            end
            x[i] /= real(H[i, i])
        end
    end

    # x ← K⁻† x for the lower triangle K of H
    function back_substitute_adjoint!(x, H)
        n = length(x)
        for i in n:-1:1
            for k in i+1:n
                x[i] -= conj(H[k, i]) * x[k]
            end
            x[i] /= real(H[i, i])
        end
    end

    # The batched contract of `Sunny.vendor_bogoliubov!`, on host arrays
    function bogoliubov_householder_batched!(T, E, H, nb, status)
        n = size(H, 1)
        λ = zeros(real(eltype(H)), n, size(H, 3))
        for b in 1:nb
            status[b] = bogoliubov_householder!(view(T, :, :, b), view(λ, :, b), view(H, :, :, b))
        end
        Sunny.finish_vendor_bogoliubov!(T, E, λ, nb, status)
    end
end

@testitem "Householder QL reference eigensolver" setup=[HouseholderQL] begin
    using .HouseholderQL: hermitian_eigen!, tridiagonalize!
    using LinearAlgebra, Random

    rng = Xoshiro(0)
    randherm(C, n) = (A = randn(rng, C, n, n); A + A')

    # Random matrices of every size class of the device kernels: tiles of 4, 8,
    # 16 and 32 lanes, and a full SIMD group with several rows per lane
    for R in (Float64, Float32), n in (1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 54, 64)
        A0 = randherm(Complex{R}, n)
        (λ, Y, converged) = hermitian_eigen!(copy(A0))
        tol = 50n * eps(R) * opnorm(A0)
        @test converged
        @test issorted(λ)
        @test λ ≈ eigvals(Hermitian(Complex{Float64}.(A0))) atol=tol
        @test opnorm(A0 * Y - Y * Diagonal(λ)) < tol
        @test opnorm(Y' * Y - I) < 50n * eps(R)
    end

    # The reduction is to a real tridiagonal, similar by the unitary Q
    n = 7
    A0 = randherm(ComplexF64, n)
    A = copy(A0)
    (d, e, τ) = (zeros(n), zeros(n), zeros(ComplexF64, n))
    tridiagonalize!(A, d, e, τ)
    Q = Matrix{ComplexF64}(I, n, n)
    HouseholderQL.backtransform!(Q, A, τ)
    @test Q' * A0 * Q ≈ SymTridiagonal(d, e[1:n-1]) atol=1e-12
    @test Q' * Q ≈ I

    # Degenerate spectra, a diagonal matrix (trivial reflectors), complex
    # subdiagonal with an otherwise trivial column, and zero
    U = qr(randn(rng, ComplexF64, 6, 6)).Q * I
    for A0 in (U * Diagonal([1, 1, 1, 2, 2, 3.0]) * U', diagm(ComplexF64[3, -1, 2, 2]),
               ComplexF64[1 0 0; 0 2 1im; 0 -1im 3], zeros(ComplexF64, 4, 4))
        (λ, Y, converged) = hermitian_eigen!(copy(A0))
        @test converged
        @test λ ≈ eigvals(Hermitian(A0)) atol=1e-12
        @test A0 * Y ≈ Y * Diagonal(λ) atol=1e-12
        @test Y' * Y ≈ I
    end
end

@testitem "Householder QL reference Bogoliubov" setup=[HouseholderQL] begin
    using .HouseholderQL: bogoliubov_householder_batched!
    using LinearAlgebra, Random
    import KernelAbstractions

    # Random positive definite Hamiltonians, as `test_vendor_bogoliubov`, and
    # the status codes of a non-finite H and one that is not positive definite
    rng = Xoshiro(0)
    for R in (Float64, Float32), L in (1, 2, 4, 8, 17, 27)
        n = 2L
        Ĩ = diagm([ones(L); -ones(L)])
        Hs = [(A = randn(rng, ComplexF64, n, n); A * A' + I) for _ in 1:3]
        H = Complex{R}.(cat(Hs...; dims=3))
        T = zeros(Complex{R}, n, n, 3)
        E = zeros(R, n, 3)
        status = zeros(Int32, 3)
        bogoliubov_householder_batched!(T, E, H, 3, status)
        @test all(==(0), status)
        (rtol, atol) = R == Float64 ? (1e-10, 1e-8) : (1e-4, 1e-3)
        for b in 1:3
            T_h = zeros(ComplexF64, n, n)
            E_h = Sunny.bogoliubov!(T_h, copy(Hs[b]))
            @test E[:, b] ≈ E_h rtol=rtol
            @test T[:, :, b]' * Ĩ * T[:, :, b] ≈ Ĩ atol=atol
            @test abs.(T[:, :, b]' * Ĩ * T_h) ≈ I(n) atol=atol
        end

        Hs[2] = -Hs[2]
        Hs[3] = fill(NaN, n, n)
        H = Complex{R}.(cat(Hs...; dims=3))
        bogoliubov_householder_batched!(T, E, H, 3, status)
        @test status[1] == 0 && status[2] > 0 && status[3] == -1
    end

    # Sunny Hamiltonians: Néel order on the fcc lattice in :dipole and :SUN
    # modes, and the √3×√3 kagome antiferromagnet, whose flat band of zero
    # modes is lifted only by the regularization
    function fcc_afm(mode)
        a = 8.289
        fcc = Crystal(lattice_vectors(a, a, a, 90, 90, 90), [[0, 0, 0]], 225)
        sys = System(fcc, [1 => Moment(s=5/2, g=2)], mode)
        set_exchange!(sys, diagm([1.9, 1.9, 1.91]), Bond(1, 2, [0, 0, 0]))
        set_onsite_coupling!(sys, S -> 0.1 * (S[1]^4 + S[2]^4 + S[3]^4), 1)
        set_dipole!(sys, (1, 1, 1), position_to_site(sys, (0, 0, 0)))
        set_dipole!(sys, (1, -1, -1), position_to_site(sys, (1/2, 1/2, 0)))
        set_dipole!(sys, (-1, -1, 1), position_to_site(sys, (1/2, 0, 1/2)))
        set_dipole!(sys, (-1, 1, -1), position_to_site(sys, (0, 1/2, 1/2)))
        return sys
    end
    function kagome_sqrt3()
        cryst = Crystal(lattice_vectors(6, 6, 40, 90, 90, 120), [[1/2, 0, 0]], 147)
        sys = System(cryst, [1 => Moment(s=1, g=2)], :dipole)
        set_exchange!(sys, 1.0, Bond(2, 3, [0, 0, 0]))
        set_dipole!(sys, [1, 0, 0], (1, 1, 1, 1))
        set_dipole!(sys, [1, 0, 0], (1, 1, 1, 2))
        set_dipole!(sys, [cos(2π/3), sin(2π/3), 0], (1, 1, 1, 3))
        return repeat_periodically_as_spiral(sys, (3, 3, 1); k=[-1/3, -1/3, 0], axis=[0, 0, 1])
    end
    for sys in (fcc_afm(:dipole), fcc_afm(:SUN), kagome_sqrt3())
        swt = SpinWaveTheory(sys; measure=nothing)
        qs = [[0, 0, 0], [0.1, 0.2, 0.3], [1/2, 1/2, 0], rand(rng, 3)]
        Hs = [Matrix(Sunny.hamiltonian_matrix(swt, Sunny.to_supercell_rlu(sys, q))) for q in qs]
        n = size(Hs[1], 1)
        nb = length(qs)
        H = cat(Hs...; dims=3)
        T = zeros(ComplexF64, n, n, nb)
        E = zeros(n, nb)
        status = zeros(Int32, nb)
        bogoliubov_householder_batched!(T, E, H, nb, status)
        @test all(==(0), status)
        for b in 1:nb
            T_h = zeros(ComplexF64, n, n)
            E_h = Sunny.bogoliubov!(T_h, copy(Hs[b]))
            @test E[:, b] ≈ E_h rtol=1e-8
        end
    end
end
