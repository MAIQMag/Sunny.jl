@testitem "Metal tiled Hermitian eigensolver" tags=[:gpu, :metal] setup=[HouseholderQL] begin
    import Metal, KernelAbstractions
    using Metal: MtlArray
    using LinearAlgebra, Random
    const MT = Base.get_extension(Sunny, :KAMetalExt)

    function solve(As)
        (n, _, nb) = size(As)
        A = MtlArray(As)
        Y = MtlArray(zeros(ComplexF32, n, n, nb))
        λ = MtlArray(zeros(Float32, n, nb))
        status = MtlArray(fill(Int32(99), nb))
        MT.hermitian_eigen_tiled!(λ, Y, A, status)
        Metal.synchronize()
        return (Array(λ), Array(Y), Array(status))
    end

    # Random Hermitian batches of every size in each tile width. A batch of 37
    # does not fill the last threadgroup. Accuracy is that of the scalar
    # reference in Float32, against LAPACK in Float64.
    rng = Xoshiro(0)
    for n in (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32)
        nb = 37
        As = zeros(ComplexF32, n, n, nb)
        for b in 1:nb
            X = randn(rng, ComplexF32, n, n)
            As[:, :, b] = X + X'
        end
        (λ, Y, status) = solve(As)
        @test all(==(0), status)
        for b in 1:nb
            A0 = ComplexF64.(As[:, :, b])
            tol = 50n * eps(Float32) * opnorm(A0)
            @test issorted(λ[:, b])
            @test λ[:, b] ≈ eigvals(Hermitian(A0)) atol=tol
            @test opnorm(A0 * Y[:, :, b] - Y[:, :, b] * Diagonal(λ[:, b])) < tol
            @test opnorm(Y[:, :, b]' * Y[:, :, b] - I) < 50n * eps(Float32)
            # The scalar reference in Float32 agrees to rounding
            (λ_ref, _, _) = HouseholderQL.hermitian_eigen!(copy(As[:, :, b]))
            @test λ[:, b] ≈ λ_ref atol=tol
        end
    end

    # Degenerate spectra, a diagonal matrix (trivial reflectors), a complex
    # subdiagonal with an otherwise trivial column, and zero, in one batch of
    # tile width 8 so that each tile of a SIMD group takes its own path
    U = Matrix(qr(randn(rng, ComplexF64, 6, 6)).Q)
    cases = [U * Diagonal([1, 1, 1, 2, 2, 3.0]) * U', diagm(ComplexF64[3, -1, 2, 2, 5, 0]),
             cat(ComplexF64[1 0 0; 0 2 1im; 0 -1im 3], I(3); dims=(1, 2)), zeros(ComplexF64, 6, 6)]
    (λ, Y, status) = solve(ComplexF32.(cat(cases...; dims=3)))
    @test all(==(0), status)
    for (b, A0) in enumerate(cases)
        @test λ[:, b] ≈ eigvals(Hermitian(A0)) atol=1e-5
        @test A0 * Y[:, :, b] ≈ Y[:, :, b] * Diagonal(λ[:, b]) atol=1e-5
        @test Y[:, :, b]' * Y[:, :, b] ≈ I atol=1e-5
    end
end
