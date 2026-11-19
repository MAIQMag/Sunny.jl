@testmodule KAModels begin
    using Sunny, LinearAlgebra, Test, Random
    import KernelAbstractions as KA

    # Ferromagnetic order with antiferromagnetic exchange is unstable
    function unstable_chain()
        cryst = Crystal(lattice_vectors(1, 1, 2, 90, 90, 90), [[0, 0, 0]])
        sys = System(cryst, [1 => Moment(s=1, g=2)], :dipole)
        set_exchange!(sys, 1.0, Bond(1, 1, [1, 0, 0]))
        polarize_spins!(sys, [0, 0, 1])
        return sys
    end

    # The device reproduces the error of the host at the first unstable q
    function test_device_instability(backend; eigensolver=nothing, precision=Sunny.device_precision(backend))
        sys = unstable_chain()
        swt = SpinWaveTheory(sys; measure=ssf_trace(sys))
        errors = map((swt, to_device(swt; backend, eigensolver, precision))) do s
            try
                intensities(s, [[0, 0, 0], [0.3, 0, 0]]; energies=[1.0], η=0.1)
            catch err
                err
            end
        end
        @test errors[2] isa Sunny.InstabilityError
        @test errors[2].msg == errors[1].msg
    end

    # The vendor eigensolver of `backend` in precision F reproduces
    # `Sunny.bogoliubov!` on random positive definite Hamiltonians, acting on
    # only the first nb of a larger batch, and reports the status codes of a
    # non-finite H and a H that is not positive definite.
    function test_vendor_bogoliubov(backend, F; L=4)
        nb = 5
        n = 2L
        Ĩ = diagm([ones(L); -ones(L)])
        rng = Xoshiro(0)
        Hs = map(1:nb+1) do _
            A = randn(rng, ComplexF64, n, n)
            A * A' + I
        end
        function solve(Hs)
            H_d = KA.allocate(backend, Complex{F}, (n, n, nb+1))
            copyto!(H_d, Complex{F}.(cat(Hs...; dims=3)))
            T_d = KA.zeros(backend, Complex{F}, (n, n, nb+1))
            E_d = KA.zeros(backend, F, (n, nb+1))
            status = KA.allocate(backend, Int32, nb)
            Sunny.vendor_bogoliubov!(T_d, E_d, H_d, nb, status)
            return (Array(T_d), Array(E_d), Array(status))
        end

        (T, E, status) = solve(Hs)
        @test all(==(0), status)
        (rtol, atol) = F == Float64 ? (1e-10, 1e-8) : (1e-4, 1e-3)
        for b in 1:nb
            T_h = zeros(ComplexF64, n, n)
            E_h = Sunny.bogoliubov!(T_h, copy(Hs[b]))
            @test E[:, b] ≈ E_h rtol=rtol
            # Para-unitary, T† Ĩ T = Ĩ, and each eigenvector is defined up to
            # a phase. The error of an eigenvector grows as its eigenvalue gap
            # shrinks, which it does with n, so the phases are compared by the
            # largest entry rather than the norm of all n² entries.
            @test T[:, :, b]' * Ĩ * T[:, :, b] ≈ Ĩ atol=atol
            @test maximum(abs, abs.(T[:, :, b]' * Ĩ * T_h) - I(n)) < atol
        end
        # The extra Hamiltonian beyond nb is untouched
        @test iszero(T[:, :, nb+1]) && iszero(E[:, nb+1])

        Hs_bad = copy(Hs)
        Hs_bad[2] = -Hs[2]
        (_, _, status) = solve(Hs_bad)
        @test status[2] > 0
        Hs_bad[4] = fill(NaN, n, n)
        (_, _, status) = solve(Hs_bad)
        # cuSOLVER and rocSOLVER skip the batch if any H is non-finite, leaving
        # 0 for the H that is not positive definite, whereas the Metal kernel
        # solves each H independently, as :host and :jacobi do
        @test status[[1, 3, 4, 5]] == [0, 0, -1, 0]
        @test status[2] == 0 || status[2] > 0
    end

    # Néel order on the fcc lattice, with cubic anisotropy
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

    # Ferromagnetic dimers, entangled into units, with antiferromagnetic
    # inter-unit exchange, in a field
    function entangled_dimers()
        latvecs = [1 0 0; 0 1 0; 0 0 2]
        crystal = Crystal(latvecs, [[0, 0, 0.0], [0, 0.5, 0.0]], 1; types=["A", "B"])
        sys = System(crystal, [1 => Moment(s=1/2, g=2), 2 => Moment(s=1/2, g=2)], :SUN)
        set_exchange!(sys, -0.3, Bond(1, 2, [0, 0, 0]))
        set_exchange!(sys, [1 0.2 0; -0.2 1 0; 0 0 1], Bond(1, 1, [1, 0, 0]))
        set_exchange!(sys, 1.0, Bond(2, 2, [1, 0, 0]))
        sys = entangle_system(resize_supercell(sys, (2, 1, 1)), [[1, 2]])
        set_field!(sys, [0, 0, 3])
        minimize_energy!(sys)
        return sys
    end

    # Every branch of the broadening, with and without temperature
    const broadenings = [
        (; η=0, kernel=gaussian(fwhm=1.0), kT=0),
        (; η=0, kernel=lorentzian(fwhm=0.5), kT=0.5),
        (; η=0.2, kernel=nothing, kT=0),
        (; η=0.2, kernel=nothing, kT=0.3),
        (; η=0.1, kernel=gaussian(fwhm=1.0), kT=0.3),
    ]

    # Intensities on `backend` reproduce those on the host, to a tolerance for
    # the `precision` on device, relative to the norm of all intensities. A
    # batch size that does not divide the number of wavevectors exercises the
    # final batch. The host reference uses the regularization of the device,
    # which in Float32 is raised above that of `swt` and shifts energies by
    # more than rounding, so that the tolerance measures rounding alone.
    function test_device_intensities(swt, backend; eigensolver=nothing, precision=Sunny.device_precision(backend),
                                     rtol=(precision == Float64 ? 1e-12 : 1e-4), energies=range(-2, 10, 25), broadenings=broadenings)
        Random.seed!(0)
        qs = [rand(3) for _ in 1:37]
        swt_d = to_device(swt; backend, eigensolver, precision)
        regularization = Float64(swt_d.data.regularization)
        if regularization != swt.regularization
            swt = SpinWaveTheory(swt.sys; swt.measure, regularization)
        end
        for b in broadenings
            res = intensities(swt, qs; energies, b...)
            res_d = intensities(swt_d, qs; energies, batch_size=8, b...)
            @test eltype(res_d.data) == eltype(res.data)
            @test res_d.data ≈ res.data rtol=rtol
            res_d = intensities(swt_d, qs; energies, batch_size=8, output=:device, b...)
            @test real(eltype(eltype(res_d.data))) == precision
            @test Sunny.Intensities(res_d).data ≈ res.data rtol=rtol
        end
        return swt_d
    end
end

@testitem "KAExt loads" begin
    import KernelAbstractions
    @test !isnothing(Base.get_extension(Sunny, :KAExt))
end

@testitem "KAExt intensities" setup=[KAModels] begin
    using .KAModels: fcc_afm, entangled_dimers, test_device_intensities
    using LinearAlgebra: tr
    import KernelAbstractions: CPU
    backend = CPU()

    for mode in (:dipole, :SUN)
        sys = fcc_afm(mode)
        swt = SpinWaveTheory(sys; measure=ssf_perp(sys; formfactors=[1 => FormFactor("Co2")]))
        test_device_intensities(swt, backend)
    end

    # Float32 on device, with both eigensolvers. The :host eigensolver works in
    # Float64 on the Float32 Hamiltonian.
    for mode in (:dipole, :SUN), eigensolver in (:host, :jacobi)
        sys = fcc_afm(mode)
        swt = SpinWaveTheory(sys; measure=ssf_perp(sys; formfactors=[1 => FormFactor("Co2")]))
        swt_d = test_device_intensities(swt, backend; eigensolver, precision=Float32)
        @test swt_d.measure_mode == :combiner
        @test occursin("in Float32", sprint(show, MIME"text/plain"(), swt_d))
    end

    # The combiner of a measure runs on device if it can. Otherwise its forms
    # are evaluated on the host, once if they are independent of q.
    sys = fcc_afm(:dipole)
    b = [(; η=0.1, kernel=gaussian(fwhm=1.0), kT=0.3)]
    # Capturing a Vector makes a combiner non-isbits
    for (f, mode) in [((q, ssf) -> ssf, :combiner),
                      let w = [1.0]; ((q, ssf) -> w[1] * real(tr(ssf)), :constant_forms) end,
                      let w = [1.0]; ((q, ssf) -> w[1] * real(ssf[1, 1] + ssf[2, 2]) * (1 + q[1]^2), :forms) end]
        for precision in (Float64, Float32)
            swt_d = test_device_intensities(SpinWaveTheory(sys; measure=ssf_custom(f, sys)), backend; precision, broadenings=b)
            @test swt_d.measure_mode == mode
        end
    end

    # Néel order of CoRh₂O₄, whose Goldstone mode at q = 0 the regularization
    # stabilizes in Float32 as well as Float64. The Float32 shift changes
    # intensities near the mode, by about 4e-4 relative to all intensities at
    # these wavevectors, two of which are at or near the mode.
    a = 8.5031
    cryst = Crystal(lattice_vectors(a, a, a, 90, 90, 90), [[1/8, 1/8, 1/8]], 227)
    sys = System(cryst, [1 => Moment(s=3/2, g=2)], :dipole; seed=0)
    set_exchange!(sys, 0.63, Bond(2, 3, [0, 0, 0]))
    randomize_spins!(sys)
    minimize_energy!(sys)
    sys = reshape_supercell(sys, primitive_cell(cryst))
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    @test Sunny.energy_per_site(sys) ≈ -2 * 0.63 * (3/2)^2
    qs = [[0, 0, 0], [0.1, 0, 0], [0.5, 0.5, 0]]
    res = intensities(swt, qs; energies=range(0, 6, 50), kernel=lorentzian(fwhm=0.8))
    for precision in (Float64, Float32), eigensolver in (:host, :jacobi)
        res_d = intensities(to_device(swt; backend, eigensolver, precision), qs; energies=range(0, 6, 50), kernel=lorentzian(fwhm=0.8))
        @test res_d.data ≈ res.data rtol=(precision == Float64 ? 1e-10 : 1e-3)
    end

    # A powder average from intensities on device, averaged there or on the
    # host, reproduces the one on the host
    sys = fcc_afm(:dipole)
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    cryst = sys.crystal
    (radii, energies, kernel) = (range(0, 2, 7), range(0, 10, 20), gaussian(fwhm=1.0))
    res = powder_average(qs -> intensities(swt, qs; energies, kernel), cryst, radii, 30)
    for precision in (Float64, Float32), output in (:host, :device)
        swt_d = to_device(swt; backend, precision)
        res_d = powder_average(qs -> intensities(swt_d, qs; energies, kernel, output), cryst, radii, 30; batch_size=3)
        @test res_d isa Sunny.PowderIntensities{Float64}
        @test res_d.data ≈ res.data rtol=(precision == Float64 ? 1e-12 : 1e-4)
    end

    # Entangled units, whose observables have several parts
    sys = entangled_dimers()
    test_device_intensities(SpinWaveTheory(sys; measure=ssf_perp(sys)), backend)
    test_device_intensities(SpinWaveTheory(sys; measure=ssf_perp(sys)), backend; precision=Float32)

    # The portable Jacobi eigensolver
    for s in (fcc_afm(:dipole), fcc_afm(:SUN))
        test_device_intensities(SpinWaveTheory(s; measure=ssf_perp(s)), backend; eigensolver=:jacobi,
                                broadenings=[(; η=0.1, kernel=gaussian(fwhm=1.0), kT=0.3)])
    end
    sys = entangled_dimers()
    test_device_intensities(SpinWaveTheory(sys; measure=ssf_perp(sys)), backend; eigensolver=:jacobi, broadenings=[(; η=0.2, kernel=nothing, kT=0)])

    # A corrected Gaussian vacuum
    sys = fcc_afm(:dipole)
    swt = SpinWaveTheory(sys; measure=ssf_trace(sys))
    swt = Sunny.swt_with_vacuum(swt; correction=Sunny.hartree_fock_correction(swt; grid=(4, 4, 4)).terms2)
    test_device_intensities(swt, backend; broadenings=[(; η=0.1, kernel=nothing, kT=0)])
end

@testitem "KAExt errors" setup=[KAModels] begin
    using .KAModels: fcc_afm, unstable_chain, test_device_instability
    import KernelAbstractions: CPU

    for eigensolver in (:host, :jacobi), precision in (Float64, Float32)
        test_device_instability(CPU(); eigensolver, precision)
    end

    sys = unstable_chain()
    @test_throws "not supported for" to_device(SpinWaveTheorySpiral(sys; measure=ssf_trace(sys), k=[0, 0, 0], axis=[0, 0, 1]))
    swt = SpinWaveTheory(sys; measure=ssf_trace(sys))
    @test_throws "Unknown eigensolver" to_device(swt; eigensolver=:magic)
    @test_throws "No vendor eigensolver" to_device(swt; eigensolver=:vendor)
    @test_throws "Unsupported precision" to_device(swt; precision=Float16)

    sys = fcc_afm(:dipole)
    swt_d = to_device(SpinWaveTheory(sys; measure=nothing))
    @test_throws "No observables!" intensities(swt_d, [[0, 0, 0]]; energies=[1.0], η=0.1)

    enable_dipole_dipole!(sys, 1.0)
    @test_throws "Dipole-dipole interactions are not yet supported" to_device(SpinWaveTheory(sys; measure=ssf_trace(sys)))
end

# The steps shared by vendor eigensolvers, applied to the solution of LAPACK on
# the host, reproduce `Sunny.bogoliubov!`
@testitem "KAExt vendor Bogoliubov steps" begin
    import KernelAbstractions
    using LinearAlgebra: Hermitian, I, diagm, eigen
    using Random: Xoshiro
    KAExt = Base.get_extension(Sunny, :KAExt)

    (L, nb) = (3, 4)
    n = 2L
    Ĩ = diagm([ones(L); -ones(L)])
    rng = Xoshiro(0)
    Hs = map(1:nb) do _
        A = randn(rng, ComplexF64, n, n)
        A * A' + I
    end
    T = zeros(ComplexF64, n, n, nb)
    λ = zeros(n, nb)
    for b in 1:nb
        (λ[:, b], T[:, :, b]) = eigen(Hermitian(Ĩ), Hermitian(Hs[b]))
    end
    # Eigenvalues that are all positive break the inertia. A failure already
    # recorded for another set is kept.
    λ[:, 3] .= sort(abs.(λ[:, 3]))
    status = Int32[0, 5, 0, 0]
    E = zeros(n, nb)
    Sunny.finish_vendor_bogoliubov!(T, E, λ, nb, status)
    @test status == [KAExt.STATUS_OK, 5, KAExt.STATUS_INERTIA, KAExt.STATUS_OK]
    for b in (1, 4)
        T_h = zeros(ComplexF64, n, n)
        E_h = Sunny.bogoliubov!(T_h, copy(Hs[b]))
        @test E[:, b] ≈ E_h
        # Para-unitary, T† Ĩ T = Ĩ, and each eigenvector is defined up to a phase
        @test T[:, :, b]' * Ĩ * T[:, :, b] ≈ Ĩ
        @test abs.(T[:, :, b]' * Ĩ * T_h) ≈ I(n) atol=1e-8
    end

    sys = System(Crystal(lattice_vectors(1, 1, 1, 90, 90, 90), [[0, 0, 0]]), [1 => Moment(s=1, g=2)], :dipole)
    swt = SpinWaveTheory(sys; measure=nothing)
    @test_throws "failed to converge" KAExt.throw_bogoliubov_status([KAExt.STATUS_OK, KAExt.STATUS_NOCONVERGE], swt, [[0, 0, 0], [0.5, 0, 0]])
end

@testitem "KAExt intensities on CUDA" tags=[:gpu, :cuda] setup=[KAModels] begin
    using .KAModels: fcc_afm, test_device_intensities, test_device_instability, test_vendor_bogoliubov
    import CUDA
    backend = CUDA.CUDABackend()

    # n = 8 and n = 48
    for precision in (Float64, Float32), L in (4, 24)
        test_vendor_bogoliubov(backend, precision; L)
    end
    for eigensolver in (:vendor, :jacobi, :host), precision in (Float64, Float32)
        for mode in (:dipole, :SUN)
            sys = fcc_afm(mode)
            swt = SpinWaveTheory(sys; measure=ssf_perp(sys; formfactors=[1 => FormFactor("Co2")]))
            test_device_intensities(swt, backend; eigensolver, precision)
        end
        test_device_instability(backend; eigensolver, precision)
    end

    # cuSOLVER is the default eigensolver in both precisions
    swt = SpinWaveTheory(fcc_afm(:dipole); measure=nothing)
    for precision in (Float64, Float32)
        @test to_device(swt; backend, precision).eigensolver == :vendor
    end
end

@testitem "KAAMDGPUExt loads" tags=[:gpu, :amdgpu] begin
    import KernelAbstractions, AMDGPU
    @test !isnothing(Base.get_extension(Sunny, :KAExt))
    @test !isnothing(Base.get_extension(Sunny, :KAAMDGPUExt))
    @test AMDGPU.functional()
    @test Sunny.device_free_memory(AMDGPU.ROCBackend()) > 0
end

# AMD GPUs support Float64, which is the default precision. The vendor
# eigensolver is rocSOLVER's batched hegvd, in either precision.
@testitem "KAExt intensities on AMDGPU" tags=[:gpu, :amdgpu] setup=[KAModels] begin
    using .KAModels: fcc_afm, entangled_dimers, test_device_intensities, test_device_instability, test_vendor_bogoliubov
    import AMDGPU
    backend = AMDGPU.ROCBackend()

    @test Sunny.device_precision(backend) == Float64
    # n = 8 and n = 48
    for precision in (Float64, Float32), L in (4, 24)
        test_vendor_bogoliubov(backend, precision; L)
    end
    for eigensolver in (:vendor, :jacobi, :host), precision in (Float64, Float32)
        for mode in (:dipole, :SUN)
            sys = fcc_afm(mode)
            swt = SpinWaveTheory(sys; measure=ssf_perp(sys; formfactors=[1 => FormFactor("Co2")]))
            swt_d = test_device_intensities(swt, backend; eigensolver, precision)
            @test swt_d.measure_mode == :combiner
        end
        sys = entangled_dimers()
        test_device_intensities(SpinWaveTheory(sys; measure=ssf_perp(sys)), backend; eigensolver, precision)
        test_device_instability(backend; eigensolver, precision)
    end

    # rocSOLVER is the default eigensolver in both precisions
    swt = SpinWaveTheory(fcc_afm(:dipole); measure=nothing)
    for precision in (Float64, Float32)
        @test to_device(swt; backend, precision).eigensolver == :vendor
    end
end

# Metal lacks Float64, so the device works in Float32, and a vendor eigensolver
@testitem "KAExt intensities on Metal" tags=[:gpu, :metal] setup=[KAModels] begin
    using .KAModels: fcc_afm, entangled_dimers, unstable_chain, test_device_intensities, test_device_instability,
                     test_vendor_bogoliubov
    import Metal
    backend = Metal.MetalBackend()

    @test Sunny.device_precision(backend) == Float32
    @test Sunny.device_free_memory(backend) > 0

    # The vendor eigensolver, a port of EigenG-Batched, for matrices of every
    # tile width, n ≤ 32, and with one SIMD group per matrix up to its limit of
    # n = 256
    for L in (1, 2, 3, 4, 8, 16, 17, 20, 27, 64, 128)
        test_vendor_bogoliubov(backend, Float32; L)
    end

    # The :SUN model has n = 40, beyond the tiles
    for eigensolver in (:vendor, :jacobi, :host)
        for mode in (:dipole, :SUN)
            sys = fcc_afm(mode)
            swt = SpinWaveTheory(sys; measure=ssf_perp(sys; formfactors=[1 => FormFactor("Co2")]))
            swt_d = test_device_intensities(swt, backend; eigensolver)
            # The combiner of `ssf_perp` computes in Float64, so its forms are
            # evaluated on the host instead
            @test_broken swt_d.measure_mode == :combiner
        end
        test_device_instability(backend; eigensolver)
    end
    # Entangled units, n = 4, on the vendor eigensolver
    sys = entangled_dimers()
    test_device_intensities(SpinWaveTheory(sys; measure=ssf_perp(sys)), backend; eigensolver=:vendor)

    # The vendor eigensolver is the default within its limits, :jacobi for at
    # most 2 bands, and otherwise :host, as without one. 200 sites give n = 400.
    for sys in (unstable_chain(), repeat_periodically(unstable_chain(), (2, 1, 1)))
        swt = SpinWaveTheory(sys; measure=nothing)
        @test to_device(swt; backend).eigensolver == :jacobi
        @test to_device(swt; backend, eigensolver=:vendor).eigensolver == :vendor
    end
    for mode in (:dipole, :SUN)
        swt = SpinWaveTheory(fcc_afm(mode); measure=nothing)
        @test to_device(swt; backend).eigensolver == :vendor
    end
    swt = SpinWaveTheory(repeat_periodically(fcc_afm(:dipole), (5, 5, 2)); measure=nothing)
    @test to_device(swt; backend).eigensolver == :host
    @test_throws "at most 128 bands" to_device(swt; backend, eigensolver=:vendor)
    @test_throws "Float64" to_device(swt; backend, precision=Float64)
end
