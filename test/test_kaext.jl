@testmodule KAModels begin
    using Sunny, LinearAlgebra, Test, Random

    # Ferromagnetic order with antiferromagnetic exchange is unstable
    function unstable_chain()
        cryst = Crystal(lattice_vectors(1, 1, 2, 90, 90, 90), [[0, 0, 0]])
        sys = System(cryst, [1 => Moment(s=1, g=2)], :dipole)
        set_exchange!(sys, 1.0, Bond(1, 1, [1, 0, 0]))
        polarize_spins!(sys, [0, 0, 1])
        return sys
    end

    # The device reproduces the error of the host at the first unstable q
    function test_device_instability(backend; eigensolver=nothing)
        sys = unstable_chain()
        swt = SpinWaveTheory(sys; measure=ssf_trace(sys))
        errors = map((swt, to_device(swt; backend, eigensolver))) do s
            try
                intensities(s, [[0, 0, 0], [0.3, 0, 0]]; energies=[1.0], η=0.1)
            catch err
                err
            end
        end
        @test errors[2] isa Sunny.InstabilityError
        @test errors[2].msg == errors[1].msg
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

    # Intensities on `backend` reproduce those on the host. A batch size that
    # does not divide the number of wavevectors exercises the final batch.
    function test_device_intensities(swt, backend; eigensolver=nothing, energies=range(-2, 10, 25), broadenings=broadenings)
        Random.seed!(0)
        qs = [rand(3) for _ in 1:37]
        swt_d = to_device(swt; backend, eigensolver)
        for b in broadenings
            res = intensities(swt, qs; energies, b...)
            res_d = intensities(swt_d, qs; energies, batch_size=8, b...)
            @test eltype(res_d.data) == eltype(res.data)
            @test res_d.data ≈ res.data rtol=1e-12
        end
    end
end

@testitem "KAExt loads" begin
    import KernelAbstractions
    @test !isnothing(Base.get_extension(Sunny, :KAExt))
end

@testitem "KAExt intensities" setup=[KAModels] begin
    using .KAModels: fcc_afm, entangled_dimers, test_device_intensities
    import KernelAbstractions: CPU
    backend = CPU()

    for mode in (:dipole, :SUN)
        sys = fcc_afm(mode)
        swt = SpinWaveTheory(sys; measure=ssf_perp(sys; formfactors=[1 => FormFactor("Co2")]))
        test_device_intensities(swt, backend)
    end

    # A measure with many real components
    sys = fcc_afm(:dipole)
    swt = SpinWaveTheory(sys; measure=ssf_custom((q, ssf) -> ssf, sys))
    test_device_intensities(swt, backend; broadenings=[(; η=0.1, kernel=gaussian(fwhm=1.0), kT=0.3)])

    # Entangled units, whose observables have several parts
    sys = entangled_dimers()
    test_device_intensities(SpinWaveTheory(sys; measure=ssf_perp(sys)), backend)

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

    test_device_instability(CPU(); eigensolver=:host)
    test_device_instability(CPU(); eigensolver=:jacobi)

    sys = unstable_chain()
    @test_throws "not supported for" to_device(SpinWaveTheorySpiral(sys; measure=ssf_trace(sys), k=[0, 0, 0], axis=[0, 0, 1]))
    swt = SpinWaveTheory(sys; measure=ssf_trace(sys))
    @test_throws "Unknown eigensolver" to_device(swt; eigensolver=:magic)
    @test_throws "No vendor eigensolver" to_device(swt; eigensolver=:vendor)

    sys = fcc_afm(:dipole)
    swt_d = to_device(SpinWaveTheory(sys; measure=nothing))
    @test_throws "No observables!" intensities(swt_d, [[0, 0, 0]]; energies=[1.0], η=0.1)

    enable_dipole_dipole!(sys, 1.0)
    @test_throws "Dipole-dipole interactions are not yet supported" to_device(SpinWaveTheory(sys; measure=ssf_trace(sys)))
end

@testitem "KAExt intensities on CUDA" tags=[:gpu] setup=[KAModels] begin
    using .KAModels: fcc_afm, test_device_intensities, test_device_instability
    import CUDA
    backend = CUDA.CUDABackend()

    for eigensolver in (:vendor, :jacobi, :host)
        for mode in (:dipole, :SUN)
            sys = fcc_afm(mode)
            swt = SpinWaveTheory(sys; measure=ssf_perp(sys; formfactors=[1 => FormFactor("Co2")]))
            test_device_intensities(swt, backend; eigensolver)
        end
        test_device_instability(backend; eigensolver)
    end
    @test to_device(SpinWaveTheory(fcc_afm(:dipole); measure=nothing); backend).eigensolver == :vendor
end
