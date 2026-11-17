# Profiles the stages of the device powder average of
# bench_CoRh2O4_device.jl, for one batch of 50 shells of 2000 points, i.e.
# one call of `intensities(swt_d, qs; output=:device)`. Each stage is the best
# of several runs, synchronized before and after, on the first functional GPU
# backend (CUDA, AMDGPU, Metal, else the KernelAbstractions CPU backend). Run
# with
#
#     julia --project=<env> examples/extra/profile_CoRh2O4_device.jl
#
# where <env> has Sunny, KernelAbstractions, and the GPU package.

using Sunny, KernelAbstractions, Printf
import KernelAbstractions as KA
try @eval using CUDA catch end
try @eval using AMDGPU catch end
try @eval using Metal catch end
backend = if @isdefined(CUDA) && CUDA.functional()
    CUDABackend()
elseif @isdefined(AMDGPU) && AMDGPU.functional()
    ROCBackend()
elseif @isdefined(Metal) && Metal.functional()
    MetalBackend()
else
    KA.CPU()
end
const KAExt = Base.get_extension(Sunny, :KAExt)

# The model of the CoRh₂O₄ tutorial
a = 8.5031
cryst = Crystal(lattice_vectors(a, a, a, 90, 90, 90), [[1/8, 1/8, 1/8]], 227; types=["Co"])
sys = System(cryst, [1 => Moment(s=3/2, g=2)], :dipole; seed=0)
set_exchange!(sys, 0.63, Bond(2, 3, [0, 0, 0]))
randomize_spins!(sys)
minimize_energy!(sys)
sys = reshape_supercell(sys, primitive_cell(cryst))
swt = SpinWaveTheory(sys; measure=ssf_perp(sys; formfactors=[1 => FormFactor("Co2")]))
swt_d = to_device(swt; backend)
energies = range(0, 6, 300)
kernel = lorentzian(fwhm=0.8)
radii = range(0, 3, 200)
(npts, nshells) = (2000, 50)
# The stages run inside a function, so that their host code is compiled for
# the types of its variables, as in `intensities`
function profile(backend, swt, swt_d, cryst; energies, kernel, radii, npts, nshells)
    # The best time of `f` in seconds, after a warm-up, synchronized before and
    # after. The optional `setup` and a garbage collection run untimed before each
    # run, so that a collection triggered by an earlier run does not land in it.
    function besttime(f; setup=() -> nothing, reps=5)
        setup(); f(); KA.synchronize(backend)
        return minimum(1:reps) do _
            setup()
            GC.gc()
            KA.synchronize(backend)
            t = @elapsed (f(); KA.synchronize(backend))
            t
        end
    end

    # The wavevectors of the first batch, as `powder_average` makes them
    function batch_qs()
        rng = Sunny.Random.Xoshiro(0)
        sphpts = Sunny.sphere_points(npts)
        to_rlu = inv(cryst.recipvecs)
        return reduce(vcat, [Ref(to_rlu * Sunny.Mat3(Sunny.random_orthogonal(rng, 3)) * radii[i]) .* sphpts for i in 1:nshells])
    end
    qs = batch_qs()
    qpts = convert(Sunny.AbstractQPoints, qs)
    Nq = length(qs)

    # Buffers of `susceptibility_poles_device` for a single batch of all Nq
    (; data, measure_mode) = swt_d
    F = KAExt.precision_of(swt_d)
    Na = Sunny.nsites(swt.sys)
    Ncells = F(Na / Sunny.natoms(cryst))
    Nf = Sunny.nflavors(swt)
    n = 2Sunny.nbands(swt)
    Nobs = Sunny.num_observables(swt.measure)
    K = Sunny.num_real_components(eltype(swt.measure))
    disp = KA.allocate(backend, F, (n, Nq))
    R = KA.allocate(backend, F, (K, n, Nq))
    H = KA.allocate(backend, Complex{F}, (n, n, Nq))
    T = KA.allocate(backend, Complex{F}, (n, n, Nq))
    E = KA.allocate(backend, F, (n, Nq))
    u = KA.allocate(backend, Complex{F}, (n, Nobs, Nq))
    A = KA.allocate(backend, Complex{F}, (Nobs, n, Nq))

    qs_supercell = [Sunny.to_supercell_rlu(swt.sys, q) for q in qs]
    qs_global = [cryst.recipvecs * Sunny.Vec3(q) for q in qs]
    qs_supercell_d = KAExt.to_backend(backend, F, qs_supercell)
    qs_global_d = KAExt.to_backend(backend, F, qs_global)

    build!() = KAExt.build_hamiltonian!(backend)(H, data.rowptr, data.cols, data.offset_index, data.vals, data.offsets,
                                                 qs_supercell_d, data.regularization; ndrange=(n, Nq))
    bogoliubov!() = KAExt.bogoliubov_batched!(T, E, H, Nq, swt_d, qs_supercell; threaded=false)
    observables!() = KAExt.observable_vectors!(backend)(u, data.coefs, data.positions, data.formfactors, qs_global_d, Nf; ndrange=(Na, Nobs, Nq))
    amplitudes!() = KAExt.amplitudes!(backend)(A, u, T; ndrange=(Nobs, n, Nq))
    # The measure as Hermitian forms per wavevector, if its combiner cannot run on
    # device, evaluated on the host
    if measure_mode == :forms
        M_h = zeros(Complex{F}, Nobs, Nobs, K, Nq)
        M = KA.allocate(backend, Complex{F}, size(M_h))
    end
    function forms!()
        for b in 1:Nq, (k, Mk) in enumerate(Sunny.measure_forms(swt.measure, qs_global[b]))
            view(M_h, :, :, k, b) .= Mk
        end
        copyto!(M, M_h)
    end
    function residues!()
        if measure_mode == :combiner
            KAExt.residues_combiner!(backend)(disp, R, E, A, qs_global_d, swt.measure.combiner, data.corr_pairs, 0, Ncells; ndrange=(n, Nq))
        else
            KAExt.residues!(backend)(disp, R, E, A, measure_mode == :forms ? M : data.forms, 0, Ncells; ndrange=(n, Nq))
        end
    end
    prepare() = KAExt.prepare_broadening(backend, disp, R; energies, η=0, kernel, kT=0)
    plan = (measure_mode == :forms && forms!(); build!(); bogoliubov!(); observables!(); amplitudes!(); residues!(); prepare())
    out = KA.allocate(backend, F, (K, length(energies), Nq))
    broaden!() = KAExt.broaden_batch!(backend, out, plan, 0, Nq, 0, nothing)
    res_d = intensities(swt_d, qs; energies, kernel, output=:device)
    means() = Sunny.shell_means(res_d, npts, nshells)

    stages = [
        "wavevectors on host"       => besttime(batch_qs),
        "q to supercell, copy"      => besttime(() -> (KAExt.to_backend(backend, F, [Sunny.to_supercell_rlu(swt.sys, q) for q in qs]),
                                                       KAExt.to_backend(backend, F, [cryst.recipvecs * Sunny.Vec3(q) for q in qs]))),
        "build_hamiltonian!"        => besttime(build!),
        "bogoliubov ($(swt_d.eigensolver))" => besttime(bogoliubov!; setup=build!),
        "observable_vectors!"       => besttime(observables!),
        "amplitudes!"               => besttime(amplitudes!),
        (measure_mode == :forms ? ["measure forms on host, copy" => besttime(forms!)] : [])...,
        "residues ($measure_mode)"  => besttime(residues!),
        "prepare_broadening"        => besttime(prepare),
        "broaden_peaks!"            => besttime(broaden!),
        "shell means, copy to host" => besttime(means),
    ]
    t_call = besttime(() -> intensities(swt_d, qs; energies, kernel, output=:device))
    t_powder = besttime(() -> powder_average(cryst, radii, npts; batch_size=nshells) do qs
        intensities(swt_d, qs; energies, kernel, output=:device)
    end)

    println(Na, " sites (n = $n), $Nq wavevectors × $(length(energies)) energies per batch, on ", backend, " in ", F,
            ", eigensolver ", repr(swt_d.eigensolver), ", measure ", repr(measure_mode))
    t_sum = sum(last, stages)
    for (name, t) in stages
        @printf("  %-30s %8.2f ms  %5.1f%%\n", name, 1e3t, 100t / t_sum)
    end
    @printf("  %-30s %8.2f ms\n", "sum of stages", 1e3t_sum)
    @printf("  %-30s %8.2f ms\n", "one call of intensities", 1e3t_call)
    @printf("  %-30s %8.2f ms for %d batches, %.2f ms per batch\n", "powder_average", 1e3t_powder,
            cld(length(radii), nshells), 1e3t_powder / cld(length(radii), nshells))
end

profile(backend, swt, swt_d, cryst; energies, kernel, radii, npts, nshells)
