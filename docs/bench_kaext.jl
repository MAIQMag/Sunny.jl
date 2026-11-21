# Benchmark `intensities` of SpinWaveTheory on a KernelAbstractions device
# against the threaded host calculation. Run with
#
#     julia --threads=auto --project=@. bench_kaext.jl [cuda|metal|cpu]
#
# with Sunny, KernelAbstractions, and the GPU package available. The backend
# defaults to CUDA, then Metal (Apple silicon, Float32 only), then the
# KernelAbstractions CPU backend. Timings are the minimum of three warm runs.

using Sunny, KernelAbstractions, LinearAlgebra
const choice = isempty(ARGS) ? "auto" : lowercase(ARGS[1])
choice in ("auto", "cuda") && try @eval using CUDA catch end
choice in ("auto", "metal") && try @eval using Metal catch end
const backend = if isdefined(Main, :CUDA) && CUDA.functional()
    CUDA.CUDABackend()
elseif isdefined(Main, :Metal) && Metal.functional()
    Metal.MetalBackend()
else
    KernelAbstractions.CPU()
end

# Néel order on the fcc lattice of CoRh2O4-like moments, with a cubic anisotropy
function fcc_afm()
    a = 8.289
    fcc = Crystal(lattice_vectors(a, a, a, 90, 90, 90), [[0, 0, 0]], 225)
    sys = System(fcc, [1 => Moment(s=5/2, g=2)], :dipole)
    set_exchange!(sys, diagm([1.9, 1.9, 1.91]), Bond(1, 2, [0, 0, 0]))
    set_onsite_coupling!(sys, S -> 0.1 * (S[1]^4 + S[2]^4 + S[3]^4), 1)
    set_dipole!(sys, (1, 1, 1), position_to_site(sys, (0, 0, 0)))
    set_dipole!(sys, (1, -1, -1), position_to_site(sys, (1/2, 1/2, 0)))
    set_dipole!(sys, (-1, -1, 1), position_to_site(sys, (1/2, 0, 1/2)))
    set_dipole!(sys, (-1, 1, -1), position_to_site(sys, (0, 1/2, 1/2)))
    return sys
end

besttime(f) = (f(); minimum(@elapsed(f()) for _ in 1:3))

function bench(sys, Nq; host=true)
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    swt_d = to_device(swt; backend)
    qs = [rand(3) for _ in 1:Nq]
    energies = range(0, 10, 200)
    println("$(Sunny.nbands(swt)) bands (n = $(2Sunny.nbands(swt))), $Nq wavevectors, $(length(energies)) energies on $backend ",
            "(eigensolver $(Symbol(swt_d.eigensolver)), measure $(Symbol(swt_d.measure_mode)))")
    for (name, η, kernel) in (("η=0, kernel", 0, gaussian(fwhm=0.5)), ("η, no kernel", 0.1, nothing),
                              ("η and kernel", 0.1, gaussian(fwhm=0.5)))
        t_host = host ? besttime(() -> intensities(swt, qs; energies, η, kernel, threaded=true)) : NaN
        t_out = besttime(() -> intensities(swt_d, qs; energies, η, kernel, threaded=true))
        t_dev = besttime(() -> (intensities(swt_d, qs; energies, η, kernel, threaded=true, output=:device); KernelAbstractions.synchronize(backend)))
        println("  ", rpad(name, 14), "host ", round(t_host; digits=2), " s, device ", round(t_out; digits=2),
                " s (output on device ", round(t_dev; digits=2), " s)")
    end
end

load_blas_for_threading()
bench(fcc_afm(), 20_000)
bench(fcc_afm(), 200_000; host=true)
bench(repeat_periodically(fcc_afm(), (2, 2, 2)), 5_000)
