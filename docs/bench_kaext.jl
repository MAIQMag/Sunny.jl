# Benchmark `intensities` of SpinWaveTheory on a KernelAbstractions device
# against the threaded host calculation. To run on an NVIDIA GPU:
#
#     julia --threads=auto --project=@. bench_kaext.jl
#
# with Sunny, CUDA and KernelAbstractions available. Timings are the minimum of
# three warm runs. Without CUDA, the KernelAbstractions CPU backend is used.

using Sunny, KernelAbstractions, LinearAlgebra
const backend = try
    @eval using CUDA
    CUDA.functional() ? CUDA.CUDABackend() : KernelAbstractions.CPU()
catch
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
            "(eigensolver $(swt_d.eigensolver), measure $(swt_d.measure_mode))")
    for (name, η, kernel) in (("η=0, kernel", 0, gaussian(fwhm=0.5)), ("η, no kernel", 0.1, nothing),
                              ("η and kernel", 0.1, gaussian(fwhm=0.5)))
        t_host = host ? besttime(() -> intensities(swt, qs; energies, η, kernel, threaded=true)) : NaN
        t_out = besttime(() -> intensities(swt_d, qs; energies, η, kernel))
        t_dev = besttime(() -> (intensities(swt_d, qs; energies, η, kernel, output=:device); KernelAbstractions.synchronize(backend)))
        println("  ", rpad(name, 14), "host ", round(t_host; digits=2), " s, device ", round(t_out; digits=2),
                " s (output on device ", round(t_dev; digits=2), " s)")
    end
end

bench(fcc_afm(), 20_000)
bench(fcc_afm(), 200_000; host=false)
bench(repeat_periodically(fcc_afm(), (2, 2, 2)), 5_000)
