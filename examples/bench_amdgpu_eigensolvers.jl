# Benchmark of the batched Bogoliubov eigensolvers of KAExt on an AMD GPU, to
# choose the default eigensolver per matrix size, and to measure the workspace
# that rocSOLVER takes from the rocBLAS handle.
#
# 1. Solvers alone, on nb random positive definite n×n Hamiltonians:
#      :vendor  `Sunny.vendor_bogoliubov!`, i.e. `eigensolver=:vendor`, which
#               calls hegvj for n ≤ 40 and hegvd beyond
#      :hegvd   rocSOLVER's batched hegvd
#      :hegvj   rocSOLVER's batched Jacobi-based hegvj
#      :jacobi  the portable Jacobi kernel of KAExt
#      :host    LAPACK on the host, threaded, with copies to and from the device
#    Each run checks the energies against `Sunny.bogoliubov!` on the host, and
#    the eigenvectors T by the residuals of T† Ĩ T = Ĩ and H T = Ĩ T diag(E),
#    and by the errors of the spectral sums Σₖ tₖ tₖ† Eₖ over modes of positive
#    energy, and over those of negative energy, which, as intensities, are
#    independent of the basis that a solver picks for degenerate modes.
# 1b. As 1, on the Hamiltonians of spin wave theory for the fcc antiferromagnet
#    of 2, at its wavevectors, which have degenerate modes, unlike random
#    Hamiltonians. As in `intensities`, the Hamiltonians are built on device, in
#    the precision of the run, and the solvers and the host reference take them
#    as they are. Their largest difference from those built on the host in
#    Float64 is reported. The checks cover every Hamiltonian, not a sample, and the
#    vendor solvers report the wavevector of the largest error of the spectral
#    sum, with the smallest gap between its positive energies.
# 2. `intensities` end to end, for an fcc antiferromagnet in :dipole (n = 8)
#    and :SUN (n = 40) modes, per eigensolver of `to_device`. On AMD, :vendor
#    runs twice, with hegvj for n ≤ 40 (the default) and with hegvd for all n.
#
# Setup, from the Sunny.jl root:
#   julia --project=bench -e 'using Pkg; Pkg.develop(path="."); Pkg.add(["AMDGPU", "KernelAbstractions", "MKL"])'
# MKL is the BLAS that `load_blas_for_threading` loads on x86_64, and
# AppleAccelerate on macOS, in place of OpenBLAS, which scales poorly under
# threads on small matrices.
# Run, with threads for the :host eigensolver and the CPU baseline:
#   julia --project=bench -t 16 examples/bench_amdgpu_eigensolvers.jl          # full
#   julia --project=bench -t 16 examples/bench_amdgpu_eigensolvers.jl quick    # smaller sizes
# Without a functional AMD GPU, only :jacobi and :host run, on the CPU backend
# and at the quick sizes, as a smoke test of this script.

using Sunny, LinearAlgebra, Random, Printf
import AMDGPU, KernelAbstractions as KA
using AMDGPU: ROCArray, rocBLAS, rocSOLVER

# For the threaded :host eigensolver and the CPU baseline
load_blas_for_threading()

const KAExt = Base.get_extension(Sunny, :KAExt)
const on_amd = AMDGPU.functional()
const backend = on_amd ? AMDGPU.ROCBackend() : KA.CPU()
const quick = "quick" in ARGS || !on_amd

# Sizes. A batch is capped so that its four n×n×nb device arrays fill at most
# half of the free device memory.
const ns = quick ? [8, 16, 40] : [8, 16, 40, 64]
const nbs = quick ? [1_000, 10_000] : [1_000, 10_000, 100_000, 400_000]
const nqs = quick ? 10_000 : 100_000
const precisions = [Float64, Float32]

# Best wall time in seconds of `f()` over `reps` calls with the device
# synchronized, after one warm-up call for compilation. If the first timed call
# takes more than 1 s, it is the only one.
function best_time(f; reps=5)
    function timed()
        t = time_ns(); f(); KA.synchronize(backend)
        return (time_ns() - t) / 1e9
    end
    KA.synchronize(backend)
    f(); KA.synchronize(backend)
    t1 = timed()
    t1 > 1 && return t1
    return min(t1, minimum(_ -> timed(), 2:reps; init=Inf))
end

to_dev(A) = copyto!(KA.allocate(backend, eltype(A), size(A)), A)

# Random positive definite Hamiltonians with eigenvalues spread over about
# [1, 5], standing in for those of spin wave theory
function random_hamiltonians(n, nb; rng=Xoshiro(0))
    H = Array{ComplexF64}(undef, n, n, nb)
    for b in 1:nb
        A = randn(rng, ComplexF64, n, n)
        H[:, :, b] .= A * A' / n + I
    end
    return H
end

const AMDExt = Base.get_extension(Sunny, :KAAMDGPUExt)

# As `Sunny.vendor_bogoliubov!` of KAAMDGPUExt, with the rocSOLVER call
# `solve!(T, H, λ, status, nb)` chosen, e.g. `AMDExt.hegvd!` or `AMDExt.hegvj!`
function bogoliubov_with!(solve!, T::ROCArray{Complex{F}, 3}, E, H, nb, status) where F
    n = size(H, 1)
    L = n ÷ 2
    Tb = view(T, :, :, 1:nb)
    fill!(Tb, 0)
    view(reshape(Tb, n^2, nb), 1:n+1:n^2, :) .= ROCArray(Complex{F}[ones(L); -ones(L)])
    λ = ROCArray{F}(undef, n, nb)
    solve!(T, H, λ, status, nb)
    sb = view(status, 1:nb)
    sb .= ifelse.(sb .> n, sb .- Int32(n), ifelse.(sb .> 0, Int32(-3), sb))
    Sunny.finish_vendor_bogoliubov!(T, E, λ, nb, status)
    AMDGPU.unsafe_free!(λ)
end

# Bytes of device workspace that the rocBLAS handle holds, which grows to the
# largest that rocSOLVER has requested
function rocblas_workspace()
    sz = Ref{Csize_t}(0)
    rocBLAS.rocblas_get_device_memory_size(rocBLAS.handle(), sz)
    return Int(sz[])
end

const solvers = [
    :vendor => (T, E, H, W, status, nb) -> Sunny.vendor_bogoliubov!(T, E, H, nb, status),
    :hegvd => (T, E, H, W, status, nb) -> bogoliubov_with!(AMDExt.hegvd!, T, E, H, nb, status),
    :hegvj => (T, E, H, W, status, nb) -> bogoliubov_with!(AMDExt.hegvj!, T, E, H, nb, status),
    :jacobi => (T, E, H, W, status, nb) -> KAExt.bogoliubov_jacobi!(backend)(T, E, H, W, status; ndrange=nb),
    # swt and qs only enter the message of an InstabilityError
    :host => (T, E, H, W, status, nb) -> KAExt.bogoliubov_host!(T, E, H, nb, nothing, nothing; threaded=true),
]
const active_solvers = on_amd ? solvers : filter(s -> s.first in (:jacobi, :host), solvers)

# Σₖ tₖ tₖ† Eₖ over the modes k of positive energy, or of negative energy if
# `sign` is -1
function spectral_sum(T, E; sign=1)
    pos = findall(e -> sign * e > 0, E)
    Tp = T[:, pos]
    return Tp * Diagonal(E[pos]) * Tp'
end

# The largest residuals over the `sample` of the batch of para-unitarity,
# T† Ĩ T = Ĩ, and of the eigenproblem, H T = Ĩ T diag(E), relative to the norms
# of H and T, and the largest relative error of `spectral_sum` against the
# reference `S_ref`, and of that over negative energies against `Sneg_ref`
function eigenvector_residuals(T, E, H, sample, S_ref, Sneg_ref)
    n = size(H, 1)
    L = n ÷ 2
    Ĩ = Diagonal([ones(L); -ones(L)])
    punit = maximum(sample) do b
        Tb = ComplexF64.(T[:, :, b])
        maximum(abs, Tb' * Ĩ * Tb - Ĩ)
    end
    resid = maximum(sample) do b
        (Tb, Hb) = (ComplexF64.(T[:, :, b]), H[:, :, b])
        norm(Hb * Tb - Ĩ * Tb * Diagonal(E[:, b])) / (norm(Hb) * norm(Tb))
    end
    serrs = map(zip(sample, S_ref)) do (b, S)
        norm(spectral_sum(ComplexF64.(T[:, :, b]), Float64.(E[:, b])) - S) / norm(S)
    end
    (serr, i) = findmax(serrs)
    snegerr = maximum(zip(sample, Sneg_ref)) do (b, S)
        norm(spectral_sum(ComplexF64.(T[:, :, b]), Float64.(E[:, b]); sign=-1) - S) / norm(S)
    end
    return (punit, resid, serr, snegerr, i)
end

free_memory() = on_amd ? AMDGPU.free() : Int(Sys.free_memory())

# The checks cover a sample of 16 Hamiltonians, or all of them if `qs`, their
# wavevectors, are given
function bench_solvers(H0_h, F; qs=nothing)
    (n, _, nb) = size(H0_h)
    H0 = to_dev(Complex{F}.(H0_h))
    (H, T, W) = (similar(H0), similar(H0), similar(H0))
    E = KA.allocate(backend, F, (n, nb))
    status = KA.allocate(backend, Int32, nb)
    t_copy = best_time(() -> copyto!(H, H0))

    # Reference energies and spectral sums on the host, in Float64, for a
    # sample of the batch
    sample = isnothing(qs) ? unique(round.(Int, range(1, nb, 16))) : collect(1:nb)
    (E_ref, S_ref, Sneg_ref) = (Vector{Float64}[], Matrix{ComplexF64}[], Matrix{ComplexF64}[])
    for b in sample
        T_ref = zeros(ComplexF64, n, n)
        e = Sunny.bogoliubov!(T_ref, copy(H0_h[:, :, b]))
        push!(E_ref, e)
        push!(S_ref, spectral_sum(T_ref, e))
        push!(Sneg_ref, spectral_sum(T_ref, e; sign=-1))
    end

    for (name, solve!) in active_solvers
        result = try
            t = best_time() do
                copyto!(H, H0)
                fill!(status, 0)
                solve!(T, E, H, W, status, nb)
            end - t_copy
            nfail = count(!=(0), Array(status))
            E_h = Array(E)
            T_h = Array(T)
            err = maximum(maximum(abs, E_h[:, b] - e) / maximum(abs, e) for (b, e) in zip(sample, E_ref))
            (punit, resid, serr, snegerr, i) = eigenvector_residuals(T_h, E_h, H0_h, sample, S_ref, Sneg_ref)
            if !isnothing(qs) && name in (:vendor, :hegvd, :hegvj)
                pos = sort(filter(>(0), E_ref[i]))
                gap = minimum(diff(pos)) / maximum(pos)
                worst = @sprintf("\n%33s worst q = [%s], smallest relative gap of positive energies %.1e", "",
                                 join((@sprintf("%.4f", x) for x in qs[sample[i]]), ", "), gap)
            else
                worst = ""
            end
            ws = name in (:vendor, :hegvd, :hegvj) ? @sprintf("%8.2f", rocblas_workspace() / nb / 1024) : "       -"
            @sprintf("%10.2f %10.3f %9.1e %9.1e %9.1e %9.1e %9.1e %8d %s", 1e3t, 1e6t / nb, err, punit, resid, serr, snegerr, nfail, ws) * worst
        catch e
            "failed: " * first(sprint(showerror, e), 120)
        end
        @printf("%4d %-8s %8d  %-7s %s\n", n, F, nb, name, result)
    end
    foreach(A -> KA.unsafe_free!(A), (H0, H, T, W, E, status))
    GC.gc()
end

# Néel order on the fcc lattice, with cubic anisotropy, as in test/test_kaext.jl.
# The Hamiltonians are 8×8 in :dipole mode and 40×40 in :SUN mode.
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

# Hamiltonians of spin wave theory at the wavevectors `qs`, built on the host in
# Float64
function swt_hamiltonians(swt, qs)
    n = 2Sunny.nbands(swt)
    H = Array{ComplexF64}(undef, n, n, length(qs))
    for (b, q) in enumerate(qs)
        Sunny.hamiltonian_matrix!(view(H, :, :, b), swt, Sunny.to_supercell_rlu(swt.sys, q))
    end
    return H
end

# As `swt_hamiltonians`, built on device in precision F by the kernel of
# `intensities`, and returned to the host
function swt_hamiltonians_device(swt, qs, F)
    swt_d = to_device(swt; backend, precision=F)
    (; data) = swt_d
    n = 2Sunny.nbands(swt)
    H = KA.allocate(backend, Complex{F}, (n, n, length(qs)))
    qs_supercell = KAExt.to_backend(backend, F, [Sunny.to_supercell_rlu(swt.sys, q) for q in qs])
    KAExt.build_hamiltonian!(backend)(H, data.rowptr, data.cols, data.offset_index, data.vals, data.offsets,
                                      qs_supercell, F(swt.regularization); ndrange=(n, length(qs)))
    KA.synchronize(backend)
    return Array(H)
end

# The wavevectors of section 2
function intensity_qs()
    rng = Xoshiro(1)
    return [rand(rng, 3) for _ in 1:nqs]
end

function bench_intensities(mode)
    sys = fcc_afm(mode)
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    qs = intensity_qs()
    kwargs = (; energies=range(0, 12, 200), kernel=gaussian(fwhm=0.5))
    # Warm up on a few wavevectors, so that the time excludes compilation
    intensities(swt, qs[1:10]; kwargs...)
    t_cpu = @elapsed res = intensities(swt, qs; kwargs...)
    n = 2Sunny.nbands(swt)
    @printf("%-7s n = %2d, %d wavevectors, CPU (%d threads) %.2f s\n", mode, n, nqs, Threads.nthreads(), t_cpu)
    # :vendor with hegvd, through a cutoff of 0 for hegvj
    labels = on_amd ? (:vendor, :vendor_hegvd, :jacobi, :host) : (:jacobi, :host)
    for precision in precisions, label in labels
        eigensolver = label == :vendor_hegvd ? :vendor : label
        on_amd && (AMDExt.HEGVJ_MAX_N[] = label == :vendor_hegvd ? 0 : 40)
        result = try
            swt_d = to_device(swt; backend, eigensolver, precision)
            t = best_time(() -> intensities(swt_d, qs; kwargs...); reps=3)
            res_d = intensities(swt_d, qs; kwargs...)
            err = norm(res_d.data - res.data) / norm(res.data)
            # The wavevector of the largest error, and its share of the error
            errs = [norm(res_d.data[:, iq] - res.data[:, iq]) for iq in axes(res.data, 2)]
            (e, iq) = findmax(errs)
            @sprintf("%9.3f s  %6.1fx vs CPU  rel. err %.1e,  worst q = [%s] with %.0f%% of the error",
                     t, t_cpu / t, err, join((@sprintf("%.4f", x) for x in qs[iq]), ", "), 100 * e^2 / sum(abs2, errs))
        catch e
            "failed: " * first(sprint(showerror, e), 120)
        end
        @printf("  %-8s %-12s %s\n", precision, label, result)
    end
end

println("Julia $VERSION, $(Threads.nthreads()) threads, backend $backend")
println("BLAS: ", join([basename(lib.libname) for lib in BLAS.get_config().loaded_libs], ", "))
on_amd ? println("Device: ", AMDGPU.device()) : println("No functional AMD GPU: smoke test on the CPU backend")
println()

println("1. Eigensolvers alone. Times exclude the copy of H; ws is rocBLAS workspace per matrix.")
const header = "   n F            nb  solver    ms total  µs/matrix   rel.err   T'ĨT-Ĩ    HT-ĨTE  Σ⁺tt†E err Σ⁻tt†E err #status  ws KiB"
println(header)
for n in ns, F in precisions, nb in nbs
    nb_max = free_memory() ÷ 2 ÷ (4n^2 * sizeof(Complex{F}))
    if nb > nb_max
        @printf("%4d %-8s %8d  skipped: exceeds the memory cap of %d matrices\n", n, F, nb, nb_max)
        continue
    end
    bench_solvers(random_hamiltonians(n, nb), F)
end
println()

# At most 20k wavevectors, for the time of the checks on the host
const qs_1b = intensity_qs()[1:min(nqs, 20_000)]
println("1b. As 1, on the Hamiltonians of spin wave theory for the fcc antiferromagnet,")
println("    at the first $(length(qs_1b)) wavevectors of 2. The checks cover every Hamiltonian.")
println(header)
for mode in (:dipole, :SUN), F in precisions
    sys = fcc_afm(mode)
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    H_d = swt_hamiltonians_device(swt, qs_1b, F)
    H_h = swt_hamiltonians(swt, qs_1b)
    dH = maximum(b -> norm(H_d[:, :, b] - H_h[:, :, b]) / norm(H_h[:, :, b]), axes(H_h, 3))
    herm = maximum(b -> norm(H_d[:, :, b] - H_d[:, :, b]') / norm(H_d[:, :, b]), axes(H_d, 3))
    @printf("%4d %-8s  device H vs host H: %.1e,  non-Hermitian part of device H: %.1e\n",
            size(H_d, 1), F, dH, herm)
    bench_solvers(ComplexF64.(H_d), F; qs=qs_1b)
end
println()

println("2. intensities end to end, with a Gaussian kernel on 200 energies")
for mode in (:dipole, :SUN)
    bench_intensities(mode)
end
