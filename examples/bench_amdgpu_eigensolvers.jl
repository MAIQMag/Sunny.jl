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
#    and by the error of the spectral sum Σₖ tₖ tₖ† Eₖ over modes of positive
#    energy, which, as intensities, is independent of the basis that a solver
#    picks for degenerate modes.
# 1b. As 1, on the Hamiltonians of spin wave theory for the fcc antiferromagnet
#    of 2, which have degenerate modes, unlike random Hamiltonians.
# 2. `intensities` end to end, for an fcc antiferromagnet in :dipole (n = 8)
#    and :SUN (n = 40) modes, per eigensolver of `to_device`.
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

# Σₖ tₖ tₖ† Eₖ over the modes k of positive energy
function spectral_sum(T, E)
    pos = findall(>(0), E)
    Tp = T[:, pos]
    return Tp * Diagonal(E[pos]) * Tp'
end

# The largest residuals over the `sample` of the batch of para-unitarity,
# T† Ĩ T = Ĩ, and of the eigenproblem, H T = Ĩ T diag(E), relative to the norms
# of H and T, and the largest relative error of `spectral_sum` against the
# reference `S_ref`
function eigenvector_residuals(T, E, H, sample, S_ref)
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
    serr = maximum(zip(sample, S_ref)) do (b, S)
        norm(spectral_sum(ComplexF64.(T[:, :, b]), Float64.(E[:, b])) - S) / norm(S)
    end
    return (punit, resid, serr)
end

free_memory() = on_amd ? AMDGPU.free() : Int(Sys.free_memory())

function bench_solvers(H0_h, F)
    (n, _, nb) = size(H0_h)
    H0 = to_dev(Complex{F}.(H0_h))
    (H, T, W) = (similar(H0), similar(H0), similar(H0))
    E = KA.allocate(backend, F, (n, nb))
    status = KA.allocate(backend, Int32, nb)
    t_copy = best_time(() -> copyto!(H, H0))

    # Reference energies and spectral sums on the host, in Float64, for a
    # sample of the batch
    sample = unique(round.(Int, range(1, nb, 16)))
    (E_ref, S_ref) = (Vector{Float64}[], Matrix{ComplexF64}[])
    for b in sample
        T_ref = zeros(ComplexF64, n, n)
        e = Sunny.bogoliubov!(T_ref, copy(H0_h[:, :, b]))
        push!(E_ref, e)
        push!(S_ref, spectral_sum(T_ref, e))
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
            (punit, resid, serr) = eigenvector_residuals(T_h, E_h, H0_h, sample, S_ref)
            ws = name in (:vendor, :hegvd, :hegvj) ? @sprintf("%8.2f", rocblas_workspace() / nb / 1024) : "       -"
            @sprintf("%10.2f %10.3f %9.1e %9.1e %9.1e %9.1e %8d %s", 1e3t, 1e6t / nb, err, punit, resid, serr, nfail, ws)
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

# Hamiltonians of spin wave theory at `nq` random wavevectors
function swt_hamiltonians(swt, nq)
    n = 2Sunny.nbands(swt)
    H = Array{ComplexF64}(undef, n, n, nq)
    rng = Xoshiro(2)
    for b in 1:nq
        q_supercell = Sunny.to_supercell_rlu(swt.sys, rand(rng, 3))
        Sunny.hamiltonian_matrix!(view(H, :, :, b), swt, q_supercell)
    end
    return H
end

function bench_intensities(mode)
    sys = fcc_afm(mode)
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    qs = [rand(Xoshiro(1), 3) for _ in 1:nqs]
    kwargs = (; energies=range(0, 12, 200), kernel=gaussian(fwhm=0.5))
    # Warm up on a few wavevectors, so that the time excludes compilation
    intensities(swt, qs[1:10]; kwargs...)
    t_cpu = @elapsed res = intensities(swt, qs; kwargs...)
    n = 2Sunny.nbands(swt)
    @printf("%-7s n = %2d, %d wavevectors, CPU (%d threads) %.2f s\n", mode, n, nqs, Threads.nthreads(), t_cpu)
    eigensolvers = on_amd ? (:vendor, :jacobi, :host) : (:jacobi, :host)
    for precision in precisions, eigensolver in eigensolvers
        result = try
            swt_d = to_device(swt; backend, eigensolver, precision)
            t = best_time(() -> intensities(swt_d, qs; kwargs...); reps=3)
            res_d = intensities(swt_d, qs; kwargs...)
            err = norm(res_d.data - res.data) / norm(res.data)
            @sprintf("%9.3f s  %6.1fx vs CPU  rel. err %.1e", t, t_cpu / t, err)
        catch e
            "failed: " * first(sprint(showerror, e), 120)
        end
        @printf("  %-8s %-7s %s\n", precision, eigensolver, result)
    end
end

println("Julia $VERSION, $(Threads.nthreads()) threads, backend $backend")
println("BLAS: ", join([basename(lib.libname) for lib in BLAS.get_config().loaded_libs], ", "))
on_amd ? println("Device: ", AMDGPU.device()) : println("No functional AMD GPU: smoke test on the CPU backend")
println()

println("1. Eigensolvers alone. Times exclude the copy of H; ws is rocBLAS workspace per matrix.")
const header = "   n F            nb  solver    ms total  µs/matrix   rel.err   T'ĨT-Ĩ    HT-ĨTE  Σtt†E err  #status  ws KiB"
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

println("1b. As 1, on the Hamiltonians of spin wave theory for the fcc antiferromagnet")
println(header)
for mode in (:dipole, :SUN), F in precisions
    sys = fcc_afm(mode)
    swt = SpinWaveTheory(sys; measure=ssf_perp(sys))
    bench_solvers(swt_hamiltonians(swt, nbs[end] ÷ 10), F)
end
println()

println("2. intensities end to end, with a Gaussian kernel on 200 energies")
for mode in (:dipole, :SUN)
    bench_intensities(mode)
end
