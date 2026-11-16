# Benchmark of the batched Bogoliubov eigensolvers of KAExt on an AMD GPU, to
# choose the default eigensolver per matrix size, and to measure the workspace
# that rocSOLVER takes from the rocBLAS handle.
#
# 1. Solvers alone, on nb random positive definite n×n Hamiltonians:
#      :hegvd   rocSOLVER's batched hegvd, i.e. `eigensolver=:vendor`
#      :hegvj   rocSOLVER's batched Jacobi-based hegvj, a candidate for :vendor
#      :jacobi  the portable Jacobi kernel of KAExt
#      :host    LAPACK on the host, threaded, with copies to and from the device
#    Each run checks the energies against `Sunny.bogoliubov!` on the host.
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

# As `Sunny.vendor_bogoliubov!` of KAAMDGPUExt, with rocSOLVER's Jacobi-based
# hegvj in place of hegvd. Its status codes other than 0 are left as rocSOLVER
# reports them. Returns the number of sweeps per Hamiltonian.
hegvj(::Type{Float64}) = rocSOLVER.rocsolver_zhegvj_strided_batched
hegvj(::Type{Float32}) = rocSOLVER.rocsolver_chegvj_strided_batched
function bogoliubov_hegvj!(T::ROCArray{Complex{F}, 3}, E, H, nb, status) where F
    n = size(H, 1)
    L = n ÷ 2
    Tb = view(T, :, :, 1:nb)
    fill!(Tb, 0)
    view(reshape(Tb, n^2, nb), 1:n+1:n^2, :) .= ROCArray(Complex{F}[ones(L); -ones(L)])
    λ = ROCArray{F}(undef, n, nb)
    residual = ROCArray{F}(undef, nb)
    n_sweeps = ROCArray{Int32}(undef, nb)
    # abstol = 0 selects rocSOLVER's default tolerance
    hegvj(F)(rocBLAS.handle(), rocSOLVER.rocblas_eform_ax, rocSOLVER.rocblas_evect_original, rocBLAS.rocblas_fill_lower,
             n, T, n, n^2, H, n, n^2, zero(F), residual, Int32(100), n_sweeps, λ, n, status, nb)
    Sunny.finish_vendor_bogoliubov!(T, E, λ, nb, status)
    AMDGPU.unsafe_free!(λ)
    AMDGPU.unsafe_free!(residual)
    return n_sweeps
end

# Bytes of device workspace that the rocBLAS handle holds, which grows to the
# largest that rocSOLVER has requested
function rocblas_workspace()
    sz = Ref{Csize_t}(0)
    rocBLAS.rocblas_get_device_memory_size(rocBLAS.handle(), sz)
    return Int(sz[])
end

const solvers = [
    :hegvd => (T, E, H, W, status, nb) -> Sunny.vendor_bogoliubov!(T, E, H, nb, status),
    :hegvj => (T, E, H, W, status, nb) -> bogoliubov_hegvj!(T, E, H, nb, status),
    :jacobi => (T, E, H, W, status, nb) -> KAExt.bogoliubov_jacobi!(backend)(T, E, H, W, status; ndrange=nb),
    # swt and qs only enter the message of an InstabilityError
    :host => (T, E, H, W, status, nb) -> KAExt.bogoliubov_host!(T, E, H, nb, nothing, nothing; threaded=true),
]
const active_solvers = on_amd ? solvers : filter(s -> s.first in (:jacobi, :host), solvers)

free_memory() = on_amd ? AMDGPU.free() : Int(Sys.free_memory())

function bench_solvers(n, F, nb)
    H0_h = random_hamiltonians(n, nb)
    H0 = to_dev(Complex{F}.(H0_h))
    (H, T, W) = (similar(H0), similar(H0), similar(H0))
    E = KA.allocate(backend, F, (n, nb))
    status = KA.allocate(backend, Int32, nb)
    t_copy = best_time(() -> copyto!(H, H0))

    # Reference energies on the host, in Float64, for a sample of the batch
    sample = unique(round.(Int, range(1, nb, 16)))
    E_ref = [Sunny.bogoliubov!(zeros(ComplexF64, n, n), copy(H0_h[:, :, b])) for b in sample]

    for (name, solve!) in active_solvers
        result = try
            t = best_time() do
                copyto!(H, H0)
                fill!(status, 0)
                solve!(T, E, H, W, status, nb)
            end - t_copy
            nfail = count(!=(0), Array(status))
            E_h = Array(E)
            err = maximum(maximum(abs, E_h[:, b] - e) / maximum(abs, e) for (b, e) in zip(sample, E_ref))
            ws = name in (:hegvd, :hegvj) ? @sprintf("%8.2f", rocblas_workspace() / nb / 1024) : "       -"
            @sprintf("%10.2f %10.3f %9.1e %8d %s", 1e3t, 1e6t / nb, err, nfail, ws)
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
println("   n F            nb  solver    ms total  µs/matrix   rel.err  #status  ws KiB")
for n in ns, F in precisions, nb in nbs
    nb_max = free_memory() ÷ 2 ÷ (4n^2 * sizeof(Complex{F}))
    if nb > nb_max
        @printf("%4d %-8s %8d  skipped: exceeds the memory cap of %d matrices\n", n, F, nb, nb_max)
        continue
    end
    bench_solvers(n, F, nb)
end
println()

println("2. intensities end to end, with a Gaussian kernel on 200 energies")
for mode in (:dipole, :SUN)
    bench_intensities(mode)
end
