# Copies a host array to a new array on `backend`
function to_backend(backend, x::AbstractArray)
    d = KA.allocate(backend, eltype(x), size(x))
    copyto!(d, convert(Array{eltype(x)}, x))
    return d
end

# The data of `Sunny.FormFactor` without its descriptive strings, so that it is
# isbits
struct FormFactorDevice
    j0        :: Sunny.ExpandedBesselIntegral
    j2        :: Sunny.ExpandedBesselIntegral
    j2_weight :: Float64
end

FormFactorDevice(ff::Sunny.FormFactor) = FormFactorDevice(ff.j0, ff.j2, ff.j2_weight)

# Mirrors `Sunny.compute_form_factor`
function compute_form_factor(ff::FormFactorDevice, q2_absolute)
    (; j0, j2, j2_weight) = ff
    (j0.A == j0.B == j0.C == j0.D == 0) && (j0.E == 1) && iszero(j2_weight) && return 1.0
    s2 = q2_absolute / (4π)^2
    J0 = Sunny.compute_gaussian_expansion(j0, s2)
    iszero(j2_weight) && return J0
    return J0 + j2_weight * Sunny.compute_gaussian_expansion(j2, s2) * s2
end

struct SpinWaveTheoryDevice{B, D <: NamedTuple}
    swt          :: Sunny.SpinWaveTheory
    backend      :: B
    eigensolver  :: Symbol
    measure_mode :: Symbol  # :combiner, :constant_forms, :forms or :none
    data         :: D
end

# Whether an extension provides `Sunny.vendor_bogoliubov!` for `backend`
function has_vendor_bogoliubov(backend)
    H = KA.allocate(backend, ComplexF64, (2, 2, 1))
    E = KA.allocate(backend, Float64, (2, 1))
    status = KA.allocate(backend, Int32, 1)
    return applicable(Sunny.vendor_bogoliubov!, H, E, H, 1, status)
end

# The Jacobi kernel uses one thread per wavevector, with strided memory access,
# and its cost grows as n³ for n×n matrices. On an NVIDIA P5000, it outpaced
# LAPACK on the host by 4× for n = 8, but lagged 10× for n = 64. It lagged
# cuSOLVER for all n, e.g. 2.9 s vs 2.2 s for 200k matrices with n = 8, and 8.9
# s vs 1.6 s for 100k matrices with n = 16.
function resolve_eigensolver(backend, eigensolver, n)
    if isnothing(eigensolver)
        backend isa KA.CPU && return :host
        has_vendor_bogoliubov(backend) && return :vendor
        return n <= 8 ? :jacobi : :host
    end
    eigensolver in (:host, :jacobi, :vendor) || error("Unknown eigensolver $(repr(eigensolver)); use :host, :jacobi or :vendor")
    if eigensolver == :vendor && !has_vendor_bogoliubov(backend)
        error("No vendor eigensolver for $backend; load its package (e.g. CUDA) or use :jacobi")
    end
    return eigensolver
end

function Base.show(io::IO, ::MIME"text/plain", swt_d::SpinWaveTheoryDevice)
    printstyled(io, "SpinWaveTheoryDevice ", Sunny.mode_to_str(swt_d.swt.sys), "\n"; bold=true, color=:underline)
    println(io, "  ", Sunny.nsites(swt_d.swt.sys), " sites on ", swt_d.backend, ", eigensolver ", repr(swt_d.eigensolver),
            ", measure ", repr(swt_d.measure_mode))
end

"""
    to_device(swt::SpinWaveTheory; backend=KernelAbstractions.CPU(), eigensolver=nothing)

Copies the data of `swt` that [`intensities`](@ref) needs to the
KernelAbstractions `backend`, e.g., `CUDABackend()`. The calculation of
`intensities(to_device(swt), qpts; ...)` then runs on that backend.

The batched `eigensolver` for the Bogoliubov transformation may be `:host`
(LAPACK on the host), `:vendor` (e.g. cuSOLVER, the default when an extension
provides it), or `:jacobi` (a portable kernel, efficient for small matrices).
The default is `:host` for `KernelAbstractions.CPU()`. Otherwise it is
`:vendor` if available, else `:jacobi` for at most 4 bands and `:host` beyond.

The measure's combiner is evaluated on device if it is `isbits` and compiles
for the backend. Otherwise, its forms (see `Sunny.measure_forms`) are evaluated
on the host, once if they are independent of the wavevector, and otherwise per
wavevector.
"""
function Sunny.to_device(swt::Sunny.SpinWaveTheory; backend=KA.CPU(), eigensolver=nothing)
    isnothing(swt.ewald) || error("Dipole-dipole interactions are not yet supported on device")

    (; sys, measure, quadratic_table, observable_coefs) = swt
    (; rowptr, cols, offset_index, vals, offsets) = quadratic_table
    Na = Sunny.nsites(sys)
    Nparts = Sunny.num_parts_per_unit(measure)

    # Fourier positions and form factors of each observable, as in
    # `Sunny.observable_prefactor`
    sites = Sunny.eachsite(sys)
    positions = [Sunny.global_position_at(sys, sites[i]) + sys.crystal.latvecs * measure.offsets[Sunny.to_atom(sites[i]), p]
                 for i in 1:Na, p in 1:Nparts]
    formfactors = [FormFactorDevice(measure.formfactors[μ, Sunny.to_atom(sites[i]), p])
                   for μ in 1:Sunny.num_observables(measure), i in 1:Na, p in 1:Nparts]

    corr_pairs = Sunny.SVector{length(measure.corr_pairs)}(measure.corr_pairs)
    (measure_mode, forms) = resolve_measure_mode(backend, measure, corr_pairs)

    data = (;
        corr_pairs,
        forms,
        rowptr       = to_backend(backend, rowptr),
        cols         = to_backend(backend, cols),
        offset_index = to_backend(backend, offset_index),
        vals         = to_backend(backend, vals),
        offsets      = to_backend(backend, offsets),
        coefs        = to_backend(backend, observable_coefs),
        positions    = to_backend(backend, positions),
        formfactors  = to_backend(backend, formfactors),
    )
    return SpinWaveTheoryDevice(swt, backend, resolve_eigensolver(backend, eigensolver, 2Sunny.nbands(swt)), measure_mode, data)
end

Sunny.to_device(x; _...) = error("`to_device` is not supported for $(typeof(x))")

# Whether `residues_combiner!` compiles for `backend` with the combiner of
# `measure`, and reproduces it on the host for sample correlations
function combiner_runs_on(backend, measure, corr_pairs)
    (; combiner) = measure
    isbits(combiner) || return false
    Nobs = Sunny.num_observables(measure)
    K = Sunny.num_real_components(eltype(measure))
    A_h = reshape([ComplexF64(μ, n) for μ in 1:Nobs, n in 1:2], Nobs, 2, 1)
    q = Vec3(0.13, 0.41, 0.77)
    try
        disp = KA.allocate(backend, Float64, (2, 1))
        R = KA.allocate(backend, Float64, (K, 2, 1))
        E = to_backend(backend, [1.0 -1.0]')
        A = to_backend(backend, A_h)
        residues_combiner!(backend)(disp, R, E, A, to_backend(backend, [q]), combiner, corr_pairs, 0, 1.0; ndrange=(2, 1))
        KA.synchronize(backend)
        expected = [s * x for (n, s) in ((1, 1), (2, -1))
                    for x in Sunny.real_components(combiner(q, [A_h[μ, n, 1] * conj(A_h[ν, n, 1]) for (μ, ν) in corr_pairs]))]
        return vec(Array(R)) ≈ expected
    catch err
        @debug "Combiner cannot run on $backend" exception=err
        return false
    end
end

# How the residues evaluate the measure, and its forms if they are independent
# of the wavevector. Forms that error, e.g. for a wavevector outside a scattering
# plane, are taken to depend on it.
function resolve_measure_mode(backend, measure, corr_pairs)
    Sunny.num_observables(measure) == 0 && return (:none, nothing)
    combiner_runs_on(backend, measure, corr_pairs) && return (:combiner, nothing)
    qs = [Vec3(0.13, 0.41, 0.77), Vec3(-0.6, 0.25, 0.1), Vec3(0.3, -0.8, 0.45)]
    Ms = try
        [Sunny.measure_forms(measure, q) for q in qs]
    catch
        return (:forms, nothing)
    end
    all(==(Ms[1]), Ms) || return (:forms, nothing)
    M = reduce((a, b) -> cat(a, b; dims=3), Ms[1]; init=zeros(ComplexF64, size(Ms[1][1])..., 0))
    return (:constant_forms, to_backend(backend, reshape(M, size(M)..., 1)))
end
