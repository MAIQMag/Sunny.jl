# Copies a host array to a new array on `backend`
function to_backend(backend, x::AbstractArray)
    d = KA.allocate(backend, eltype(x), size(x))
    copyto!(d, convert(Array{eltype(x)}, x))
    return d
end

# Copies a host array to `backend`, with its floating point numbers converted to
# precision F
to_backend(backend, ::Type{F}, x::AbstractArray) where F = to_backend(backend, map(v -> with_precision(F, v), x))

# The value x with its floating point numbers converted to precision F
with_precision(::Type{F}, x::AbstractFloat) where F = F(x)
with_precision(::Type{F}, x::Complex) where F = Complex{F}(x)
with_precision(::Type{F}, x::Integer) where F = x
with_precision(::Type{F}, x::Sunny.SArray) where F = map(v -> with_precision(F, v), x)

# The type of `with_precision(F, x)` for x of type S, e.g. for the values of a
# measure
precision_type(::Type{F}, ::Type{<:AbstractFloat}) where F = F
precision_type(::Type{F}, ::Type{<:Complex}) where F = Complex{F}
precision_type(::Type{F}, ::Type{Sunny.SArray{S, E, N, L}}) where {F, S, E, N, L} = Sunny.SArray{S, precision_type(F, E), N, L}

# The precision of `to_device` for `backend`, unless a backend-specific
# extension overrides it
Sunny.device_precision(_) = Float64

# The coefficients of `Sunny.ExpandedBesselIntegral` in precision F
struct BesselIntegralDevice{F}
    A :: F
    a :: F
    B :: F
    b :: F
    C :: F
    c :: F
    D :: F
    d :: F
    E :: F
end

BesselIntegralDevice{F}(j::Sunny.ExpandedBesselIntegral) where F = BesselIntegralDevice{F}(j.A, j.a, j.B, j.b, j.C, j.c, j.D, j.d, j.E)

# Mirrors `Sunny.compute_gaussian_expansion`
function compute_gaussian_expansion(j::BesselIntegralDevice, s2)
    (; A, a, B, b, C, c, D, d, E) = j
    return A*exp(-a*s2) + B*exp(-b*s2) + C*exp(-c*s2) + D*exp(-d*s2) + E
end

# The data of `Sunny.FormFactor` in precision F, without its descriptive
# strings, so that it is isbits
struct FormFactorDevice{F}
    j0        :: BesselIntegralDevice{F}
    j2        :: BesselIntegralDevice{F}
    j2_weight :: F
end

FormFactorDevice{F}(ff::Sunny.FormFactor) where F =
    FormFactorDevice{F}(BesselIntegralDevice{F}(ff.j0), BesselIntegralDevice{F}(ff.j2), ff.j2_weight)

# Mirrors `Sunny.compute_form_factor`
function compute_form_factor(ff::FormFactorDevice{F}, q2_absolute) where F
    (; j0, j2, j2_weight) = ff
    (j0.A == j0.B == j0.C == j0.D == 0) && (j0.E == 1) && iszero(j2_weight) && return one(F)
    s2 = q2_absolute / F((4π)^2)
    J0 = compute_gaussian_expansion(j0, s2)
    iszero(j2_weight) && return J0
    return J0 + j2_weight * compute_gaussian_expansion(j2, s2) * s2
end

# The data of `swt` on `backend`, with floating point numbers in precision F
struct SpinWaveTheoryDevice{F, B, D <: NamedTuple}
    swt          :: Sunny.SpinWaveTheory
    backend      :: B
    eigensolver  :: Symbol
    measure_mode :: Symbol  # :combiner, :constant_forms, :forms or :none
    data         :: D

    SpinWaveTheoryDevice{F}(swt, backend::B, eigensolver, measure_mode, data::D) where {F, B, D} =
        new{F, B, D}(swt, backend, eigensolver, measure_mode, data)
end

precision_of(::SpinWaveTheoryDevice{F}) where F = F

# Whether an extension provides `Sunny.vendor_bogoliubov!` for `backend` in
# precision F
function has_vendor_bogoliubov(backend, F)
    H = KA.allocate(backend, Complex{F}, (2, 2, 1))
    E = KA.allocate(backend, F, (2, 1))
    status = KA.allocate(backend, Int32, 1)
    return applicable(Sunny.vendor_bogoliubov!, H, E, H, 1, status)
end

# The Jacobi kernel uses one thread per wavevector, with strided memory access,
# and its cost grows as n³ for n×n matrices. On an NVIDIA P5000, it outpaced
# LAPACK on the host by 4× for n = 8, but lagged 10× for n = 64. It lagged
# cuSOLVER for all n, e.g. 2.9 s vs 2.2 s for 200k matrices with n = 8, and 8.9
# s vs 1.6 s for 100k matrices with n = 16. On an AMD MI100, it outpaced LAPACK
# on the host (MKL, 16 threads) by 3.3× for n = 8, and matched it for n = 16
# and 40. It lagged rocSOLVER for all n, e.g. 1.7 vs 0.37 µs per matrix for
# n = 8, and 337 vs 44 µs for n = 40, with 100k matrices in Float64.
function resolve_eigensolver(backend, eigensolver, n, F)
    if isnothing(eigensolver)
        backend isa KA.CPU && return :host
        has_vendor_bogoliubov(backend, F) && return :vendor
        return n <= 8 ? :jacobi : :host
    end
    eigensolver in (:host, :jacobi, :vendor) || error("Unknown eigensolver $(repr(eigensolver)); use :host, :jacobi or :vendor")
    if eigensolver == :vendor && !has_vendor_bogoliubov(backend, F)
        error("No vendor eigensolver for $backend in $F; load its package (e.g. CUDA or AMDGPU) or use :jacobi")
    end
    return eigensolver
end

function Base.show(io::IO, ::MIME"text/plain", swt_d::SpinWaveTheoryDevice)
    printstyled(io, "SpinWaveTheoryDevice ", Sunny.mode_to_str(swt_d.swt.sys), "\n"; bold=true, color=:underline)
    println(io, "  ", Sunny.nsites(swt_d.swt.sys), " sites on ", swt_d.backend, " in ", precision_of(swt_d),
            ", eigensolver ", repr(swt_d.eigensolver), ", measure ", repr(swt_d.measure_mode))
end

"""
    to_device(swt::SpinWaveTheory; backend=KernelAbstractions.CPU(), eigensolver=nothing,
              precision=Sunny.device_precision(backend))

Copies the data of `swt` that [`intensities`](@ref) needs to the
KernelAbstractions `backend`, e.g., `CUDABackend()`. The calculation of
`intensities(to_device(swt), qpts; ...)` then runs on that backend.

The floating point `precision` of the device data and arithmetic may be
`Float64` or `Float32`. The default is `Float64`, except `Float32` for Metal,
which lacks `Float64`. Intensities returned to the host are `Float64` in either
case.

The batched `eigensolver` for the Bogoliubov transformation may be `:host`
(LAPACK on the host), `:vendor` (cuSOLVER or rocSOLVER, the default when CUDA
or AMDGPU is loaded), or `:jacobi` (a portable kernel, efficient for small
matrices).
The default is `:host` for `KernelAbstractions.CPU()`. Otherwise it is
`:vendor` if available, else `:jacobi` for at most 4 bands and `:host` beyond.

The measure's combiner is evaluated on device if it is `isbits` and compiles
for the backend. Otherwise, its forms (see `Sunny.measure_forms`) are evaluated
on the host, once if they are independent of the wavevector, and otherwise per
wavevector.
"""
function Sunny.to_device(swt::Sunny.SpinWaveTheory; backend=KA.CPU(), eigensolver=nothing,
                         precision::Type=Sunny.device_precision(backend))
    isnothing(swt.ewald) || error("Dipole-dipole interactions are not yet supported on device")
    F = precision
    F in (Float32, Float64) || error("Unsupported precision $F; use Float64 or Float32")

    (; sys, measure, quadratic_table, observable_coefs) = swt
    (; rowptr, cols, offset_index, vals, offsets) = quadratic_table
    Na = Sunny.nsites(sys)
    Nparts = Sunny.num_parts_per_unit(measure)

    # Fourier positions and form factors of each observable, as in
    # `Sunny.observable_prefactor`
    sites = Sunny.eachsite(sys)
    positions = [Sunny.global_position_at(sys, sites[i]) + sys.crystal.latvecs * measure.offsets[Sunny.to_atom(sites[i]), p]
                 for i in 1:Na, p in 1:Nparts]
    formfactors = [FormFactorDevice{F}(measure.formfactors[μ, Sunny.to_atom(sites[i]), p])
                   for μ in 1:Sunny.num_observables(measure), i in 1:Na, p in 1:Nparts]

    corr_pairs = Sunny.SVector{length(measure.corr_pairs)}(measure.corr_pairs)
    (measure_mode, forms) = resolve_measure_mode(backend, F, measure, corr_pairs)

    data = (;
        corr_pairs,
        forms,
        rowptr       = to_backend(backend, rowptr),
        cols         = to_backend(backend, cols),
        offset_index = to_backend(backend, offset_index),
        vals         = to_backend(backend, F, vals),
        offsets      = to_backend(backend, F, offsets),
        coefs        = to_backend(backend, F, observable_coefs),
        positions    = to_backend(backend, F, positions),
        formfactors  = to_backend(backend, formfactors),
    )
    eigensolver = resolve_eigensolver(backend, eigensolver, 2Sunny.nbands(swt), F)
    return SpinWaveTheoryDevice{F}(swt, backend, eigensolver, measure_mode, data)
end

Sunny.to_device(x; _...) = error("`to_device` is not supported for $(typeof(x))")

# Whether `residues_combiner!` compiles for `backend` in precision F with the
# combiner of `measure`, and reproduces it on the host for sample correlations
function combiner_runs_on(backend, F, measure, corr_pairs)
    (; combiner) = measure
    isbits(combiner) || return false
    Nobs = Sunny.num_observables(measure)
    K = Sunny.num_real_components(eltype(measure))
    A_h = reshape([Complex{F}(μ, n) for μ in 1:Nobs, n in 1:2], Nobs, 2, 1)
    q = Sunny.SVector{3, F}(0.13, 0.41, 0.77)
    try
        disp = KA.allocate(backend, F, (2, 1))
        R = KA.allocate(backend, F, (K, 2, 1))
        E = to_backend(backend, F[1 -1]')
        A = to_backend(backend, A_h)
        residues_combiner!(backend)(disp, R, E, A, to_backend(backend, [q]), combiner, corr_pairs, 0, one(F); ndrange=(2, 1))
        KA.synchronize(backend)
        # The reference in Float64, which `Sunny.real_components` requires
        expected = [s * x for (n, s) in ((1, 1), (2, -1))
                    for x in Sunny.real_components(combiner(Vec3(q), [ComplexF64(A_h[μ, n, 1] * conj(A_h[ν, n, 1])) for (μ, ν) in corr_pairs]))]
        return vec(Array(R)) ≈ expected
    catch err
        @debug "Combiner cannot run on $backend" exception=err
        return false
    end
end

# How the residues evaluate the measure, and its forms if they are independent
# of the wavevector. Forms that error, e.g. for a wavevector outside a scattering
# plane, are taken to depend on it.
function resolve_measure_mode(backend, F, measure, corr_pairs)
    Sunny.num_observables(measure) == 0 && return (:none, nothing)
    combiner_runs_on(backend, F, measure, corr_pairs) && return (:combiner, nothing)
    qs = [Vec3(0.13, 0.41, 0.77), Vec3(-0.6, 0.25, 0.1), Vec3(0.3, -0.8, 0.45)]
    Ms = try
        [Sunny.measure_forms(measure, q) for q in qs]
    catch
        return (:forms, nothing)
    end
    all(==(Ms[1]), Ms) || return (:forms, nothing)
    M = reduce((a, b) -> cat(a, b; dims=3), Ms[1]; init=zeros(ComplexF64, size(Ms[1][1])..., 0))
    return (:constant_forms, to_backend(backend, F, reshape(M, size(M)..., 1)))
end
