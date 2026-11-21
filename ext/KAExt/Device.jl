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

# The fields of the struct x, each adapted by `to`. A device struct S{F, ...}
# defines a constructor S{F}(fields...) that infers the other type parameters,
# so that `Adapt.adapt_structure(to, x::S{F})` is S{F}(adapt_fields(to, x)...),
# e.g. with the device arrays of a kernel argument in place of its arrays.
adapt_fields(to, x) = ntuple(i -> adapt(to, getfield(x, i)), Val(fieldcount(typeof(x))))

# Mirrors `Sunny.QuadraticTable` on a device, in precision F
struct QuadraticTableDevice{F, VO, VI, VV}
    offsets      :: VO
    rowptr       :: VI
    cols         :: VI
    offset_index :: VI
    vals         :: VV
end

QuadraticTableDevice{F}(offsets::VO, rowptr::VI, cols::VI, offset_index::VI, vals::VV) where {F, VO, VI, VV} =
    QuadraticTableDevice{F, VO, VI, VV}(offsets, rowptr, cols, offset_index, vals)

Adapt.adapt_structure(to, t::QuadraticTableDevice{F}) where F = QuadraticTableDevice{F}(adapt_fields(to, t)...)

function QuadraticTableDevice{F}(backend, t::Sunny.QuadraticTable) where F
    (; offsets, rowptr, cols, offset_index, vals) = t
    return QuadraticTableDevice{F}(to_backend(backend, F, offsets), to_backend(backend, rowptr), to_backend(backend, cols),
                                   to_backend(backend, offset_index), to_backend(backend, F, vals))
end

# The data of `Sunny.MeasureSpec` that the residues need on a device, in
# precision F: the pairs of observables to correlate, and the Fourier positions
# and form factors of each observable, as in `Sunny.observable_prefactor`
struct MeasureSpecDevice{F, C, VP, VF}
    corr_pairs  :: C
    positions   :: VP # [site, part]
    formfactors :: VF # [observable, site, part]
end

MeasureSpecDevice{F}(corr_pairs::C, positions::VP, formfactors::VF) where {F, C, VP, VF} =
    MeasureSpecDevice{F, C, VP, VF}(corr_pairs, positions, formfactors)

Adapt.adapt_structure(to, m::MeasureSpecDevice{F}) where F = MeasureSpecDevice{F}(adapt_fields(to, m)...)

function MeasureSpecDevice{F}(backend, measure::Sunny.MeasureSpec, sys::Sunny.System) where F
    Na = Sunny.nsites(sys)
    Nparts = Sunny.num_parts_per_unit(measure)
    sites = Sunny.eachsite(sys)
    positions = [Sunny.global_position_at(sys, sites[i]) + sys.crystal.latvecs * measure.offsets[Sunny.to_atom(sites[i]), p]
                 for i in 1:Na, p in 1:Nparts]
    formfactors = [FormFactorDevice{F}(measure.formfactors[μ, Sunny.to_atom(sites[i]), p])
                   for μ in 1:Sunny.num_observables(measure), i in 1:Na, p in 1:Nparts]
    corr_pairs = Sunny.SVector{length(measure.corr_pairs)}(measure.corr_pairs)
    return MeasureSpecDevice{F}(corr_pairs, to_backend(backend, F, positions), to_backend(backend, formfactors))
end

# How the residues evaluate the measure, named by Symbols for display: by its
# combiner on device, by its forms (see `Sunny.measure_forms`) if they are
# independent of the wavevector, by its forms on the host per wavevector, or not
# at all, without observables. See `resolve_measure_mode`.
abstract type MeasureMode end
struct CombinerMode <: MeasureMode end
struct ConstantFormsMode{M} <: MeasureMode
    forms :: M  # M[μ, ν, k, 1] on device
end
struct FormsMode <: MeasureMode end
struct NoMeasureMode <: MeasureMode end

Base.Symbol(::CombinerMode) = :combiner
Base.Symbol(::ConstantFormsMode) = :constant_forms
Base.Symbol(::FormsMode) = :forms
Base.Symbol(::NoMeasureMode) = :none

# The batched eigensolvers of `bogoliubov_batched!`, named by the Symbols of the
# option `eigensolver` of `to_device`
abstract type DeviceEigensolver end
struct HostEigensolver <: DeviceEigensolver end    # LAPACK on the host
struct JacobiEigensolver <: DeviceEigensolver end  # The kernel `bogoliubov_jacobi!`
struct VendorEigensolver <: DeviceEigensolver end  # `Sunny.vendor_bogoliubov!` of an extension

function DeviceEigensolver(name::Symbol)
    name == :host && return HostEigensolver()
    name == :jacobi && return JacobiEigensolver()
    name == :vendor && return VendorEigensolver()
    error("Unknown eigensolver $(repr(name)); use :host, :jacobi or :vendor")
end

Base.Symbol(::HostEigensolver) = :host
Base.Symbol(::JacobiEigensolver) = :jacobi
Base.Symbol(::VendorEigensolver) = :vendor

# Mirrors `Sunny.SpinWaveTheory` on `backend`, with floating point numbers in
# precision F
struct SpinWaveTheoryDevice{F, B, Q, E, C, M}
    swt              :: Sunny.SpinWaveTheory
    backend          :: B
    eigensolver      :: DeviceEigensolver
    measure_mode     :: MeasureMode
    regularization   :: F
    quadratic_table  :: Q
    ewald            :: E  # Or `nothing`
    observable_coefs :: C
    measure          :: M

    SpinWaveTheoryDevice{F}(swt, backend::B, eigensolver, measure_mode, regularization, quadratic_table::Q, ewald::E,
                            observable_coefs::C, measure::M) where {F, B, Q, E, C, M} =
        new{F, B, Q, E, C, M}(swt, backend, eigensolver, measure_mode, regularization, quadratic_table, ewald,
                              observable_coefs, measure)
end

precision_of(::SpinWaveTheoryDevice{F}) where F = F

# Whether an extension provides `Sunny.vendor_bogoliubov!` for `backend` in
# precision F, for n×n matrices
Sunny.vendor_bogoliubov_max_size(_) = typemax(Int)
Sunny.vendor_bogoliubov_min_default_size(_) = 1
function has_vendor_bogoliubov(backend, F, n)
    H = KA.allocate(backend, Complex{F}, (2, 2, 1))
    E = KA.allocate(backend, F, (2, 1))
    status = KA.allocate(backend, Int32, 1)
    return applicable(Sunny.vendor_bogoliubov!, H, E, H, 1, status) && n <= Sunny.vendor_bogoliubov_max_size(backend)
end

# The Jacobi kernel uses one thread per wavevector, with strided memory access,
# and its cost grows as n³ for n×n matrices. On an NVIDIA P5000, it outpaced
# LAPACK on the host by 4× for n = 8, but lagged 10× for n = 64. It lagged
# cuSOLVER for all n, e.g. 2.9 s vs 2.2 s for 200k matrices with n = 8, and 8.9
# s vs 1.6 s for 100k matrices with n = 16. On an AMD MI100, it outpaced LAPACK
# on the host (MKL, 16 threads) by 3.3× for n = 8, and matched it for n = 16
# and 40. It lagged rocSOLVER for all n, e.g. 1.7 vs 1.5 µs per matrix for
# n = 8, and 337 vs 56 µs for n = 40, with 100k matrices in Float64.
function resolve_eigensolver(backend, eigensolver, n, F)
    return DeviceEigensolver(resolve_eigensolver_name(backend, eigensolver, n, F))
end

function resolve_eigensolver_name(backend, eigensolver, n, F)
    if isnothing(eigensolver)
        backend isa KA.CPU && return :host
        if has_vendor_bogoliubov(backend, F, n) && n >= Sunny.vendor_bogoliubov_min_default_size(backend)
            return :vendor
        end
        return n <= 8 ? :jacobi : :host
    end
    eigensolver in (:host, :jacobi, :vendor) || error("Unknown eigensolver $(repr(eigensolver)); use :host, :jacobi or :vendor")
    if eigensolver == :vendor && !has_vendor_bogoliubov(backend, F, n)
        max_n = Sunny.vendor_bogoliubov_max_size(backend)
        if n > max_n
            error("Vendor eigensolver for $backend supports at most $(max_n ÷ 2) bands, got $(n ÷ 2); use :host or :jacobi")
        end
        error("No vendor eigensolver for $backend in $F; load its package (e.g. CUDA or AMDGPU) or use :jacobi")
    end
    return eigensolver
end

function Base.show(io::IO, ::MIME"text/plain", swt_d::SpinWaveTheoryDevice)
    printstyled(io, "SpinWaveTheoryDevice ", Sunny.mode_to_str(swt_d.swt.sys), "\n"; bold=true, color=:underline)
    println(io, "  ", Sunny.nsites(swt_d.swt.sys), " sites on ", swt_d.backend, " in ", precision_of(swt_d),
            ", eigensolver ", repr(Symbol(swt_d.eigensolver)), ", measure ", repr(Symbol(swt_d.measure_mode)))
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
case. In `Float32`, the `regularization` of `swt` is raised to at least the
rounding error of its Hamiltonian, so that modes of vanishing energy, e.g.
Goldstone modes, remain stable.

The batched `eigensolver` for the Bogoliubov transformation may be `:host`
(LAPACK on the host), `:vendor` (cuSOLVER or rocSOLVER, the default when CUDA
or AMDGPU is loaded, or a Householder–QL kernel on Metal for at most 128 bands),
or `:jacobi` (a portable kernel, efficient for small matrices).
The default is `:host` for `KernelAbstractions.CPU()`. Otherwise it is
`:vendor` if available, except `:jacobi` on Metal for at most 2 bands, else
`:jacobi` for at most 4 bands and `:host` beyond.

Dipole-dipole interactions are supported, with their Ewald sums on device.

The measure's combiner is evaluated on device if it is `isbits` and compiles
for the backend. Otherwise, its forms (see `Sunny.measure_forms`) are evaluated
on the host, once if they are independent of the wavevector, and otherwise per
wavevector.
"""
function Sunny.to_device(swt::Sunny.SpinWaveTheory; backend=KA.CPU(), eigensolver=nothing,
                         precision::Type=Sunny.device_precision(backend))
    F = precision
    F in (Float32, Float64) || error("Unsupported precision $F; use Float64 or Float32")

    (; sys, measure, quadratic_table, observable_coefs, ewald) = swt
    L = Sunny.nbands(swt)
    corr_pairs = Sunny.SVector{length(measure.corr_pairs)}(measure.corr_pairs)
    measure_mode = resolve_measure_mode(backend, F, measure, corr_pairs)
    eigensolver = resolve_eigensolver(backend, eigensolver, 2L, F)
    return SpinWaveTheoryDevice{F}(swt, backend, eigensolver, measure_mode, device_regularization(swt, F),
                                   QuadraticTableDevice{F}(backend, quadratic_table),
                                   isnothing(ewald) ? nothing : QuadraticEwaldDevice{F}(backend, ewald, L),
                                   to_backend(backend, F, observable_coefs),
                                   MeasureSpecDevice{F}(backend, measure, sys))
end

# The constants of `Sunny.ewald_interaction_tensor` in precision F
struct EwaldTensorParams{F}
    dims        :: NTuple{3, Int32}
    recipvecs   :: Sunny.SMatrix{3, 3, F, 9}  # Of the supercell
    mmax        :: NTuple{3, Int32}
    kmax²       :: F
    σ²          :: F
    invV        :: F
    μ0_μB²      :: F
    demag_V     :: Sunny.SMatrix{3, 3, F, 9}
    self_energy :: Sunny.SMatrix{3, 3, F, 9}
end

# Mirrors `Sunny.EwaldTensorCache` on a device, in precision F. The real-space
# terms of each linear index of [cell, i, j] are in CSR form, `real_ptr`, as
# are the excluded couplings, which are few: the linear index, winding and
# tensor of each.
struct EwaldTensorCacheDevice{F, VI, V3, VM}
    dims        :: NTuple{3, Int32}
    na          :: Int
    params      :: EwaldTensorParams{F}
    positions   :: V3
    real_ptr    :: VI
    real_shifts :: V3
    real_mats   :: VM
    ex_index    :: VI
    ex_winding  :: V3
    ex_mats     :: VM
end

EwaldTensorCacheDevice{F}(dims, na, params, positions::V3, real_ptr::VI, real_shifts::V3, real_mats::VM, ex_index::VI,
                          ex_winding::V3, ex_mats::VM) where {F, VI, V3, VM} =
    EwaldTensorCacheDevice{F, VI, V3, VM}(dims, na, params, positions, real_ptr, real_shifts, real_mats, ex_index,
                                          ex_winding, ex_mats)

Adapt.adapt_structure(to, c::EwaldTensorCacheDevice{F}) where F = EwaldTensorCacheDevice{F}(adapt_fields(to, c)...)

function EwaldTensorCacheDevice{F}(backend, cache::Sunny.EwaldTensorCache) where F
    (; dims, cryst, μ0_μB², demag, σ², mmax, kmax², real_terms, excluded) = cache
    na = Sunny.natoms(cryst)
    V = abs(det(cryst.latvecs)) * prod(dims)
    params = EwaldTensorParams{F}(Int32.(dims), cryst.recipvecs / diagm(Vec3(dims)), Int32.(mmax), kmax², σ², 1/V,
                                  μ0_μB², demag / V, -Sunny.Mat3(I) / (3(2π)^(3/2) * σ²^(3/2)))
    terms = vec(real_terms)
    LI = LinearIndices((prod(dims), na, na))
    LC = LinearIndices(dims)
    return EwaldTensorCacheDevice{F}(
        Int32.(dims), na, params,
        to_backend(backend, F, cryst.positions),
        to_backend(backend, Int32[1; 1 .+ cumsum(length.(terms))]),
        to_backend(backend, F, [Vec3(n) for t in terms for (n, _) in t]),
        to_backend(backend, F, [A for t in terms for (_, A) in t]),
        to_backend(backend, Int32[LI[LC[(mod.(bond.n, dims) .+ 1)...], bond.i, bond.j] for (; bond) in excluded]),
        to_backend(backend, F, [Vec3(fld.(bond.n, dims)) for (; bond) in excluded]),
        to_backend(backend, F, [A_loc for (; A_loc) in excluded]),
    )
end

ncells(cache::EwaldTensorCacheDevice) = Int(prod(cache.dims))

# Mirrors `Sunny.QuadraticEwald` on a device, in precision F, for L bands.
#
# The entries of T are given twice: per bare site j, as in `ptr`, and per Nambu
# label r, as in `label_ptr`, so that the thread of row r of H gathers its
# entries without races. The entries of each label are ordered by site, so
# that each entry of H sums in the order of `Sunny.accum_ewald!`.
struct QuadraticEwaldDevice{F, C <: EwaldTensorCacheDevice{F}, VI, VC, VN}
    cache      :: C
    ptr        :: VI
    labels     :: VI
    coefs      :: VC
    atoms      :: VI
    cells      :: VN
    label_ptr  :: VI
    label_site :: VI
    label_coef :: VC
end

QuadraticEwaldDevice{F}(cache::C, ptr::VI, labels::VI, coefs::VC, atoms::VI, cells::VN, label_ptr::VI, label_site::VI,
                        label_coef::VC) where {F, C, VI, VC, VN} =
    QuadraticEwaldDevice{F, C, VI, VC, VN}(cache, ptr, labels, coefs, atoms, cells, label_ptr, label_site, label_coef)

Adapt.adapt_structure(to, E::QuadraticEwaldDevice{F}) where F = QuadraticEwaldDevice{F}(adapt_fields(to, E)...)

function QuadraticEwaldDevice{F}(backend, E::Sunny.QuadraticEwald, L) where F
    (; cache, ptr, labels, coefs, atoms, cells) = E
    by_label = [Int[] for _ in 1:2L]
    for i in eachindex(cells), e in ptr[i]:ptr[i+1]-1
        push!(by_label[labels[e]], e)
    end
    es = reduce(vcat, by_label)
    site_of = [i for i in eachindex(cells) for _ in ptr[i]:ptr[i+1]-1]

    return QuadraticEwaldDevice{F}(
        EwaldTensorCacheDevice{F}(backend, cache),
        to_backend(backend, ptr),
        to_backend(backend, labels),
        to_backend(backend, F, coefs),
        to_backend(backend, Int32.(atoms)),
        to_backend(backend, [Int32.(c) for c in cells]),
        to_backend(backend, Int32[1; 1 .+ cumsum(length.(by_label))]),
        to_backend(backend, Int32.(site_of[es])),
        to_backend(backend, F, coefs[es]),
    )
end

# The wavevector `q_supercell` as `Sunny.ewald_interaction_tensor` takes it:
# rounded to an integer within round-off, then its offset q0 from the nearest
# integer, and whether q0 vanishes, so that the k = 0 mode is singular
function ewald_wavevector(q_supercell)
    q = map(x -> Sunny.is_integer(x; tol=1e-12) ? round(x) : x, Vec3(q_supercell))
    q0 = q - round.(q)
    return (q, q0, norm(q0) < 1e-16)
end

# The shift of the diagonal of the Hamiltonian in precision F. Rounding H to F
# perturbs it by about eps(F) times its spectral scale. In Float32, this
# exceeds the default `swt.regularization` of 1e-8, so H may be indefinite at a
# Goldstone mode, and the shift is raised to that perturbation. For the Néel
# order of CoRh₂O₄ on Metal, a third of it sufficed. The shift changes gapped
# energies in proportion, and intensities near a Goldstone mode with its square
# root, e.g. by 4e-5 relative to all intensities along a path through q = 0.
function device_regularization(swt, F)
    return F(max(swt.regularization, eps(F) * Sunny.spectral_scale(swt)))
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

# The `MeasureMode` of the residues. Forms that error, e.g. for a wavevector
# outside a scattering plane, are taken to depend on it.
function resolve_measure_mode(backend, F, measure, corr_pairs)
    Sunny.num_observables(measure) == 0 && return NoMeasureMode()
    combiner_runs_on(backend, F, measure, corr_pairs) && return CombinerMode()
    qs = [Vec3(0.13, 0.41, 0.77), Vec3(-0.6, 0.25, 0.1), Vec3(0.3, -0.8, 0.45)]
    Ms = try
        [Sunny.measure_forms(measure, q) for q in qs]
    catch
        return FormsMode()
    end
    all(==(Ms[1]), Ms) || return FormsMode()
    M = reduce((a, b) -> cat(a, b; dims=3), Ms[1]; init=zeros(ComplexF64, size(Ms[1][1])..., 0))
    return ConstantFormsMode(to_backend(backend, F, reshape(M, size(M)..., 1)))
end
