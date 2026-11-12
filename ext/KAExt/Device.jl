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
    swt     :: Sunny.SpinWaveTheory
    backend :: B
    data    :: D
end

function Base.show(io::IO, ::MIME"text/plain", swt_d::SpinWaveTheoryDevice)
    printstyled(io, "SpinWaveTheoryDevice ", Sunny.mode_to_str(swt_d.swt.sys), "\n"; bold=true, color=:underline)
    println(io, "  ", Sunny.nsites(swt_d.swt.sys), " sites on ", swt_d.backend)
end

"""
    to_device(swt::SpinWaveTheory; backend=KernelAbstractions.CPU())

Copies the data of `swt` that [`intensities`](@ref) needs to the
KernelAbstractions `backend`, e.g., `CUDABackend()`. The calculation of
`intensities(to_device(swt), qpts; ...)` then runs on that backend.
"""
function Sunny.to_device(swt::Sunny.SpinWaveTheory; backend=KA.CPU())
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

    data = (;
        rowptr       = to_backend(backend, rowptr),
        cols         = to_backend(backend, cols),
        offset_index = to_backend(backend, offset_index),
        vals         = to_backend(backend, vals),
        offsets      = to_backend(backend, offsets),
        coefs        = to_backend(backend, observable_coefs),
        positions    = to_backend(backend, positions),
        formfactors  = to_backend(backend, formfactors),
    )
    return SpinWaveTheoryDevice(swt, backend, data)
end

Sunny.to_device(x; _...) = error("`to_device` is not supported for $(typeof(x))")
