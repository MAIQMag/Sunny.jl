struct MeasureSpecDevice{TObsArr, TCorrArr, F, TFormArr, TOffsets}
    observables :: TObsArr    # (nobs × sys_dims × natoms)
    corr_pairs :: TCorrArr    # (ncorr)
    combiner   :: F           # (q::Vec3, obs) -> Ret
    formfactors :: TFormArr   # (nobs × natoms)
    offsets     :: TOffsets   # (nunits × nparts)
end

function MeasureSpecDevice(host::Sunny.MeasureSpec, backend, ::Type{T}=Float64) where T<:AbstractFloat
    corr_pairs_d = _backend_array(backend, host.corr_pairs, T)
    formfactors_d = _backend_array(backend, host.formfactors, T)
    if eltype(host.observables) == Sunny.Vec3
        observables_d = _backend_array(backend, host.observables, T)
    else
        outer_size = size(host.observables)
        inner_size = size(host.observables[begin])
        observables_h = Array{ComplexF64}(undef, inner_size..., outer_size...)
        for (ind, val) in pairs(host.observables)
            view(observables_h, :, :, ind) .= val
        end
        observables_d = _backend_array(backend, observables_h, T)
    end
    offsets_d = _backend_array(backend, host.offsets, T)
    return MeasureSpecDevice(observables_d, corr_pairs_d, host.combiner, formfactors_d, offsets_d) 
end

function Adapt.adapt_structure(to, data::MeasureSpecDevice)
    observables = Adapt.adapt_structure(to, data.observables)
    corr_pairs = Adapt.adapt_structure(to, data.corr_pairs)
    combiner = Adapt.adapt_structure(to, data.combiner)
    formfactors = Adapt.adapt_structure(to, data.formfactors)
    offsets = Adapt.adapt_structure(to, data.offsets)
    MeasureSpecDevice(observables, corr_pairs, combiner, formfactors, offsets)
end

function Sunny.num_observables(measure::MeasureSpecDevice)
    if eltype(measure.observables) == Sunny.Vec3
        return size(measure.observables, 1)
    else
        return size(measure.observables, 3)
    end
end

Sunny.num_correlations(measure::MeasureSpecDevice) = length(measure.corr_pairs) 
num_lattice_dims(measure::MeasureSpecDevice) = size(measure.observables)[2:4]
num_units_per_cell(measure::MeasureSpecDevice) = size(measure.observables, 5)
num_parts_per_unit(measure::MeasureSpecDevice) = size(measure.observables, 6)

Base.eltype(device::MeasureSpecDevice)  = only(Base.return_types(device.combiner, (Sunny.Vec3, AbstractVector{ComplexF64})))