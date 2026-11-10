struct SWTDataDipoleDeviceKA{TVecRot, TArrObs, TVecCoef, TVecS}
    local_rotations       :: TVecRot
    observables :: TArrObs
    stevens_coefs         :: TVecCoef
    sqrtS                 :: TVecS
end

function SWTDataDipoleDeviceKA(host::Sunny.SWTDataDipole, backend, ::Type{T}=Float64) where T<:AbstractFloat
    lr_d    = _backend_array(backend, host.local_rotations, T)
    obs_d   = _backend_array(backend, host.observables, T)
    sc_d    = _backend_array(backend, map(x -> StevensExpansionP{T}(x), host.stevens_coefs))
    sqrtS_d = _backend_array(backend, host.sqrtS, T)
    SWTDataDipoleDeviceKA(lr_d, obs_d, sc_d, sqrtS_d)
end

function Adapt.adapt_structure(to, data::SWTDataDipoleDeviceKA)
    local_rotations       = Adapt.adapt_structure(to, data.local_rotations)
    observables = Adapt.adapt_structure(to, data.observables)
    stevens_coefs         = Adapt.adapt_structure(to, data.stevens_coefs)
    sqrtS                 = Adapt.adapt_structure(to, data.sqrtS)
    SWTDataDipoleDeviceKA(local_rotations, observables, stevens_coefs, sqrtS)
end

struct SWTDataSUNDeviceKA{TUnitaries, TObservables, TSpins}
    local_unitaries :: TUnitaries
    observables :: TObservables
    spin_ops :: TSpins
end

function SWTDataSUNDeviceKA(host::Sunny.SWTDataSUN, backend, ::Type{T}=Float64) where T<:AbstractFloat
    Na        = length(host.local_unitaries)
    N         = size(host.local_unitaries[begin], 1)

    unitaries_h = zeros(Complex{T}, N, N, Na)
    for (i, u) in enumerate(host.local_unitaries)
        unitaries_h[:, :, i] .= u
    end

    outer_obs = size(host.observables)
    Nobs      = outer_obs[1]
    observables_h = zeros(Complex{T}, N, N, Nobs, Na)
    for j in 1:Na, ξ in 1:Nobs
        observables_h[:, :, ξ, j] .= host.observables[ξ, j]
    end

    spins_h = zeros(Complex{T}, N, N, 3, Na)
    for j in 1:Na, α in 1:3
        spins_h[:, :, α, j] .= host.spin_ops[α, j]
    end

    SWTDataSUNDeviceKA(
        _backend_array(backend, unitaries_h),
        _backend_array(backend, observables_h),
        _backend_array(backend, spins_h),
    )
end

function Adapt.adapt_structure(to, data::SWTDataSUNDeviceKA)
    local_unitaries       = Adapt.adapt_structure(to, data.local_unitaries)
    observables = Adapt.adapt_structure(to, data.observables)
    spin_ops       = Adapt.adapt_structure(to, data.spin_ops)
    SWTDataSUNDeviceKA(local_unitaries, observables, spin_ops)
end

struct SpinWaveTheoryDeviceKA{TSys, TData, TMeasure} <: Sunny.AbstractSpinWaveTheory
    sys            :: TSys
    data           :: TData
    measure        :: TMeasure
    regularization :: Float64
end

function SpinWaveTheoryDeviceKA(host::Sunny.SpinWaveTheory, backend, ::Type{T}=Float64) where T<:AbstractFloat
    if isa(host.data, Sunny.SWTDataDipole)
        return SpinWaveTheoryDeviceKA(
            SystemDeviceKA(host.sys, backend, T),
            SWTDataDipoleDeviceKA(host.data, backend, T),
            MeasureSpecDevice(host.measure, backend, T),
            host.regularization,
        )
    else

        if T === Float32
            return SpinWaveTheoryDeviceKA(
                SystemDeviceSUNF32KA(host.sys, backend),
                MeasureSpecDevice(host.measure, backend, T),
                SWTDataSUNDeviceKA(host.data::Sunny.SWTDataSUN, backend, T),
                host.regularization,
            )
        else
            return SpinWaveTheoryDeviceKA(
                SystemDeviceSUNKA(host.sys, backend),
                SWTDataSUNDeviceKA(host.data::Sunny.SWTDataSUN, backend, T),
                MeasureSpecDevice(host.measure, backend, T),
                host.regularization,
            )
        end
    end
end

function Adapt.adapt_structure(to, swt::SpinWaveTheoryDeviceKA)
    sys            = Adapt.adapt_structure(to, swt.sys)
    data           = Adapt.adapt_structure(to, swt.data)
    measure        = Adapt.adapt_structure(to, swt.measure)
    regularization = Adapt.adapt_structure(to, swt.regularization)
    SpinWaveTheoryDeviceKA(sys, data, measure, regularization)
end

function Sunny.nflavors(swt::SpinWaveTheoryDeviceKA)
    sys = swt.sys
    return (sys isa SystemDeviceSUNKA || sys isa SystemDeviceSUNF32KA) ? sys.Ns - 1 : 1
end

function Sunny.nbands(swt::SpinWaveTheoryDeviceKA)
    return Sunny.nflavors(swt) * Sunny.natoms(swt.sys.crystal)
end

function Sunny.to_device(swt::Sunny.SpinWaveTheory, backend::KernelAbstractions.Backend,
                         ::Type{T}=Float64) where T<:AbstractFloat
    return SpinWaveTheoryDeviceKA(swt, backend, T)
end

# Fourier prefactor for observable μ acting on a flattened unit i of swt.sys,
# with optional part index.
function observable_prefactor(measure, μ, i, q_reshaped, q_global, sys; part=1)
    unit = fld1(i, prod(num_lattice_dims(measure)))
    r = sys.crystal.positions[i] + measure.offsets[unit, part] ./ Vec3(num_lattice_dims(measure))
    ff = measure.formfactors[μ, unit, part]
    cis(2π * dot(q_reshaped, r)) * compute_form_factor(ff, norm2(q_global))
end

# Linearize each Fourier observable Â(q) = Σᵢ exp(+i q⋅rᵢ) Âᵢ as a coefficient
# vector u in the Nambu basis of Holstein-Primakoff bosons [a_q, a_-q†].
function set_swt_observable_vectors!(u, swt::SpinWaveTheoryDeviceKA, q_reshaped, q_global)
    (; sys, data, measure) = swt
    Na = Sunny.nsites(sys)
    Nobs = Sunny.num_observables(measure)
    Nf = Sunny.nflavors(swt)
    L = Nf * Na

    if sys.mode == :SUN
        (; observables) = data::SWTDataSUNDeviceKA
        @assert allequal(sys.Ns)
        N = first(sys.Ns)
        for μ in 1:Nobs, i in 1:Na
            view(u, (i-1)*Nf .+ (1:Nf), μ) .= 0
            view(u, (i-1)*Nf .+ (1:Nf) .+ L, μ) .= 0
            for p in 1:num_parts_per_unit(measure)
                pref = observable_prefactor(measure, μ, i, q_reshaped, q_global, sys; part=p)
                O = observables[μ, i, p]
                for f in 1:Nf
                    u[f + (i-1)*Nf,     μ] += pref * O[f, N]
                    u[f + (i-1)*Nf + L, μ] += pref * O[N, f]
                end
            end
        end
    else
        #@assert sys.mode in (:dipole, :dipole_uncorrected)
        num_parts_per_unit(measure) == 1 || error("Distinct parts unsupported in dipole-mode.")
        (; sqrtS, observables) = data::SWTDataDipoleDeviceKA
        for μ in 1:Nobs, i in 1:Na
            pref = observable_prefactor(measure, μ, i, q_reshaped, q_global, sys)
            O = observables[μ, i]
            u[i, μ]   = pref * (sqrtS[i] / √2) * (O[1] + im*O[2])
            u[i+L, μ] = pref * (sqrtS[i] / √2) * (O[1] - im*O[2])
        end
    end

    return u
end
