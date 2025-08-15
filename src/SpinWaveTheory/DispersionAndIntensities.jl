# Bogoliubov transformation that diagonalizes a quadratic bosonic Hamiltonian,
# allowing for anomalous terms. The general procedure derives from Colpa,
# Physica A, 93A, 327-353 (1978). Overwrites data in H.
function bogoliubov!(T::Matrix{ComplexF64}, H::Matrix{ComplexF64})
    L = div(size(H, 1), 2)
    @assert size(T) == size(H) == (2L, 2L)
    # H0 = copy(H)

    # Initialize T to the para-unitary identity Ĩ = diagm([ones(L), -ones(L)])
    T .= 0
    for i in 1:L
        T[i, i] = 1
        T[i+L, i+L] = -1
    end

    # Solve generalized eigenvalue problem, Ĩ t = λ H t, for columns t of T.
    # Eigenvalues are sorted such that positive values appear first, and are
    # otherwise ascending in absolute value.
    sortby(x) = (-sign(x), abs(x))
    λ, T0 = eigen!(Hermitian(T), Hermitian(H); sortby)

    # Note that T0 and T refer to the same data.
    @assert T0 === T

    # Normalize columns of T so that para-unitarity holds, T† Ĩ T = Ĩ.
    for j in axes(T, 2)
        c = 1 / sqrt(abs(λ[j]))
        view(T, :, j) .*= c
    end

    # Inverse of λ are eigenvalues of Ĩ H, or equivalently, of √H Ĩ √H.
    energies = λ        # reuse storage
    @. energies = 1 / λ

    # By Sylvester's theorem, "inertia" (sign signature) is invariant under a
    # congruence transform Ĩ → √H Ĩ √H. The first L elements are positive,
    # while the next L elements are negative. Their absolute values are
    # excitation energies for the wavevectors q and -q, respectively.
    @assert all(>(0), view(energies, 1:L)) && all(<(0), view(energies, L+1:2L))

    # Disable tests below for speed. Note that the data in H has been
    # overwritten by eigen!, so H0 should refer to an original copy of H.
    #=
    Ĩ = Diagonal([ones(L); -ones(L)])
    @assert T' * Ĩ * T ≈ Ĩ
    @assert diag(T' * H0 * T) ≈ Ĩ * energies
    # Reflection symmetry H(q) = H(-q) is identified as H11 = conj(H22). In this
    # case, eigenvalues come in pairs.
    if H0[1:L, 1:L] ≈ conj(H0[L+1:2L, L+1:2L])
        @assert energies[1:L] ≈ -energies[L+1:2L]
    end
    =#

    return energies
end


# Returns |1 + nB(ω)| where nB(ω) = 1 / (exp(βω) - 1) is the Bose function.
# Equivalent to |1 / expm1(-βω)| where expm1(x) = e^x-1.
function thermal_prefactor(ω; kT)
    @assert kT >= 0
    iszero(ω) && return Inf
    return abs(1 / expm1(-ω/kT))
end


"""
    excitations!(T, tmp, swt::SpinWaveTheory, q)

Given a wavevector `q`, solves for the matrix `T` representing quasi-particle
excitations, and returns a list of quasi-particle energies. Both `T` and `tmp`
must be supplied as ``2L×2L`` complex matrices, where ``L`` is the number of
bands for a single ``𝐪`` value.

The columns of `T` are understood to be contracted with the Holstein-Primakoff
bosons ``[𝐛_𝐪, 𝐛_{-𝐪}^†]``. The first ``L`` columns provide the eigenvectors
of the quadratic Hamiltonian for the wavevector ``𝐪``. The next ``L`` columns
of `T` describe eigenvectors for ``-𝐪``. The return value is a vector with
similar grouping: the first ``L`` values are energies for ``𝐪``, and the next
``L`` values are the _negation_ of energies for ``-𝐪``.

    excitations!(T, tmp, swt::SpinWaveTheorySpiral, q; branch)

Calculations on a [`SpinWaveTheorySpiral`](@ref) additionally require a `branch`
index. The possible branches ``(1, 2, 3)`` correspond to scattering processes
``𝐪 - 𝐤, 𝐪, 𝐪 + 𝐤`` respectively, where ``𝐤`` is the ordering wavevector.
Each branch will contribute ``L`` excitations, where ``L`` is the number of
spins in the magnetic cell. This yields a total of ``3L`` excitations for a
given momentum transfer ``𝐪``.
"""
function excitations!(T, tmp, swt::SpinWaveTheory, q)
    L = nbands(swt)
    size(T) == size(tmp) == (2L, 2L) || error("Arguments T and tmp must be $(2L)×$(2L) matrices")

    q_reshaped = to_reshaped_rlu(swt.sys, q)
    dynamical_matrix!(tmp, swt, q_reshaped)

    try
        return bogoliubov!(T, tmp)
    catch _
        rethrow(ErrorException("Not an energy-minimum; wavevector q = $q unstable."))
    end
end

"""
    excitations(swt::SpinWaveTheory, q)
    excitations(swt::SpinWaveTheorySpiral, q; branch)

Returns a pair `(energies, T)` providing the excitation energies and
eigenvectors. Prefer [`excitations!`](@ref) for performance, which avoids matrix
allocations. See the documentation of [`excitations!`](@ref) for more details.
"""
function excitations(swt::SpinWaveTheory, q)
    L = nbands(swt)
    T = zeros(ComplexF64, 2L, 2L)
    H = zeros(ComplexF64, 2L, 2L)
    energies = excitations!(T, copy(H), swt, q)
    return (energies, T)
end

"""
    dispersion(swt::SpinWaveTheory, qpts)

Given a list of wavevectors `qpts` in reciprocal lattice units (RLU), returns
excitation energies for each band. The return value `ret` is 2D array, and
should be indexed as `ret[band_index, q_index]`.
"""
function dispersion(swt::SpinWaveTheory, qpts)
    L = nbands(swt)
    qpts = convert(AbstractQPoints, qpts)
    disp = zeros(L, length(qpts.qs))
    for (iq, q) in enumerate(qpts.qs)
        view(disp, :, iq) .= view(excitations(swt, q)[1], 1:L)
    end
    return reshape(disp, L, size(qpts.qs)...)
end

"""
    intensities_bands(swt::SpinWaveTheory, qpts; kT=0)

Calculate spin wave excitation bands for a set of q-points in reciprocal space.
This calculation is analogous to [`intensities`](@ref), but does not perform
line broadening of the bands.
"""
function intensities_bands(swt::SpinWaveTheory, qpts; kT=0, with_negative=false)
    (; sys, measure) = swt
    isempty(measure.observables) && error("No observables! Construct SpinWaveTheory with a `measure` argument.")
    with_negative && error("Option `with_negative=true` not yet supported.")

    qpts = convert(AbstractQPoints, qpts)
    cryst = orig_crystal(sys)

    # Number of (magnetic) atoms in magnetic cell
    @assert sys.dims == (1,1,1)
    Na = nsites(sys)
    # Number of chemical cells in magnetic cell
    Ncells = Na / natoms(cryst)
    # Number of quasiparticle modes
    L = nbands(swt)
    # Number of wavevectors
    Nq = length(qpts.qs)

    # Temporary storage for pair correlations
    Nobs = num_observables(measure)
    Ncorr = num_correlations(measure)
    corrbuf = zeros(ComplexF64, Ncorr)

    # Preallocation
    T = zeros(ComplexF64, 2L, 2L, Nq)
    H = zeros(ComplexF64, 2L, 2L, Nq)
    Avec_pref = zeros(ComplexF64, Nobs, Na)
    disp = zeros(Float64, L, Nq)
    intensity = zeros(eltype(measure), L, Nq)

    for (iq, q) in enumerate(qpts.qs)
        q_reshaped = to_reshaped_rlu(swt.sys, q)
        dynamical_matrix!(view(H, :, :, iq), swt, q_reshaped)
        for i in 1:L
            T[i, i, iq] = 1
            T[i+L, i+L, iq] = -1
        end
    end

    for (iq, q) in enumerate(qpts.qs)
        Hq = view(H,:,:,iq)
        Tq = view(T,:,:,iq)

        # Solve generalized eigenvalue problem, Ĩ t = λ H t, for columns t of T.
        # Eigenvalues are sorted such that positive values appear first, and are
        # otherwise ascending in absolute value.
        sortby(x) = (-sign(x), abs(x))
        λ, T0 = eigen!(Hermitian(Tq), Hermitian(Hq); sortby)

        # Note that T0 and T refer to the same data.
        @assert T0 === Tq

        # Normalize columns of T so that para-unitarity holds, T† Ĩ T = Ĩ.
        for j in axes(Tq, 2)
            c = 1 / sqrt(abs(λ[j]))
            view(Tq, :, j) .*= c
        end

        # Inverse of λ are eigenvalues of Ĩ H, or equivalently, of √H Ĩ √H.
        energies = λ        # reuse storage
        @. energies = 1 / λ

        # By Sylvester's theorem, "inertia" (sign signature) is invariant under a
        # congruence transform Ĩ → √H Ĩ √H. The first L elements are positive,
        # while the next L elements are negative. Their absolute values are
        # excitation energies for the wavevectors q and -q, respectively.
        @assert all(>(0), view(energies, 1:L)) && all(<(0), view(energies, L+1:2L))

        view(disp, :, iq) .= view(energies, 1:L)
    end

    for (iq, q) in enumerate(qpts.qs)
        q_global = cryst.recipvecs * q
        Tq = view(T,:,:,iq)
        for i in 1:Na, μ in 1:Nobs
            r_global = global_position(sys, (1, 1, 1, i)) # + offsets[μ, i]
            ff = get_swt_formfactor(measure, μ, i)
            Avec_pref[μ, i] = exp(- im * dot(q_global, r_global))
            Avec_pref[μ, i] *= compute_form_factor(ff, norm2(q_global))
        end

        Avec = zeros(ComplexF64, Nobs)

        # Fill `intensity` array
        for band in 1:L
            fill!(Avec, 0)
            if sys.mode == :SUN
                data = swt.data::SWTDataSUN
                N = sys.Ns[1]
                t = reshape(view(Tq, :, band), N-1, Na, 2)
                for i in 1:Na, μ in 1:Nobs
                    O = data.observables_localized[μ, i]
                    for α in 1:N-1
                        Avec[μ] += Avec_pref[μ, i] * (O[α, N] * t[α, i, 2] + O[N, α] * t[α, i, 1])
                    end
                end
            else
                @assert sys.mode in (:dipole, :dipole_uncorrected)
                data = swt.data::SWTDataDipole
                t = reshape(view(Tq, :, band), Na, 2)
                for i in 1:Na, μ in 1:Nobs
                    O = data.observables_localized[μ, i]
                    # This is the Avec of the two transverse and one
                    # longitudinal directions in the local frame. (In the
                    # local frame, z is longitudinal, and we are computing
                    # the transverse part only, so the last entry is zero)
                    displacement_local_frame = SA[t[i, 2] + t[i, 1], im * (t[i, 2] - t[i, 1]), 0.0]
                    Avec[μ] += Avec_pref[μ, i] * (data.sqrtS[i]/√2) * (O' * displacement_local_frame)[1]
                end
            end

            map!(corrbuf, measure.corr_pairs) do (μ, ν)
                Avec[μ] * conj(Avec[ν]) / Ncells
            end
            intensity[band, iq] = thermal_prefactor(disp[band, iq]; kT) * measure.combiner(q_global, corrbuf)
        end
    end

    disp = reshape(disp, L, size(qpts.qs)...)
    intensity = reshape(intensity, L, size(qpts.qs)...)
    return BandIntensities(cryst, qpts, disp, intensity)
end

"""
    intensities!(data, swt::SpinWaveTheory, qpts; energies, kernel, kT=0)
    intensities!(data, sc::SampledCorrelations, qpts; energies, kernel=nothing, kT=0)

Like [`intensities`](@ref), but makes use of storage space `data` to avoid
allocation costs.
"""
function intensities!(data, swt::AbstractSpinWaveTheory, qpts; energies, kernel::AbstractBroadening, kT=0)
    @assert size(data) == (length(energies), size(qpts.qs)...)
    bands = intensities_bands(swt, qpts; kT)
    @assert eltype(bands) == eltype(data)
    broaden!(data, bands; energies, kernel)
    return Intensities(bands.crystal, bands.qpts, collect(Float64, energies), data)
end

"""
    intensities(swt::SpinWaveTheory, qpts; energies, kernel, kT=0)
    intensities(sc::SampledCorrelations, qpts; energies, kernel=nothing, kT)

Calculates dynamical pair correlation intensities for a set of ``𝐪``-points in
reciprocal space.

Linear spin wave theory calculations are performed with an instance of
[`SpinWaveTheory`](@ref). The alternative [`SpinWaveTheorySpiral`](@ref) allows
to study generalized spiral orders with a single, incommensurate-``𝐤`` ordering
wavevector. Another alternative [`SpinWaveTheoryKPM`](@ref) is favorable for
calculations on large magnetic cells, and allows to study systems with disorder.
An optional nonzero temperature `kT` will scale intensities by the quantum
thermal occupation factor ``|1 + n_B(ω)|`` where ``n_B(ω) = 1/(e^{βω}-1)`` is
the Bose function.

Intensities can also be calculated for `SampledCorrelations` associated with
classical spin dynamics. In this case, thermal broadening will already be
present, and the line-broadening `kernel` becomes optional. Conversely, the
parameter `kT` becomes required. If positive, it will introduce an intensity
correction factor ``|βω (1 + n_B(ω))|`` that undoes the occupation factor for
the classical Boltzmann distribution and applies the quantum thermal occupation
factor. The special choice `kT = nothing` will suppress the classical-to-quantum
correction factor, and yield statistics consistent with the classical Boltzmann
distribution.
"""
function intensities(swt::AbstractSpinWaveTheory, qpts; energies, kernel::AbstractBroadening, kT=0)
    return broaden(intensities_bands(swt, qpts; kT); energies, kernel)
end

"""
    intensities_static(swt::SpinWaveTheory, qpts; bounds=(-Inf, Inf), kernel=nothing, kT=0)
    intensities_static(sc::SampledCorrelations, qpts; bounds=(-Inf, Inf), kT)
    intensities_static(sc::SampledCorrelationsStatic, qpts)

Like [`intensities`](@ref), but integrates the dynamical correlations
``\\mathcal{S}(𝐪, ω)`` over a range of energies ``ω``. By default, the
integration `bounds` are ``(-∞, ∞)``, yielding the instantaneous (equal-time)
correlations.

In [`SpinWaveTheory`](@ref), the integral will be realized as a sum over
discrete bands. Alternative calculation methods are
[`SpinWaveTheorySpiral`](@ref) and [`SpinWaveTheoryKPM`](@ref).

Classical dynamics data in [`SampledCorrelations`](@ref) can also be used to
calculate static intensities. In this case, the domain of integration will be a
finite grid of available `energies`. Here, the parameter `kT` will be used to
account for the quantum thermal occupation of excitations, as documented in
[`intensities`](@ref).

Static intensities calculated from [`SampledCorrelationsStatic`](@ref) are
dynamics-independent. Instead, instantaneous correlations sampled from the
classical Boltzmann distribution will be reported.
"""
function intensities_static(swt::AbstractSpinWaveTheory, qpts; bounds=(-Inf, Inf), kernel=nothing, kT=0)
    res = intensities_bands(swt, qpts; kT)  # TODO: with_negative=true
    data_reduced = zeros(eltype(res.data), size(res.data)[2:end])
    for ib in axes(res.data, 1), iq in CartesianIndices(data_reduced)
        ϵ = res.disp[ib, iq]
        if isnothing(kernel) || bounds == (-Inf, Inf)
            if bounds[1] <= ϵ < bounds[2]
                data_reduced[iq] += res.data[ib, iq]
            end
        else
            isnothing(kernel.integral) && error("Kernel must provide integral")
            ihi = kernel.integral(bounds[2] - ϵ)
            ilo = kernel.integral(bounds[1] - ϵ)
            data_reduced[iq] += res.data[ib, iq] * (ihi - ilo)
        end
    end
    StaticIntensities(res.crystal, res.qpts, data_reduced)
end
