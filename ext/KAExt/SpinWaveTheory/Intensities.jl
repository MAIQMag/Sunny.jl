"""
    intensities(swt_d::SpinWaveTheoryDevice, qpts; energies, η=0, kernel=nothing, kT=0, batch_size=nothing,
                output=:host, threaded=false)

Like [`intensities`](@ref) for a `SpinWaveTheory`, but calculated on the
backend of `swt_d = to_device(swt; backend)`. Wavevectors are processed in
batches of `batch_size`, which by default is chosen from the free device
memory. With `output=:host`, the returned `Intensities` are on the host, and
are copied batch by batch. With `output=:device`, the returned
`IntensitiesDevice` keeps its data on the device; `Intensities(res)` copies
it to the host. The option `threaded` applies to the parts of the calculation
that run on the host.
"""
function Sunny.intensities(swt_d::SpinWaveTheoryDevice, qpts; energies, η=0, kernel=nothing, kT=0, batch_size=nothing,
                           output=:host, threaded=false)
    (; swt, backend) = swt_d
    (; sys, measure) = swt
    Sunny.num_observables(measure) == 0 && error("No observables! Construct SpinWaveTheory with a `measure` argument.")
    output in (:host, :device) || error("Option `output` must be :host or :device")
    qpts = convert(Sunny.AbstractQPoints, qpts)
    Ret = eltype(measure)
    F = precision_of(swt_d)
    Nq = length(qpts.qs)
    Nω = length(energies)
    budget = memory_budget(backend)

    (disp, R) = susceptibility_poles_device(swt_d, qpts; batch_size, budget, threaded)
    plan = prepare_broadening(backend, disp, R; energies, η, kernel, kT)
    K = plan.K

    # The broadening of a batch, and on output to the host also its copy, are
    # bounded by the budget
    bq = clamp(something(batch_size, budget ÷ max(plan.bytes_per_q, 1)), 1, max(Nq, 1))
    scratch = broadening_scratch(backend, plan, bq)
    if output == :device
        data = KA.allocate(backend, F, (K, Nω, Nq))
        for q0 in 0:bq:Nq-1
            broaden_batch!(backend, data, plan, q0, min(bq, Nq - q0), q0, scratch)
        end
        KA.synchronize(backend)
        data = reshape(reinterpret(precision_type(F, Ret), data), Nω, size(qpts.qs)...)
        return IntensitiesDevice(Sunny.orig_crystal(sys), qpts, collect(Float64, energies), data)
    else
        data = Array{F}(undef, K, Nω, Nq)
        buf = KA.allocate(backend, F, (K, Nω, bq))
        for q0 in 0:bq:Nq-1
            nb = min(bq, Nq - q0)
            broaden_batch!(backend, buf, plan, q0, nb, 0, scratch)
            KA.synchronize(backend)
            copyto!(data, K*Nω*q0 + 1, buf, 1, K*Nω*nb)
        end
        data = convert(Array{Float64}, data)
        data = reshape(Ret == Float64 ? data : Sunny.measure_values(Ret, data), Nω, size(qpts.qs)...)
        return Sunny.Intensities(Sunny.orig_crystal(sys), qpts, collect(Float64, energies), data)
    end
end

# Bytes of device memory that a calculation may use, a quarter of the free
# memory if an extension reports it
function memory_budget(backend)
    free = applicable(Sunny.device_free_memory, backend) ? Sunny.device_free_memory(backend) : 2^30
    return Int(max(free ÷ 4, 2^20))
end

Sunny.device_free_memory(::KA.CPU) = Sys.free_memory()

# Mirrors `Sunny.susceptibility_poles`. Returns the poles disp[n, iq] and the
# residues R[k, n, iq] by real component k of the measure, on device.
function susceptibility_poles_device(swt_d::SpinWaveTheoryDevice, qpts; batch_size, budget, threaded)
    (; swt, backend, quadratic_table, ewald, observable_coefs, regularization, measure_mode) = swt_d
    (; sys, measure) = swt
    (; corr_pairs, forms, positions, formfactors) = swt_d.measure
    cryst = Sunny.orig_crystal(sys)

    Na = Sunny.nsites(sys)
    F = precision_of(swt_d)
    Ncells = F(Na / Sunny.natoms(cryst))
    Nf = Sunny.nflavors(swt)
    L = Sunny.nbands(swt)
    n = 2L
    Nq = length(qpts.qs)
    Nobs = Sunny.num_observables(measure)
    K = Sunny.num_real_components(eltype(measure))

    # Up to three complex n×n matrices per wavevector (H, T, and the scratch of
    # the Jacobi eigensolver) dominate memory, unless the Ewald tensor does
    nÃ = isnothing(ewald) ? 0 : 9 * ncells(ewald.cache) * ewald.cache.na^2
    bytes_per_q = 2sizeof(F) * (3n^2 + 2n*Nobs + Nobs^2*K + nÃ) + sizeof(F)*n*(1 + K)
    batch_size = clamp(something(batch_size, budget ÷ bytes_per_q), 1, max(Nq, 1))

    disp = KA.allocate(backend, F, (n, Nq))
    R = KA.allocate(backend, F, (K, n, Nq))
    H = KA.allocate(backend, Complex{F}, (n, n, batch_size))
    T = KA.allocate(backend, Complex{F}, (n, n, batch_size))
    E = KA.allocate(backend, F, (n, batch_size))
    u = KA.allocate(backend, Complex{F}, (n, Nobs, batch_size))
    A = KA.allocate(backend, Complex{F}, (Nobs, n, batch_size))
    if !isnothing(ewald)
        Ã = KA.allocate(backend, Sunny.SMatrix{3, 3, Complex{F}, 9}, (ncells(ewald.cache), ewald.cache.na, ewald.cache.na, batch_size))
    end
    if measure_mode == :forms
        M_h = zeros(Complex{F}, Nobs, Nobs, K, batch_size)
        M = KA.allocate(backend, Complex{F}, size(M_h))
    end

    for q0 in 0:batch_size:Nq-1
        nb = min(batch_size, Nq - q0)
        qs = view(qpts.qs, q0+1:q0+nb)
        qs_supercell = [Sunny.to_supercell_rlu(sys, q) for q in qs]
        qs_global = [cryst.recipvecs * Vec3(q) for q in qs]

        # The measure as Hermitian forms per wavevector, if the combiner cannot
        # run on device
        if measure_mode == :forms
            Sunny.foreach_chunked(Returns(nothing), 1:nb; threaded) do _, b
                for (k, Mk) in enumerate(Sunny.measure_forms(measure, qs_global[b]))
                    view(M_h, :, :, k, b) .= Mk
                end
            end
            copyto!(M, M_h)
        end

        qs_supercell_d = to_backend(backend, F, qs_supercell)
        qs_global_d = to_backend(backend, F, qs_global)

        (; rowptr, cols, offset_index, vals, offsets) = quadratic_table
        build_hamiltonian!(backend)(H, rowptr, cols, offset_index, vals, offsets, qs_supercell_d, regularization;
                                    ndrange=(n, nb))
        if !isnothing(ewald)
            accum_ewald_batched!(backend, H, Ã, ewald, qs_supercell, qs_supercell_d, nb)
        end
        KA.synchronize(backend)
        bogoliubov_batched!(T, E, H, nb, swt_d, qs_supercell; threaded)

        observable_vectors!(backend)(u, observable_coefs, positions, formfactors, qs_global_d, Nf;
                                     ndrange=(Na, Nobs, nb))
        amplitudes!(backend)(A, u, T; ndrange=(Nobs, n, nb))
        if measure_mode == :combiner
            residues_combiner!(backend)(disp, R, E, A, qs_global_d, measure.combiner, corr_pairs, q0, Ncells; ndrange=(n, nb))
        else
            residues!(backend)(disp, R, E, A, measure_mode == :forms ? M : forms, q0, Ncells; ndrange=(n, nb))
        end
        KA.synchronize(backend)
    end

    return (disp, R)
end

# H += T† Ã(q) T for the first `nb` wavevectors, then the Hermitian part of H,
# as `Sunny.accum_ewald!` and `hermitianpart!` for each. Ã is scratch for the
# Ewald tensor at each wavevector, as `Sunny.ewald_interaction_tensor` gives it.
function accum_ewald_batched!(backend, H, Ã, ewald::QuadraticEwaldDevice{F}, qs_supercell, qs_supercell_d, nb) where F
    (; cache, ptr, labels, coefs, atoms, cells, label_ptr, label_site, label_coef) = ewald
    (; params, positions, real_ptr, real_shifts, real_mats, ex_index, ex_winding, ex_mats) = cache
    n = size(H, 1)
    ewald_qs = ewald_wavevector.(qs_supercell)
    ewald_tensor!(backend)(Ã, params, positions, real_ptr, real_shifts, real_mats, ex_index, ex_winding, ex_mats,
                           to_backend(backend, F, first.(ewald_qs)), to_backend(backend, F, getindex.(ewald_qs, 2)),
                           to_backend(backend, last.(ewald_qs)); ndrange=(ncells(cache), cache.na, cache.na, nb))
    accum_ewald!(backend)(H, Ã, label_ptr, label_site, label_coef, ptr, labels, coefs, atoms, cells, qs_supercell_d,
                          cache.dims; ndrange=(n, nb))
    hermitianize!(backend)(H; ndrange=(n, n, nb))
end
