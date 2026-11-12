"""
    intensities(swt_d::SpinWaveTheoryDevice, qpts; energies, η=0, kernel=nothing, kT=0, batch_size=nothing, threaded=false)

Like [`intensities`](@ref) for a `SpinWaveTheory`, but calculated on the
backend of `swt_d = to_device(swt; backend)`. Wavevectors are processed in
batches of `batch_size`, which by default is chosen to bound memory use. The
returned `Intensities` are on the host. The option `threaded` applies to the
parts of the calculation that run on the host.
"""
function Sunny.intensities(swt_d::SpinWaveTheoryDevice, qpts; energies, η=0, kernel=nothing, kT=0, batch_size=nothing, threaded=false)
    (; swt, backend) = swt_d
    (; sys, measure) = swt
    Sunny.num_observables(measure) == 0 && error("No observables! Construct SpinWaveTheory with a `measure` argument.")
    qpts = convert(Sunny.AbstractQPoints, qpts)
    Ret = eltype(measure)

    (disp, R) = susceptibility_poles_device(swt_d, qpts; batch_size, threaded)
    data_d = broaden_poles_device(backend, Ret, disp, R; energies, η, kernel, kT)
    data = Sunny.measure_values(Ret, Array(data_d))
    data = reshape(data, length(energies), size(qpts.qs)...)
    return Sunny.Intensities(Sunny.orig_crystal(sys), qpts, collect(Float64, energies), data)
end

# Mirrors `Sunny.susceptibility_poles`. Returns the poles disp[n, iq] and the
# residues R[k, n, iq] by real component k of the measure, on device.
function susceptibility_poles_device(swt_d::SpinWaveTheoryDevice, qpts; batch_size, threaded)
    (; swt, backend, data) = swt_d
    (; sys, measure) = swt
    cryst = Sunny.orig_crystal(sys)

    Na = Sunny.nsites(sys)
    Ncells = Na / Sunny.natoms(cryst)
    Nf = Sunny.nflavors(swt)
    L = Sunny.nbands(swt)
    n = 2L
    Nq = length(qpts.qs)
    Nobs = Sunny.num_observables(measure)
    K = Sunny.num_real_components(eltype(measure))

    # Two complex n×n matrices per wavevector dominate memory; by default bound
    # them to about 256 MiB
    batch_size = clamp(something(batch_size, 2^28 ÷ (32n^2)), 1, max(Nq, 1))

    disp = KA.allocate(backend, Float64, (n, Nq))
    R = KA.allocate(backend, Float64, (K, n, Nq))
    H = KA.allocate(backend, ComplexF64, (n, n, batch_size))
    T = KA.allocate(backend, ComplexF64, (n, n, batch_size))
    E = KA.allocate(backend, Float64, (n, batch_size))
    u = KA.allocate(backend, ComplexF64, (n, Nobs, batch_size))
    A = KA.allocate(backend, ComplexF64, (Nobs, n, batch_size))
    M_h = zeros(ComplexF64, Nobs, Nobs, K, batch_size)
    M = KA.allocate(backend, ComplexF64, size(M_h))

    for q0 in 0:batch_size:Nq-1
        nb = min(batch_size, Nq - q0)
        qs = view(qpts.qs, q0+1:q0+nb)
        qs_supercell = [Sunny.to_supercell_rlu(sys, q) for q in qs]
        qs_global = [cryst.recipvecs * Vec3(q) for q in qs]

        # The measure as Hermitian forms, which need not be evaluated on device
        Sunny.foreach_chunked(Returns(nothing), 1:nb; threaded) do _, b
            for (k, Mk) in enumerate(Sunny.measure_forms(measure, qs_global[b]))
                view(M_h, :, :, k, b) .= Mk
            end
        end
        copyto!(M, M_h)

        qs_supercell_d = to_backend(backend, qs_supercell)
        qs_global_d = to_backend(backend, qs_global)

        build_hamiltonian!(backend)(H, data.rowptr, data.cols, data.offset_index, data.vals, data.offsets,
                                    qs_supercell_d, swt.regularization; ndrange=(n, nb))
        KA.synchronize(backend)
        bogoliubov_batched!(T, E, H, nb, swt, qs_supercell; threaded)

        observable_vectors!(backend)(u, data.coefs, data.positions, data.formfactors, qs_global_d, Nf; ndrange=(Na, Nobs, nb))
        amplitudes!(backend)(A, u, T; ndrange=(Nobs, n, nb))
        residues!(backend)(disp, R, E, A, M, q0, Ncells; ndrange=(n, nb))
        KA.synchronize(backend)
    end

    return (disp, R)
end
