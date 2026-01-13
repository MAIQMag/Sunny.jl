# Set the dynamical quadratic Hamiltonian matrix in SU(N) mode. 
function swt_hamiltonian_SUN!(H::Matrix{ComplexF64}, swt::SpinWaveTheory, q_reshaped::Vec3)
    (; sys, data) = swt
    (; spins_localized) = data
    (; gs) = sys

    N = sys.Ns[1]
    Na = natoms(sys.crystal)
    L = (N-1) * Na

    # Clear the Hamiltonian
    @assert size(H) == (2L, 2L)
    H .= 0
    blockdims = (N-1, Na, N-1, Na)
    H11 = reshape(view(H, 1:L, 1:L), blockdims)
    H12 = reshape(view(H, 1:L, L+1:2L), blockdims)
    H21 = reshape(view(H, L+1:2L, 1:L), blockdims)
    H22 = reshape(view(H, L+1:2L, L+1:2L), blockdims)

    for (i, int) in enumerate(sys.interactions_union)

        # Onsite coupling, including Zeeman. Note that op has already been
        # transformed according to the local frame of sublattice i.
        op = int.onsite
        for m in 1:N-1
            for n in 1:N-1
                c = op[m, n] - δ(m, n) * op[N, N]
                H11[m, i, n, i] += c
                H22[n, i, m, i] += c
            end
        end

        for coupling in int.pair
            (; isculled, bond) = coupling
            isculled && break

            @assert i == bond.i
            j = bond.j

            phase = exp(2π*im * dot(q_reshaped, bond.n)) # Phase associated with periodic wrapping

            # Set "general" pair interactions of the form Aᵢ⊗Bⱼ. Note that Aᵢ
            # and Bᵢ have already been transformed according to the local frames
            # of sublattice i and j, respectively.
            for (Ai, Bj) in coupling.general.data 
                for m in 1:N-1, n in 1:N-1
                    c = (Ai[m,n] - δ(m,n)*Ai[N,N]) * (Bj[N,N])
                    H11[m, i, n, i] += c
                    H22[n, i, m, i] += c

                    c = Ai[N,N] * (Bj[m,n] - δ(m,n)*Bj[N,N])
                    H11[m, j, n, j] += c
                    H22[n, j, m, j] += c

                    c = Ai[m,N] * Bj[N,n]
                    H11[m, i, n, j] += c * phase
                    H22[n, j, m, i] += c * conj(phase)

                    c = Ai[N,m] * Bj[n,N]
                    H11[n, j, m, i] += c * conj(phase)
                    H22[m, i, n, j] += c * phase

                    c = Ai[m,N] * Bj[n,N]
                    H12[m, i, n, j] += c * phase
                    H12[n, j, m, i] += c * conj(phase)
                    H21[n, j, m, i] += conj(c) * conj(phase)
                    H21[m, i, n, j] += conj(c) * phase
                end
            end
        end
    end

    if !isnothing(sys.ewald)
        (; demag, μ0_μB², A) = sys.ewald
        N = sys.Ns[1]

        # Interaction matrix for wavevector (0,0,0). It could be recalculated as:
        # precompute_dipole_ewald(sys.crystal, (1,1,1), demag) * μ0_μB²
        A0 = reshape(A, Na, Na)

        # Interaction matrix for wavevector q
        Aq = precompute_dipole_ewald_at_wavevector(sys.crystal, (1,1,1), demag, q_reshaped) * μ0_μB²
        Aq = reshape(Aq, Na, Na)

        for i in 1:Na, j in 1:Na
            # An ordered pair of magnetic moments contribute (μᵢ A μⱼ)/2 to the
            # energy, where μ = - g S. A symmetric contribution will appear for
            # the bond reversal (i, j) → (j, i).
            J = gs[i]' * Aq[i, j] * gs[j] / 2
            J0 = gs[i]' * A0[i, j] * gs[j] / 2

            for α in 1:3, β in 1:3
                Ai = spins_localized[α, i]
                Bj = spins_localized[β, j]

                for m in 1:N-1, n in 1:N-1
                    c = (Ai[m,n] - δ(m,n)*Ai[N,N]) * (Bj[N,N])
                    H11[m, i, n, i] += c * J0[α, β]
                    H22[n, i, m, i] += c * J0[α, β]

                    c = Ai[N,N] * (Bj[m,n] - δ(m,n)*Bj[N,N])
                    H11[m, j, n, j] += c * J0[α, β]
                    H22[n, j, m, j] += c * J0[α, β]

                    c = Ai[m,N] * Bj[N,n]
                    H11[m, i, n, j] += c * J[α, β]
                    H22[n, j, m, i] += c * conj(J[α, β])

                    c = Ai[N,m] * Bj[n,N]
                    H11[n, j, m, i] += c * conj(J[α, β])
                    H22[m, i, n, j] += c * J[α, β]

                    c = Ai[m,N] * Bj[n,N]
                    H12[m, i, n, j] += c * J[α, β]
                    H12[n, j, m, i] += c * conj(J[α, β])
                    H21[n, j, m, i] += conj(c) * conj(J[α, β])
                    H21[m, i, n, j] += conj(c) * J[α, β]
                end
            end
        end
    end

    # H must be hermitian up to round-off errors
    @assert diffnorm2(H, H') < 1e-12

    # Make H exactly hermitian
    hermitianpart!(H)

    # Add small constant shift for positive-definiteness
    for i in 1:2L
        H[i,i] += swt.regularization
    end
end

function swt_hamiltonian_dipole!(H::CUDA.CuArray{ComplexF64, 3}, swt::SpinWaveTheoryDevice, qs_reshaped, qs::CUDA.CuArray{Sunny.Vec3})
    L = Sunny.nbands(swt)
    Nq = size(qs, 1)
    @assert size(H, 3) == Nq
    @assert size(view(H,:,:,1)) == (2L, 2L)

    H .= 0.0

    kernel = CUDA.@cuda launch=false fill_matrix(H, swt, qs_reshaped, qs, L)
    config = launch_configuration(kernel.fun)
    threads = Base.min(Nq, config.threads)
    blocks = cld(Nq, threads)
    kernel(H, swt, qs_reshaped, qs, L; threads=threads, blocks=blocks)
end