# The Hamiltonian matrix H(q) of each wavevector b, mirroring
# `Sunny.hamiltonian_matrix!` without the Ewald term, which `accum_ewald!` adds.
# A thread owns row r, and sums its entries in the order of `Sunny.accum_table!`,
# so that H is exactly Hermitian.
@kernel function build_hamiltonian!(H, @Const(rowptr), @Const(cols), @Const(offset_index), @Const(vals),
                                    @Const(offsets), @Const(qs_supercell), regularization)
    r, b = @index(Global, NTuple)
    F = real(eltype(H))
    for c in axes(H, 2)
        H[r, c, b] = 0
    end
    q = qs_supercell[b]
    for e in rowptr[r]:rowptr[r+1]-1
        c = cols[e]
        v = cis(F(2π) * dot(q, offsets[offset_index[e]])) * vals[e]
        H[r, c, b] += r == c ? real(v) : v
    end
    H[r, r, b] += regularization
end

# H += T† Ã(q) T, the Ewald pair term of each wavevector b, mirroring
# `Sunny.accum_ewald!` for the data of `ewald_device`. Here Ã[cell, a, b′, b] is
# `Sunny.ewald_interaction_tensor` at q, with its cells in linear order. A
# thread owns row r, and sums its entries in the order of the host. The rows
# of H are then Hermitian only up to round-off, which `hermitianize!` removes.
@kernel function accum_ewald!(H, @Const(Ã), @Const(label_ptr), @Const(label_site), @Const(label_coef), @Const(ptr),
                              @Const(labels), @Const(coefs), @Const(atoms), @Const(cells), @Const(qs_supercell), dims)
    r, b = @index(Global, NTuple)
    F = real(eltype(H))
    q = qs_supercell[b]
    for e in label_ptr[r]:label_ptr[r+1]-1
        i = label_site[e]
        ce = label_coef[e]
        for j in eachindex(atoms)
            # The cell displacement wraps into the supercell with a winding w,
            # as in `Sunny.ewald_tensor_at`
            Δ = cells[j] .- cells[i]
            c = mod.(Δ, dims)
            cell = 1 + c[1] + dims[1] * (c[2] + dims[2] * c[3])
            w = Sunny.SVector{3, F}(fld.(Δ, dims))
            Aij = cis(-F(2π) * dot(q, w)) * Ã[cell, atoms[i], atoms[j], b]
            for f in ptr[j]:ptr[j+1]-1
                H[r, labels[f], b] += dot(ce, Aij * coefs[f])
            end
        end
    end
end

# H = (H + H†)/2 for each wavevector b, as `LinearAlgebra.hermitianpart!`
@kernel function hermitianize!(H)
    r, c, b = @index(Global, NTuple)
    if r == c
        H[r, r, b] = real(H[r, r, b])
    elseif r < c
        v = (H[r, c, b] + conj(H[c, r, b])) / 2
        H[r, c, b] = v
        H[c, r, b] = conj(v)
    end
end

# The observables in the Nambu basis, mirroring `Sunny.accum_observable_vectors!`.
# A thread owns the Nf flavors of site i for observable μ.
@kernel function observable_vectors!(u, @Const(coefs), @Const(positions), @Const(formfactors), @Const(qs_global), Nf)
    i, μ, b = @index(Global, NTuple)
    L = size(u, 1) ÷ 2
    q = qs_global[b]
    q2 = sum(abs2, q)  # unlike `Sunny.norm2`, in the precision of q
    for a in (i-1)*Nf+1:i*Nf
        u[a, μ, b] = 0
        u[L+a, μ, b] = 0
    end
    for p in axes(coefs, 3)
        pref = cis(dot(q, positions[i, p])) * compute_form_factor(formfactors[μ, i, p], q2)
        for a in (i-1)*Nf+1:i*Nf
            u[a, μ, b]   += pref * coefs[L+a, μ, p]
            u[L+a, μ, b] += pref * coefs[a, μ, p]
        end
    end
end

# Amplitudes A[μ, n] = ⟨0|Â_μ(q)†|n⟩ of each band n
@kernel function amplitudes!(A, @Const(u), @Const(T))
    μ, n, b = @index(Global, NTuple)
    s = zero(eltype(A))
    for a in axes(u, 1)
        s += conj(u[a, μ, b]) * T[a, n, b]
    end
    A[μ, n, b] = s
end

# The floating point components of x, an isbits composite of them, in memory
# order. Like `Sunny.real_components`, but without `reinterpret`, which fails to
# compile for GPUs.
@generated function real_components(x::T) where T
    exprs = Any[]
    walk(ex, S) = S <: Base.IEEEFloat ? push!(exprs, ex) :
        foreach(((i, F),) -> walk(:(getfield($ex, $i)), F), enumerate(fieldtypes(S)))
    walk(:x, T)
    return :(tuple($(exprs...)))
end

# The residues of the poles of χ″, as real components k of the measure,
# mirroring `Sunny.susceptibility_poles`. The combiner receives the pair
# correlations as a static vector. A pole at negative energy has a negative
# weight.
@kernel function residues_combiner!(disp, R, @Const(E), @Const(A), @Const(qs_global), combiner, corr_pairs, q0, Ncells)
    n, b = @index(Global, NTuple)
    ε = E[n, b]
    disp[n, q0+b] = ε
    corr = map(corr_pairs) do (μ, ν)
        A[μ, n, b] * conj(A[ν, n, b]) / Ncells
    end
    xs = real_components(combiner(qs_global[b], corr))
    for k in eachindex(xs)
        R[k, n, q0+b] = sign(ε) * xs[k]
    end
end

# Like `residues_combiner!`, for a combiner that cannot run on device. The
# correlations C = A A† / Ncells of a band are rank one, so component k of the
# combiner is tr(M_k C) = A† M_k A / Ncells, for the forms M_k of
# `Sunny.measure_forms`, given per wavevector or once for all.
@kernel function residues!(disp, R, @Const(E), @Const(A), @Const(M), q0, Ncells)
    n, b = @index(Global, NTuple)
    ε = E[n, b]
    disp[n, q0+b] = ε
    bM = size(M, 4) == 1 ? 1 : b
    for k in axes(M, 3)
        s = zero(eltype(A))
        for ν in axes(A, 1)
            Aν = A[ν, n, b]
            for μ in axes(A, 1)
                s += conj(A[μ, n, b]) * M[μ, ν, k, bM] * Aν
            end
        end
        R[k, n, q0+b] = sign(ε) * real(s) / Ncells
    end
end
