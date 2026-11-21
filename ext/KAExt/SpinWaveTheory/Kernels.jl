# The Hamiltonian matrix H(q) of each wavevector b, mirroring
# `Sunny.hamiltonian_matrix!` without the Ewald term, which `accum_ewald!` adds.
# A thread owns row r, and sums its entries in the order of `Sunny.accum_table!`,
# so that H is exactly Hermitian.
@kernel function build_hamiltonian!(H, table, @Const(qs_supercell), regularization)
    r, b = @index(Global, NTuple)
    (; rowptr, cols, offset_index, vals, offsets) = table
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

# The entry A[cell, i, j] of `Sunny.ewald_interaction_tensor` at the wavevector
# (q, q0, q0_zero) of `ewald_wavevector`, before the excluded couplings, with
# `cell` the linear index of a cell of the supercell. The Fourier modes and the
# real-space terms sum in the order of the host. The phases of the Fourier modes
# are one `cis` each, rather than products of 1D factors.
@inline function ewald_entry(cache::EwaldTensorCacheDevice{F}, q, q0, q0_zero, ncells, na, cell, i, j) where F
    (; params, positions, real_ptr, real_shifts, real_mats) = cache
    (; dims, recipvecs, mmax, kmax², σ², invV) = params
    c = cell - 1
    c1 = c % dims[1]
    c2 = (c ÷ dims[1]) % dims[2]
    c3 = c ÷ (dims[1] * dims[2])
    x = (Sunny.SVector{3, F}(c1, c2, c3) + positions[j] - positions[i]) ./ Sunny.SVector{3, F}(dims)

    acc = zero(Sunny.SMatrix{3, 3, Complex{F}, 9})
    for m1 in -mmax[1]:mmax[1], m2 in -mmax[2]:mmax[2], m3 in -mmax[3]:mmax[3]
        # The k = 0 mode is the `demag` term instead
        if !(q0_zero && m1 == 0 && m2 == 0 && m3 == 0)
            mq = Sunny.SVector{3, F}(m1, m2, m3) + q0
            k = recipvecs * mq
            k² = dot(k, k)
            if k² <= kmax²
                Aₖ = (invV * exp(-σ² * k² / 2) / k²) * (k * k')
                acc += cis(-F(2π) * dot(mq, x)) * Aₖ
            end
        end
    end

    q0_zero && (acc += params.demag_V)
    idx = cell + ncells * ((i - 1) + na * (j - 1))
    for e in real_ptr[idx]:real_ptr[idx+1]-1
        acc += cis(F(2π) * dot(q, real_shifts[e])) * real_mats[e]
    end
    (cell == 1 && i == j) && (acc += params.self_energy)
    return params.μ0_μB² * acc
end

# The Ewald tensor Ã[cell, i, j, b] of each wavevector b, mirroring
# `Sunny.ewald_interaction_tensor`, with its cells in linear order. As on the
# host, the zero-offset cell is completed by Hermiticity, then the excluded
# couplings are removed. A single call of `ewald_entry`, which is inlined,
# keeps the kernel small: a call per branch tripled the compile time on POCL.
@kernel function ewald_tensor!(Ã, cache, @Const(qs), @Const(q0s), @Const(q0_zeros))
    cell, i, j, b = @index(Global, NTuple)
    (; ex_index, ex_winding, ex_mats) = cache
    F = real(eltype(eltype(Ã)))
    ncells = size(Ã, 1)
    na = size(Ã, 2)
    q = qs[b]
    q0 = q0s[b]
    q0_zero = q0_zeros[b]
    flip = cell == 1 && i > j
    (i′, j′) = flip ? (j, i) : (i, j)
    v = ewald_entry(cache, q, q0, q0_zero, ncells, na, cell, i′, j′)
    flip && (v = v')
    idx = cell + ncells * ((i - 1) + na * (j - 1))
    for e in eachindex(ex_index)
        if ex_index[e] == idx
            v -= cis(F(2π) * dot(q, ex_winding[e])) * ex_mats[e]
        end
    end
    Ã[cell, i, j, b] = v
end

# H += T† Ã(q) T, the Ewald pair term of each wavevector b, mirroring
# `Sunny.accum_ewald!` for a `QuadraticEwaldDevice`. Here Ã[cell, a, b′, b] is
# `Sunny.ewald_interaction_tensor` at q, with its cells in linear order. A
# thread owns row r, and sums its entries in the order of the host. The rows
# of H are then Hermitian only up to round-off, which `hermitianize!` removes.
@kernel function accum_ewald!(H, @Const(Ã), ewald, @Const(qs_supercell))
    r, b = @index(Global, NTuple)
    (; ptr, labels, coefs, atoms, cells, label_ptr, label_site, label_coef) = ewald
    dims = ewald.cache.dims
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
@kernel function observable_vectors!(u, @Const(coefs), measure, @Const(qs_global), Nf)
    i, μ, b = @index(Global, NTuple)
    (; positions, formfactors) = measure
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
