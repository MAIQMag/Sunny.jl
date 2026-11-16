# Status codes of the batched Bogoliubov solvers, per wavevector. A positive
# code j means the Cholesky factorization of H failed at column j, i.e., H is
# not positive definite. STATUS_NOCONVERGE is for a vendor eigensolver that
# reports failure to converge.
const STATUS_OK = Int32(0)
const STATUS_NONFINITE = Int32(-1)
const STATUS_INERTIA = Int32(-2)
const STATUS_NOCONVERGE = Int32(-3)

# Solves the Bogoliubov problem for the first `nb` Hamiltonians of `H`, storing
# the transformations in `T` and the energies in `E`, as `Sunny.bogoliubov!`
# does for one. The contents of H are overwritten.
function bogoliubov_batched!(T, E, H, nb, swt_d, qs_supercell; threaded)
    (; swt, eigensolver, backend) = swt_d
    if eigensolver == :host
        bogoliubov_host!(T, E, H, nb, swt, qs_supercell; threaded)
        return
    end
    n = size(H, 1)
    status = KA.allocate(backend, Int32, nb)
    if eigensolver == :jacobi
        W = KA.allocate(backend, eltype(H), (n, n, nb))
        bogoliubov_jacobi!(backend)(T, E, H, W, status; ndrange=nb)
        KA.synchronize(backend)
    else
        @assert eigensolver == :vendor
        Sunny.vendor_bogoliubov!(T, E, H, nb, status)
    end
    throw_bogoliubov_status(Array(status), swt, qs_supercell)
end

# Errors of `Sunny.bogoliubov_or_throw!` for the first failed wavevector
function throw_bogoliubov_status(status, swt, qs_supercell)
    b = findfirst(!=(STATUS_OK), status)
    isnothing(b) && return
    s = status[b]
    s == STATUS_NONFINITE && throw(ArgumentError("Hamiltonian contains non-finite entries"))
    s == STATUS_INERTIA && error("Bogoliubov eigenvalues lack the expected inertia")
    s == STATUS_NOCONVERGE && error("Vendor eigensolver failed to converge; use `to_device(swt; eigensolver=:jacobi)` or `:host`")
    throw(Sunny.InstabilityError(Sunny.instability_message(swt, qs_supercell[b])))
end

# Completes a method of `Sunny.vendor_bogoliubov!` after its solver of Ĩ t = λ
# H t for the first `nb` Hamiltonians. On entry, `T[:, :, b]` holds the
# eigenvectors t with t† H t = 1, and `λ[:, b]` the eigenvalues in ascending
# order, as LAPACK's zhegvd gives them. Where the `status` is still STATUS_OK,
# it becomes STATUS_INERTIA unless exactly L of the λ are negative, by
# Sylvester's law of inertia. Then T and E take the order, normalization and
# energies of `Sunny.bogoliubov!`. Works on any backend that supports
# broadcasting and indexing by a device vector.
function Sunny.finish_vendor_bogoliubov!(T, E, λ, nb, status)
    n = size(T, 1)
    L = n ÷ 2
    Tb = view(T, :, :, 1:nb)
    λb = view(λ, :, 1:nb)
    sb = view(status, 1:nb)

    inertia = vec((view(λb, L:L, :) .< 0) .& (view(λb, L+1:L+1, :) .> 0))
    sb .= ifelse.((sb .== STATUS_OK) .& .!inertia, STATUS_INERTIA, sb)

    perm = KA.allocate(KA.get_backend(T), Int, n)
    copyto!(perm, [L+1:2L; L:-1:1])
    λp = λb[perm, :]
    Tb .= Tb[:, perm, :] ./ sqrt.(abs.(reshape(λp, 1, n, nb)))
    view(E, :, 1:nb) .= 1 ./ λp
    return
end

# Copies to the host and calls LAPACK per wavevector, in Float64 whatever the
# precision of H
function bogoliubov_host!(T, E, H, nb, swt, qs_supercell; threaded)
    n = size(H, 1)
    L = n ÷ 2
    on_host = H isa Array{ComplexF64}
    (H_h, T_h, E_h) = on_host ? (H, T, E) : (Array(view(H, :, :, 1:nb)), similar(Array{ComplexF64}, n, n, nb), similar(Array{Float64}, n, nb))
    newbuf() = (ws = Sunny.BogoliubovWorkspace(L), Hm = zeros(ComplexF64, n, n))
    Sunny.foreach_chunked(newbuf, 1:nb; threaded) do (; ws, Hm), b
        copyto!(Hm, view(H_h, :, :, b))
        ε = Sunny.bogoliubov_or_throw!(() -> Sunny.instability_message(swt, qs_supercell[b]), ws, Hm)
        view(T_h, :, :, b) .= ws.T
        view(E_h, :, b) .= ε
    end
    if !on_host
        copyto!(view(T, :, :, 1:nb), convert(Array{eltype(T)}, T_h))
        copyto!(view(E, :, 1:nb), convert(Array{eltype(E)}, E_h))
    end
    return
end

# The position of eigenvalue j of `λ` in the order of `Sunny.bogoliubov!`. That
# function sorts the λ ascending, as LAPACK returns them, and then takes the
# permutation [L+1:2L; L:-1:1], so that the positive λ come first. Ties are
# broken by index.
@inline function bogoliubov_position(λ, j, b, L)
    m = 1
    for i in axes(λ, 1)
        m += (λ[i, b] < λ[j, b]) | ((λ[i, b] == λ[j, b]) & (i < j))
    end
    return m > L ? m - L : 2L + 1 - m
end

# One rotation of the cyclic Jacobi method, which zeroes A[p, q] of the
# Hermitian matrix A by A ← G† A G, and accumulates V ← V G. With A[p, q] = g
# exp(iφ), the unitary G = D R is a phase D = diag(1, exp(-iφ)) on (p, q),
# which makes the element real, followed by the real rotation of Numerical
# Recipes, Eq. (11.1.8).
@inline function jacobi_rotate!(A, V, p, q, b)
    apq = A[p, q, b]
    g = abs(apq)
    eiφ = apq / g
    app = real(A[p, p, b])
    aqq = real(A[q, q, b])
    θ = (aqq - app) / 2g
    # Beyond 1/√ε, the relative error of t ≈ 1/2θ is below ε, and θ² may overflow
    t = abs(θ) > inv(sqrt(eps(typeof(θ)))) ? 1 / 2θ : copysign(1 / (abs(θ) + sqrt(θ^2 + 1)), θ)
    c = 1 / sqrt(t^2 + 1)
    s = t * c
    (Gpp, Gpq, Gqp, Gqq) = (complex(c), complex(s), -s * conj(eiφ), c * conj(eiφ))
    for k in axes(A, 1)
        (akp, akq) = (A[k, p, b], A[k, q, b])
        A[k, p, b] = akp * Gpp + akq * Gqp
        A[k, q, b] = akp * Gpq + akq * Gqq
    end
    for k in axes(A, 1)
        (apk, aqk) = (A[p, k, b], A[q, k, b])
        A[p, k, b] = conj(Gpp) * apk + conj(Gqp) * aqk
        A[q, k, b] = conj(Gpq) * apk + conj(Gqq) * aqk
    end
    A[p, q, b] = A[q, p, b] = 0
    A[p, p, b] = app - t * g
    A[q, q, b] = aqq + t * g
    for k in axes(V, 1)
        (vkp, vkq) = (V[k, p, b], V[k, q, b])
        V[k, p, b] = vkp * Gpp + vkq * Gqp
        V[k, q, b] = vkp * Gpq + vkq * Gqq
    end
end

# A portable batched variant of `Sunny.bogoliubov!`, with one thread per
# wavevector b. The generalized problem Ĩ t = λ H t is reduced by the Cholesky
# factor H = K K† to the Hermitian A = K⁻¹ Ĩ K⁻†, whose eigenvectors y give t =
# K⁻† y, normalized as LAPACK's zhegvd does, t† H t = 1. The matrix A is
# diagonalized by the cyclic Jacobi method, which is accurate and needs no
# workspace beyond W. On exit, H, W and the status are overwritten.
@kernel function bogoliubov_jacobi!(T, E, H, W, status, max_sweeps=50)
    b = @index(Global)
    n = size(H, 1)
    L = n ÷ 2
    C = eltype(H)
    st = STATUS_OK

    for j in 1:n, i in 1:n
        isfinite(H[i, j, b]) || (st = STATUS_NONFINITE)
    end

    # Lower Cholesky factor K, in place in H
    if st == STATUS_OK
        for j in 1:n
            d = real(H[j, j, b])
            for k in 1:j-1
                d -= abs2(H[j, k, b])
            end
            if !(d > 0)
                st = Int32(j)
                break
            end
            d = sqrt(d)
            H[j, j, b] = d
            for i in j+1:n
                x = H[i, j, b]
                for k in 1:j-1
                    x -= H[i, k, b] * conj(H[j, k, b])
                end
                H[i, j, b] = x / d
            end
        end
    end

    if st == STATUS_OK
        # W = K⁻¹, lower triangular
        for j in 1:n
            for i in 1:j-1
                W[i, j, b] = 0
            end
            W[j, j, b] = 1 / real(H[j, j, b])
            for i in j+1:n
                x = zero(C)
                for k in j:i-1
                    x += H[i, k, b] * W[k, j, b]
                end
                W[i, j, b] = -x / real(H[i, i, b])
            end
        end

        # A = K⁻¹ Ĩ K⁻† in T, with Ĩ = diag(1, …, 1, -1, …, -1)
        for j in 1:n, i in j:n
            x = zero(C)
            for k in 1:j
                x += ifelse(k <= L, 1, -1) * W[i, k, b] * conj(W[j, k, b])
            end
            T[i, j, b] = x
            T[j, i, b] = conj(x)
        end
        for j in 1:n
            T[j, j, b] = real(T[j, j, b])
        end

        # Eigenvectors V accumulate in H, which is no longer needed
        for j in 1:n, i in 1:n
            H[i, j, b] = i == j
        end
        for _ in 1:max_sweeps
            rotated = false
            for q in 2:n, p in 1:q-1
                g = abs(T[p, q, b])
                if g > eps(real(C)) * sqrt(abs(real(T[p, p, b]) * real(T[q, q, b])))
                    jacobi_rotate!(T, H, p, q, b)
                    rotated = true
                end
            end
            rotated || break
        end
        for j in 1:n
            E[j, b] = real(T[j, j, b])
        end

        # Exactly L of the λ are negative, by Sylvester's law of inertia
        nneg = 0
        for j in 1:n
            nneg += E[j, b] < 0
        end
        nneg == L || (st = STATUS_INERTIA)
    end

    if st == STATUS_OK
        # Each t = K⁻† y, normalized for para-unitarity, in its sorted column
        for j in 1:n
            pos = bogoliubov_position(E, j, b, L)
            c = 1 / sqrt(abs(E[j, b]))
            for i in 1:n
                x = zero(C)
                for k in i:n
                    x += conj(W[k, i, b]) * H[k, j, b]
                end
                T[i, pos, b] = c * x
            end
        end
        # W is now free to hold the sorted energies 1/λ
        for j in 1:n
            W[bogoliubov_position(E, j, b, L), 1, b] = 1 / E[j, b]
        end
        for j in 1:n
            E[j, b] = real(W[j, 1, b])
        end
    end
    status[b] = st
end
