# Solves the Bogoliubov problem for the first `nb` Hamiltonians of `H`, storing
# the transformations in `T` and the energies in `E`, as `Sunny.bogoliubov!`
# does for one. This fallback copies to the host and calls LAPACK per
# wavevector. Backends with batched eigensolvers may specialize it.
function bogoliubov_batched!(T, E, H, nb, swt::Sunny.SpinWaveTheory, qs_supercell; threaded)
    n = size(H, 1)
    L = n ÷ 2
    on_host = H isa Array
    (H_h, T_h, E_h) = on_host ? (H, T, E) : (Array(view(H, :, :, 1:nb)), similar(Array{ComplexF64}, n, n, nb), similar(Array{Float64}, n, nb))
    newbuf() = (ws = Sunny.BogoliubovWorkspace(L), Hm = zeros(ComplexF64, n, n))
    Sunny.foreach_chunked(newbuf, 1:nb; threaded) do (; ws, Hm), b
        copyto!(Hm, view(H_h, :, :, b))
        ε = Sunny.bogoliubov_or_throw!(() -> Sunny.instability_message(swt, qs_supercell[b]), ws, Hm)
        view(T_h, :, :, b) .= ws.T
        view(E_h, :, b) .= ε
    end
    if !on_host
        copyto!(view(T, :, :, 1:nb), T_h)
        copyto!(view(E, :, 1:nb), E_h)
    end
    return nothing
end
