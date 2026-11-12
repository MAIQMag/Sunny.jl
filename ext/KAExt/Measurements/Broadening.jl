# Isbits wrappers for the broadening kernels of `Sunny.broaden_poles`, called
# as k(ε, ω) for a pole at ε and an output energy ω

struct StationaryKernel{F}
    f :: F
end
(k::StationaryKernel)(ε, ω) = k.f(ω - ε)

struct NonstationaryKernel{F}
    f :: F
end
(k::NonstationaryKernel)(ε, ω) = k.f(ε, ω)

device_kernel(k::Sunny.Broadening) = StationaryKernel(k.kernel)
device_kernel(k::Sunny.NonstationaryBroadening) = NonstationaryKernel(k.kernel)

# Mirrors `Sunny.thermal_factor`, without the assertion on kT
thermal_factor(ω, kT) = iszero(kT) ? (ω > 0 ? 1.0 : 0.0) : -1 / expm1(-ω/kT)

# Lorentzian of half width η times the thermal factor, mirroring the kernel of
# `Sunny.broaden_poles` and `Sunny.thermal_factor_at`
struct LorentzianThermal
    η  :: Float64
    kT :: Float64
end
function (k::LorentzianThermal)(ε, ω)
    (; η, kT) = k
    L = (η/π) / ((ω - ε)^2 + η^2)
    th = if iszero(kT)
        ω > 0 ? 1.0 : iszero(ω) ? 0.5 : 0.0
    elseif iszero(ω)
        2kT * ε / (ε^2 + η^2)
    else
        -1 / expm1(-ω/kT)
    end
    return L * th
end

# Accumulates each pole of weight W, broadened by `kern`, into data[k, iω, iq].
# A thread owns one output, and sums over the poles in the order of
# `Sunny.broaden_peaks!`. Poles with `keep` false are below its cutoff.
@kernel function broaden_peaks!(data, @Const(disp), @Const(W), @Const(keep), @Const(energies), kern)
    iω, iq = @index(Global, NTuple)
    ω = energies[iω]
    for k in axes(data, 1)
        data[k, iω, iq] = 0
    end
    for n in axes(disp, 1)
        keep[n, iq] || continue
        c = kern(disp[n, iq], ω)
        for k in axes(data, 1)
            data[k, iω, iq] += c * W[k, n, iq]
        end
    end
end

# Sums of the poles at each node z_j of `Sunny.RetardedBroadening`, split into
# the real and imaginary parts of 1/(ε - z_j)
@kernel function retarded_susceptibility!(χr, χi, @Const(disp), @Const(R), @Const(zs))
    j, iq = @index(Global, NTuple)
    for k in axes(R, 1)
        χr[k, j, iq] = 0
        χi[k, j, iq] = 0
    end
    z = zs[j]
    for n in axes(disp, 1)
        g = 1 / (disp[n, iq] - z)
        for k in axes(R, 1)
            χr[k, j, iq] += real(g) * R[k, n, iq]
            χi[k, j, iq] += imag(g) * R[k, n, iq]
        end
    end
end

# The quadrature weights of `Sunny.RetardedBroadening` applied to the sums at
# its nodes. Node j contributes to the energies lo[j]:hi[j].
@kernel function retarded_apply!(data, @Const(χr), @Const(χi), @Const(Wr), @Const(Wi), @Const(lo), @Const(hi))
    i, iq = @index(Global, NTuple)
    for k in axes(data, 1)
        data[k, i, iq] = 0
    end
    for j in eachindex(lo)
        lo[j] <= i <= hi[j] || continue
        for k in axes(data, 1)
            data[k, i, iq] += Wr[i, j] * χi[k, j, iq] + Wi[i, j] * χr[k, j, iq]
        end
    end
end

# The weight of each pole as a value of the measure type `Ret`, and whether it
# exceeds the cutoff of `Sunny.broaden_peaks!`
function peak_cutoff_mask(::Type{Ret}, W_h) where Ret
    norms = norm.(Sunny.measure_values(Ret, W_h))
    cutoff = 1e-12 * Sunny.Statistics.quantile(vec(norms), 0.95)
    return norms .>= cutoff
end

# Mirrors `Sunny.broaden_poles` for poles `disp` and residues R[k, n, iq] by
# real component k of the measure type `Ret`. Returns data[k, iω, iq].
function broaden_poles_device(backend, ::Type{Ret}, disp, R; energies, η, kernel, kT) where Ret
    η >= 0 || error("Require η ≥ 0")
    kT >= 0 || error("Require kT ≥ 0")
    iszero(η) && isnothing(kernel) && error("Specify a `kernel`, a broadening `η`, or both.")
    issorted(energies) || error("Energies must be sorted")

    K = size(R, 1)
    Nq = size(disp, 2)
    Nω = length(energies)
    energies_d = to_backend(backend, collect(Float64, energies))
    data = KA.allocate(backend, Float64, (K, Nω, Nq))

    if iszero(η) || isnothing(kernel)
        (disp_h, R_h) = (Array(disp), Array(R))
        if iszero(η)
            W_h = thermal_factor.(reshape(disp_h, 1, size(disp_h)...), kT) .* R_h
            kern = device_kernel(kernel)
        else
            W_h = R_h
            kern = LorentzianThermal(η, kT)
        end
        keep = to_backend(backend, peak_cutoff_mask(Ret, W_h))
        W = to_backend(backend, W_h)
        broaden_peaks!(backend)(data, disp, W, keep, energies_d, kern; ndrange=(Nω, Nq))
    else
        (; zs, W, ranges) = Sunny.RetardedBroadening(energies; η, kernel, kT, order=16, cutoff=1e-16)
        Nz = length(zs)
        χr = KA.allocate(backend, Float64, (K, Nz, Nq))
        χi = KA.allocate(backend, Float64, (K, Nz, Nq))
        Nz > 0 && retarded_susceptibility!(backend)(χr, χi, disp, R, to_backend(backend, zs); ndrange=(Nz, Nq))
        Wr = to_backend(backend, real(W) / π)
        Wi = to_backend(backend, imag(W) / π)
        lo = to_backend(backend, Int32[first(r) for r in ranges])
        hi = to_backend(backend, Int32[last(r) for r in ranges])
        retarded_apply!(backend)(data, χr, χi, Wr, Wi, lo, hi; ndrange=(Nω, Nq))
    end
    KA.synchronize(backend)
    return data
end
