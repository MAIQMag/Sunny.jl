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

# The kernels of `Sunny.gaussian` and `Sunny.lorentzian` in precision F. The
# closures of Sunny capture their widths in Float64.
struct GaussianKernel{F}
    inv_2σ2 :: F
    norm    :: F
end
GaussianKernel{F}(σ) where F = GaussianKernel{F}(1 / 2σ^2, 1 / √(2π*σ^2))
(k::GaussianKernel)(x) = exp(-x^2 * k.inv_2σ2) * k.norm

struct LorentzianKernel{F}
    Γ2  :: F  # half width Γ/2
    Γ2π :: F
end
LorentzianKernel{F}(Γ) where F = LorentzianKernel{F}(Γ/2, (Γ/2) / π)
(k::LorentzianKernel)(x) = k.Γ2π / (x^2 + k.Γ2^2)

function device_kernel(k::Sunny.Broadening, F)
    k.name == "Gaussian" && return StationaryKernel(GaussianKernel{F}(k.fwhm / 2√(2log(2))))
    k.name == "Lorentzian" && return StationaryKernel(LorentzianKernel{F}(k.fwhm))
    return StationaryKernel(k.kernel)
end
device_kernel(k::Sunny.NonstationaryBroadening, F) = NonstationaryKernel(k.kernel)

# Mirrors `Sunny.thermal_factor`, without the assertion on kT
thermal_factor(ω, kT) = iszero(kT) ? (ω > 0 ? one(ω) : zero(ω)) : -1 / expm1(-ω/kT)

# Lorentzian of half width η times the thermal factor, mirroring the kernel of
# `Sunny.broaden_poles` and `Sunny.thermal_factor_at`
struct LorentzianThermal{F}
    η  :: F
    kT :: F
end
function (k::LorentzianThermal{F})(ε, ω) where F
    (; η, kT) = k
    L = (η/F(π)) / ((ω - ε)^2 + η^2)
    th = if iszero(kT)
        ω > 0 ? one(F) : iszero(ω) ? F(0.5) : zero(F)
    elseif iszero(ω)
        2kT * ε / (ε^2 + η^2)
    else
        -1 / expm1(-ω/kT)
    end
    return L * th
end

# The weight W of each pole for `broaden_peaks!`, which includes the thermal
# factor if `thermal`, and the norm of W as a value of the measure
@kernel function peak_weights!(W, norms, @Const(disp), @Const(R), kT, thermal)
    n, iq = @index(Global, NTuple)
    f = thermal ? thermal_factor(disp[n, iq], kT) : one(eltype(W))
    s = zero(eltype(W))
    for k in axes(R, 1)
        w = f * R[k, n, iq]
        W[k, n, iq] = w
        s += w^2
    end
    norms[n, iq] = sqrt(s)
end

# Accumulates each pole of weight W at wavevector iq0+iq, broadened by `kernel`,
# into data[k, iω, d0+iq]. A thread owns one output, and sums over the poles in
# the order of `Sunny.broaden_peaks!`, skipping those below its cutoff.
@kernel function broaden_peaks!(data, @Const(disp), @Const(W), @Const(norms), cutoff, @Const(energies), kernel, iq0, d0)
    iω, iq = @index(Global, NTuple)
    ω = energies[iω]
    for k in axes(data, 1)
        data[k, iω, d0+iq] = 0
    end
    for n in axes(disp, 1)
        norms[n, iq0+iq] < cutoff && continue
        c = kernel(disp[n, iq0+iq], ω)
        for k in axes(data, 1)
            data[k, iω, d0+iq] += c * W[k, n, iq0+iq]
        end
    end
end

# Sums of the poles of wavevector iq0+iq at each node z_j of
# `Sunny.RetardedBroadening`, split into the real and imaginary parts of 1/(ε -
# z_j), stored as χ[j, k, iq] for real component k of the measure
@kernel function retarded_susceptibility!(χr, χi, @Const(disp), @Const(R), @Const(zs), iq0)
    j, iq = @index(Global, NTuple)
    for k in axes(R, 1)
        χr[j, k, iq] = 0
        χi[j, k, iq] = 0
    end
    z = zs[j]
    for n in axes(disp, 1)
        # 1/d without the scaling of complex division, as |d| ≥ Im z > 0
        d = disp[n, iq0+iq] - z
        g = conj(d) / abs2(d)
        for k in axes(R, 1)
            χr[j, k, iq] += real(g) * R[k, n, iq0+iq]
            χi[j, k, iq] += imag(g) * R[k, n, iq0+iq]
        end
    end
end

# Reorders y[i, k, iq] into data[k, i, d0+iq]
@kernel function scatter_components!(data, @Const(y), d0)
    i, iq = @index(Global, NTuple)
    for k in axes(data, 1)
        data[k, i, d0+iq] = y[i, k, iq]
    end
end

# The broadening of `Sunny.broaden_poles` for poles disp[n, iq] and residues
# R[k, n, iq] by real component k of the measure, prepared for application to
# batches of wavevectors by `broaden_batch!`. The three cases are those of the
# host.
function broaden_poles_plan(disp, R; energies, η, kernel, kT)
    η >= 0 || error("Require η ≥ 0")
    kT >= 0 || error("Require kT ≥ 0")
    iszero(η) && isnothing(kernel) && error("Specify a `kernel`, a broadening `η`, or both.")
    issorted(energies) || error("Energies must be sorted")
    F = eltype(R)
    if iszero(η)
        return PeakBroadeningDevice(disp, R, energies, device_kernel(kernel, F); kT, thermal=true)
    elseif isnothing(kernel)
        return PeakBroadeningDevice(disp, R, energies, LorentzianThermal{F}(η, kT); kT, thermal=false)
    else
        return RetardedBroadeningDevice(disp, R, energies; η, kernel, kT)
    end
end

# The poles and weights of `Sunny.broaden_peaks!` on a device. The weights
# W[k, n, iq] include the thermal factor if the `kernel` does not, and poles
# whose weight has a norm below `cutoff` are skipped.
struct PeakBroadeningDevice{F, A2, A3, VE, Kern}
    disp     :: A2
    W        :: A3
    norms    :: A2
    cutoff   :: F
    energies :: VE
    kernel   :: Kern
end

function PeakBroadeningDevice(disp, R, energies, kernel; kT, thermal)
    backend = KA.get_backend(R)
    (K, n, Nq) = size(R)
    F = eltype(R)
    W = KA.allocate(backend, F, (K, n, Nq))
    norms = KA.allocate(backend, F, (n, Nq))
    Nq > 0 && peak_weights!(backend)(W, norms, disp, R, F(kT), thermal; ndrange=(n, Nq))
    KA.synchronize(backend)
    cutoff = F(Nq > 0 ? 1e-12 * Sunny.Statistics.quantile(vec(Array(norms)), 0.95) : 0.0)
    energies = to_backend(backend, collect(F, energies))
    return PeakBroadeningDevice{F, typeof(disp), typeof(W), typeof(energies), typeof(kernel)}(disp, W, norms, cutoff,
                                                                                             energies, kernel)
end

# Mirrors `Sunny.RetardedBroadening` on a device, with the poles and residues
# to which it applies. The weights are split into real and imaginary parts
# divided by π, in the banded form of `banded_quadrature`.
struct RetardedBroadeningDevice{F, A2, A3, VZ, M}
    disp   :: A2
    R      :: A3
    Nω     :: Int
    zs     :: VZ
    Wr     :: M
    Wi     :: M
    blocks :: Vector{Tuple{UnitRange{Int}, UnitRange{Int}}}
end

function RetardedBroadeningDevice(disp, R, energies; η, kernel, kT)
    backend = KA.get_backend(R)
    F = eltype(R)
    (; zs, W, ranges) = Sunny.RetardedBroadening(energies; η, kernel, kT, order=16, cutoff=1e-16)
    (zs, W, blocks) = banded_quadrature(zs, W, ranges)
    zs = to_backend(backend, F, zs)
    Wr = to_backend(backend, F, real(W) / π)
    return RetardedBroadeningDevice{F, typeof(disp), typeof(R), typeof(zs), typeof(Wr)}(
        disp, R, length(energies), zs, Wr, to_backend(backend, F, imag(W) / π), blocks)
end

num_components(plan::PeakBroadeningDevice) = size(plan.W, 1)
num_components(plan::RetardedBroadeningDevice) = size(plan.R, 1)
num_energies(plan::PeakBroadeningDevice) = length(plan.energies)
num_energies(plan::RetardedBroadeningDevice) = plan.Nω

# Bytes of device memory per wavevector for the output of `broaden_batch!` and
# its scratch
bytes_per_q(plan::PeakBroadeningDevice{F}) where F = sizeof(F) * num_components(plan) * num_energies(plan)
bytes_per_q(plan::RetardedBroadeningDevice{F}) where F = sizeof(F) * num_components(plan) * (2plan.Nω + 2length(plan.zs))

# The quadrature of `Sunny.RetardedBroadening` with its weights W[i, j] in a
# banded form. Nodes that contribute to no energy are dropped, the others are
# sorted by the midpoints of their `ranges`, and the weights that
# `Sunny.broaden_poles` neglects are set to zero. Each block of energies then
# draws on a contiguous window of nodes.
function banded_quadrature(zs, W, ranges; block=32)
    js = filter(j -> !isempty(ranges[j]), eachindex(ranges))
    sort!(js; by=j -> first(ranges[j]) + last(ranges[j]))
    W′ = zeros(ComplexF64, size(W, 1), length(js))
    for (j′, j) in enumerate(js), i in ranges[j]
        W′[i, j′] = W[i, j]
    end
    blocks = Tuple{UnitRange{Int}, UnitRange{Int}}[]
    for i1 in 1:block:size(W, 1)
        is = i1:min(i1+block-1, size(W, 1))
        hits = findall(j -> !isempty(intersect(ranges[j], is)), js)
        isempty(hits) || push!(blocks, (is, first(hits):last(hits)))
    end
    return (zs[js], W′, blocks)
end

# Broadens the wavevectors iq0+1:iq0+nb into data[k, iω, d0+1:d0+nb], as
# `Sunny.broaden_peaks!` does
function broaden_batch!(data, plan::PeakBroadeningDevice, iq0, nb, d0, _)
    (; disp, W, norms, cutoff, energies, kernel) = plan
    backend = KA.get_backend(data)
    broaden_peaks!(backend)(data, disp, W, norms, cutoff, energies, kernel, iq0, d0; ndrange=(length(energies), nb))
    return
end

# The same for the retarded broadening, with `scratch` for at least nb
# wavevectors. The quadrature applies to the sums at the nodes, y = Wr χi + Wi
# χr, as one matrix product over all components and wavevectors.
function broaden_batch!(data, plan::RetardedBroadeningDevice, iq0, nb, d0, scratch)
    (; disp, R, Nω, zs, Wr, Wi, blocks) = plan
    backend = KA.get_backend(data)
    K = num_components(plan)
    Nz = length(zs)
    (χr, χi, y) = scratch
    fill!(y, 0)
    if Nz > 0
        retarded_susceptibility!(backend)(χr, χi, disp, R, zs, iq0; ndrange=(Nz, nb))
        m = K * nb
        (y2, χr2, χi2) = (reshape(y, Nω, :), reshape(χr, Nz, :), reshape(χi, Nz, :))
        for (is, js) in blocks
            yb = view(y2, is, 1:m)
            mul!(yb, view(Wr, is, js), view(χi2, js, 1:m))
            mul!(yb, view(Wi, is, js), view(χr2, js, 1:m), true, true)
        end
    end
    scatter_components!(backend)(data, y, d0; ndrange=(Nω, nb))
    return
end

broadening_scratch(::PeakBroadeningDevice, nb) = nothing
function broadening_scratch(plan::RetardedBroadeningDevice{F}, nb) where F
    backend = KA.get_backend(plan.R)
    (K, Nz, Nω) = (num_components(plan), length(plan.zs), plan.Nω)
    return (KA.allocate(backend, F, (Nz, K, nb)), KA.allocate(backend, F, (Nz, K, nb)), KA.allocate(backend, F, (Nω, K, nb)))
end
