# Like `Sunny.Intensities`, but with `data` on a device, as returned by
# `intensities(swt_d, qpts; output=:device)`
struct IntensitiesDevice{T, Q <: Sunny.AbstractQPoints, D <: AbstractArray{T}}
    crystal  :: Sunny.Crystal
    qpts     :: Q
    energies :: Vector{Float64}
    data     :: D  # (nω × nq...)
end

Sunny.Intensities(res::IntensitiesDevice) = Sunny.Intensities(res.crystal, res.qpts, res.energies, Array(res.data))

# For `Sunny.powder_average`, the means over shells on the device, so that only
# they are copied to the host, in the Float64 element type of host intensities
function Sunny.shell_means(res::IntensitiesDevice, n, m)
    nω = length(res.energies)
    means = Sunny.Statistics.mean(reshape(res.data, nω, n, m); dims=2)
    return Array{precision_type(Float64, eltype(res.data))}(dropdims(Array(means); dims=2))
end

function Base.show(io::IO, res::IntensitiesDevice)
    sz = join([length(res.energies), Sunny.sizestr(res.qpts)], "×")
    print(io, "IntensitiesDevice{$(eltype(res.data))} ($sz elements)")
end
