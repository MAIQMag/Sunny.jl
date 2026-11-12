# Like `Sunny.Intensities`, but with `data` on a device, as returned by
# `intensities(swt_d, qpts; output=:device)`
struct IntensitiesDevice{T, Q <: Sunny.AbstractQPoints, D <: AbstractArray{T}}
    crystal  :: Sunny.Crystal
    qpts     :: Q
    energies :: Vector{Float64}
    data     :: D  # (nω × nq...)
end

Sunny.Intensities(res::IntensitiesDevice) = Sunny.Intensities(res.crystal, res.qpts, res.energies, Array(res.data))

function Base.show(io::IO, res::IntensitiesDevice)
    sz = join([length(res.energies), Sunny.sizestr(res.qpts)], "×")
    print(io, "IntensitiesDevice{$(eltype(res.data))} ($sz elements)")
end
