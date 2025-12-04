struct BandIntensitiesDevice{T, Q <: Sunny.AbstractQPoints, D} <: Sunny.AbstractIntensities
    # Original chemical cell
    crystal :: Crystal
    # Wavevectors in RLU
    qpts :: Q
    # Dispersion for each band
    disp :: CUDA.CuArray{Float64, D} # (nbands × nq...)
    # Intensity data as Dirac-magnitudes
    data :: CUDA.CuArray{T, D} # (nbands × nq...)
end

struct IntensitiesDevice{T, Q <: Sunny.AbstractQPoints, D} <: Sunny.AbstractIntensities
    # Original chemical cell
    crystal :: Crystal
    # Wavevectors in RLU
    qpts :: Q
    # Regular grid of energies
    energies :: Vector{Float64}
    # Intensity data as continuum density
    data :: CUDA.CuArray{T, D} # (nω × nq...)
end

Sunny.Intensities(device::IntensitiesDevice) = Sunny.Intensities(device.crystal, device.qpts, device.energies, Array(device.data))
