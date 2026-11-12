module KAExt

using KernelAbstractions: KernelAbstractions as KA, @Const, @index, @kernel
using LinearAlgebra: dot, mul!, norm
using PrecompileTools: @compile_workload, @setup_workload
using Sunny: Sunny, Vec3

include("Device.jl")
include("SpinWaveTheory/Kernels.jl")
include("SpinWaveTheory/Eigen.jl")
include("Measurements/Broadening.jl")
include("Measurements/IntensitiesDevice.jl")
include("SpinWaveTheory/Intensities.jl")
include("Precompile.jl")

end
