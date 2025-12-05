module CUDAExt

using CUDA
using Adapt
using Sunny

include("FormFactor.jl")
include("Symmetry/Crystal.jl")
include("Measurements/IntensitiesTypes.jl")
include("Measurements/Broadening.jl")
include("Measurements/MeasureSpec.jl")
include("System/Types.jl")
include("System/System.jl")
include("SpinWaveTheory/SpinWaveTheory.jl")
include("SpinWaveTheory/HamiltonianDipole.jl")
include("SpinWaveTheory/DispersionAndIntensities.jl")
end
