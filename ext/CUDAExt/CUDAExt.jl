module CUDAExt

using CUDA
using Adapt
using Sunny

include("Symmetry/Crystal.jl")
include("Measurements/IntensitiesTypes.jl")
include("Measurements/Broadening.jl")
end
