module CUDAExt

using Adapt: Adapt
using CUDA: CUDA, @cuda, CuArray, CuDynamicSharedArray,
            CuMatrix, CuPtr, CuVector, StridedCuArray, blockDim, blockIdx,
            launch_configuration, threadIdx
using cuBLAS: unsafe_strided_batch, handle, cublasZtrsmBatched_64, cublasCtrsmBatched_64
using cuSOLVER: dense_handle, CuSolverParameters, cusolverDnZpotrfBatched, cusolverDnCpotrfBatched, cusolverDnXsyevBatched_bufferSize, cusolverDnXsyevBatched
using Sunny: Sunny, Bond, Crystal, SpinWaveTheory, System, global_position, 
             intensities_bands

include("FormFactor.jl")
include("EigenBatched.jl")
include("Symmetry/Crystal.jl")
include("Measurements/IntensitiesTypes.jl")
include("Measurements/Broadening.jl")
include("Measurements/MeasureSpec.jl")
include("Measurements/QPoints.jl")
include("Measurements/RotationalAverages.jl")
include("System/Ewald.jl")
include("System/Types.jl")
include("System/System.jl")
include("SpinWaveTheory/SpinWaveTheory.jl")
include("SpinWaveTheory/HamiltonianCommon.jl")
include("SpinWaveTheory/HamiltonianDipole.jl")
include("SpinWaveTheory/HamiltonianSUN.jl")
include("SpinWaveTheory/DispersionAndIntensities.jl")

end
