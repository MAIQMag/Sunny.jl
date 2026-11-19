# Metal-specific hooks for KAExt. Metal Performance Shaders lacks a batched
# Hermitian eigensolver, so there is no `Sunny.vendor_bogoliubov!` method, and
# KAExt falls back to its :jacobi or :host eigensolvers.
module KAMetalExt

using Metal: Metal
using Sunny: Sunny

include("Tile.jl")
include("HermitianEigen.jl")

# Metal GPUs lack Float64
Sunny.device_precision(::Metal.MetalBackend) = Float32

# Memory is unified with the host, so this is the remaining working set that
# Metal recommends for the device, not a hard limit
function Sunny.device_free_memory(::Metal.MetalBackend)
    dev = Metal.device()
    return max(Int(dev.recommendedMaxWorkingSetSize) - Int(dev.currentAllocatedSize), 0)
end

end
