# Metal-specific hooks for KAExt. Metal Performance Shaders lacks a batched
# Hermitian eigensolver, so `Sunny.vendor_bogoliubov!` is a port of
# EigenG-Batched to Metal, in Float32 for n ≤ 256. Beyond, KAExt falls back to
# its :host or :jacobi eigensolvers.
module KAMetalExt

using Metal: Metal
using Sunny: Sunny

include("Tile.jl")
include("HermitianEigen.jl")
include("Bogoliubov.jl")

# Metal GPUs lack Float64
Sunny.device_precision(::Metal.MetalBackend) = Float32

# Memory is unified with the host, so this is the remaining working set that
# Metal recommends for the device, not a hard limit
function Sunny.device_free_memory(::Metal.MetalBackend)
    dev = Metal.device()
    return max(Int(dev.recommendedMaxWorkingSetSize) - Int(dev.currentAllocatedSize), 0)
end

end
