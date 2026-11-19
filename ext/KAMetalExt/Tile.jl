# Cooperative primitives for a tile of W lanes of a SIMD group, where W is 4, 8,
# 16 or 32, after the `*_on_cg` functions of EigenG-Batched on CUDA's
# cooperative groups. A SIMD group of 32 threads holds 32/W tiles, each acting
# on its own matrix, and the lane of a thread is its 1-based index in its tile.
#
# Metal leaves the result of a shuffle undefined if its source lane is
# inactive. Every primitive here, except `tile_sync`, reads only the lanes of
# the caller's tile, so the tiles of a SIMD group may follow different control
# flow (e.g., different iteration counts), as long as all lanes of a tile stay
# together. `tile_sync` is a SIMD-group barrier, and so must be reached by
# every thread of the SIMD group.

using Metal: simd_shuffle, simd_shuffle_xor, simdgroup_barrier, thread_index_in_simdgroup,
             MemoryFlagDevice, MemoryFlagThreadGroup

# The 1-based lane of the calling thread in its tile
@inline tile_lane(::Val{W}) where W = ((Int(thread_index_in_simdgroup()) - 1) & (W - 1)) + 1

# The 1-based index in the SIMD group of lane 1 of the caller's tile
@inline tile_first(::Val{W}) where W = Int(thread_index_in_simdgroup()) - tile_lane(Val(W)) + 1

# Shuffles of the types that Metal supports natively, and of complex numbers as
# pairs of their parts
@inline shfl_xor(x::Union{Float32, Int32}, m) = simd_shuffle_xor(x, m)
@inline shfl_xor(x::Complex{Float32}, m) = complex(simd_shuffle_xor(real(x), m), simd_shuffle_xor(imag(x), m))
@inline shfl(x::Union{Float32, Int32}, lane) = simd_shuffle(x, lane)
@inline shfl(x::Complex{Float32}, lane) = complex(simd_shuffle(real(x), lane), simd_shuffle(imag(x), lane))

# Applies the associative `op` across the tile by a butterfly of log₂(W)
# shuffles, leaving the result on every lane. The xor masks stay below W, and
# so within the tile.
@inline function tile_reduce(op, x, ::Val{W}) where W
    m = W ÷ 2
    while m >= 1
        x = op(x, shfl_xor(x, m))
        m ÷= 2
    end
    return x
end

# The sum of `x` over the tile, on every lane
@inline tile_sum(x, ::Val{W}) where W = tile_reduce(+, x, Val(W))

# The maximum of `x` over the tile, on every lane
@inline tile_max(x, ::Val{W}) where W = tile_reduce(max, x, Val(W))

# The sums of `x` and of `y` over the tile, on every lane, in log₂(W) + 1
# shuffles instead of 2 log₂(W). The first exchange leaves partial sums of x on
# lanes of even 0-based index and of y on odd ones, which the remaining
# exchanges, with even masks, preserve. A final exchange with the neighbor
# completes the pair. This is the device of `red2_on_cg` of EigenG.
@inline function tile_sum2(x, y, ::Val{W}) where W
    odd = isodd(Int(thread_index_in_simdgroup()) - 1)
    a = ifelse(odd, y, x)
    b = ifelse(odd, x, y)
    a += shfl_xor(b, 1)
    m = 2
    while m < W
        a += shfl_xor(a, m)
        m *= 2
    end
    other = shfl_xor(a, 1)
    return odd ? (other, a) : (a, other)
end

# The value of `x` on lane `root` of the caller's tile, on every lane. The root
# must be the same on every lane of the tile.
@inline tile_bcast(x, root, ::Val{W}) where W = shfl(x, tile_first(Val(W)) + root - 1)

# Orders the memory operations of the SIMD group, device and threadgroup, so
# that writes of one lane before it are visible to reads of another after it.
# Every thread of the SIMD group must reach it.
@inline tile_sync() = simdgroup_barrier(MemoryFlagDevice | MemoryFlagThreadGroup)
