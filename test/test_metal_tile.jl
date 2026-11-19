@testitem "Metal tile primitives" tags=[:gpu, :metal] begin
    import Metal, KernelAbstractions
    using Metal: @metal, thread_position_in_grid_1d, MtlArray
    using Random
    const MT = Base.get_extension(Sunny, :KAMetalExt)

    # Each primitive, applied by every lane of a grid of 4 threadgroups of 256
    # threads, so 1024/W tiles, against the same reduction over each block of W
    # consecutive threads on the host
    const N = 1024
    tiles(x, W) = reshape(x, W, :)
    rng = Xoshiro(0)

    function sum_kernel!(out, x, ::Val{W}) where W
        i = thread_position_in_grid_1d()
        out[i] = MT.tile_sum(x[i], Val(W))
        return
    end
    function max_kernel!(out, x, ::Val{W}) where W
        i = thread_position_in_grid_1d()
        out[i] = MT.tile_max(x[i], Val(W))
        return
    end
    function sum2_kernel!(out1, out2, x, y, ::Val{W}) where W
        i = thread_position_in_grid_1d()
        (out1[i], out2[i]) = MT.tile_sum2(x[i], y[i], Val(W))
        return
    end
    # A root that differs between tiles, but not within one
    function bcast_kernel!(out, lanes, x, ::Val{W}) where W
        i = thread_position_in_grid_1d()
        root = (i - 1) ÷ W % W + 1
        out[i] = MT.tile_bcast(x[i], root, Val(W))
        lanes[i] = MT.tile_lane(Val(W))
        return
    end
    # Tiles of one SIMD group take different numbers of iterations
    function divergent_kernel!(out, x, ::Val{W}) where W
        i = thread_position_in_grid_1d()
        trips = (i - 1) ÷ W % 5 + 1
        acc = zero(eltype(out))
        for k in 1:trips
            acc += MT.tile_sum(convert(eltype(x), k) * x[i], Val(W))
        end
        out[i] = acc
        return
    end
    # Lane 1 writes device memory that the other lanes read after the barrier
    function sync_kernel!(out, buf, x, ::Val{W}) where W
        i = thread_position_in_grid_1d()
        t = (i - 1) ÷ W + 1
        MT.tile_lane(Val(W)) == 1 && (buf[t] = x[i])
        MT.tile_sync()
        out[i] = buf[t]
        return
    end

    launch(f, args...) = (@metal threads=256 groups=N÷256 f(args...); Metal.synchronize())

    for W in (4, 8, 16, 32)
        for T in (Int32, Float32, ComplexF32)
            x = T == Int32 ? rand(rng, Int32(-100):Int32(100), N) : randn(rng, T, N)
            out = MtlArray(zeros(T, N))
            launch(sum_kernel!, out, MtlArray(x), Val(W))
            expected = repeat(sum(tiles(x, W); dims=1); inner=(W, 1))
            T == Int32 ? (@test Array(out) == vec(expected)) : (@test Array(out) ≈ vec(expected) rtol=1e-5)

            launch(divergent_kernel!, out, MtlArray(x), Val(W))
            trips = (0:N÷W-1) .% 5 .+ 1
            expected = repeat(sum(tiles(x, W); dims=1) .* (trips .* (trips .+ 1) .÷ 2)'; inner=(W, 1))
            T == Int32 ? (@test Array(out) == vec(expected)) : (@test Array(out) ≈ vec(expected) rtol=1e-5)

            lanes = MtlArray(zeros(Int, N))
            launch(bcast_kernel!, out, lanes, MtlArray(x), Val(W))
            roots = (0:N÷W-1) .% W .+ 1
            @test Array(out) == vec(repeat(permutedims([tiles(x, W)[roots[t], t] for t in 1:N÷W]); inner=(W, 1)))
            @test Array(lanes) == repeat(1:W, N÷W)

            buf = MtlArray(zeros(T, N÷W))
            launch(sync_kernel!, out, buf, MtlArray(x), Val(W))
            @test Array(out) == vec(repeat(tiles(x, W)[1:1, :]; inner=(W, 1)))
        end

        x = randn(rng, Float32, N)
        out = MtlArray(zeros(Float32, N))
        launch(max_kernel!, out, MtlArray(x), Val(W))
        @test Array(out) == vec(repeat(maximum(tiles(x, W); dims=1); inner=(W, 1)))

        (x, y) = (randn(rng, ComplexF32, N), randn(rng, ComplexF32, N))
        (out1, out2) = (MtlArray(zeros(ComplexF32, N)), MtlArray(zeros(ComplexF32, N)))
        launch(sum2_kernel!, out1, out2, MtlArray(x), MtlArray(y), Val(W))
        @test Array(out1) ≈ vec(repeat(sum(tiles(x, W); dims=1); inner=(W, 1))) rtol=1e-5
        @test Array(out2) ≈ vec(repeat(sum(tiles(y, W); dims=1); inner=(W, 1))) rtol=1e-5
    end
end
