# The CPU backend of KernelAbstractions compiles kernels through GPUCompiler,
# which loading Sunny invalidates. Running the kernels here caches much of that
# compilation in the package image, and roughly halves the latency of the first
# call. A failure, e.g. of the OpenCL runtime, must not prevent loading.
@setup_workload begin
    cryst = Sunny.Crystal(Sunny.lattice_vectors(1, 1, 2, 90, 90, 90), [[0, 0, 0]])
    sys = Sunny.System(cryst, [1 => Sunny.Moment(s=1, g=2)], :dipole)
    Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, [1, 0, 0]))
    Sunny.polarize_spins!(sys, [0, 0, 1])
    energies = range(0, 5, 3)
    @compile_workload begin
        try
            swt_d = Sunny.to_device(Sunny.SpinWaveTheory(sys; measure=Sunny.ssf_perp(sys)))
            for (η, kernel) in ((0, Sunny.gaussian(fwhm=1.0)), (0.1, nothing), (0.1, Sunny.gaussian(fwhm=1.0)))
                Sunny.intensities(swt_d, [[0.1, 0, 0]]; energies, η, kernel)
            end
        catch err
            @debug "KAExt precompile workload failed" exception=err
        end
    end
end
