struct BroadeningDevice{F} <: Sunny.AbstractBroadening
    kernel :: F   # Function mapping x = (ω - ϵ) to an intensity scaling factor
    fwhm :: Float64
end

BroadeningDevice(host::Sunny.Broadening) = BroadeningDevice(host.kernel, host.fwhm)

function Adapt.adapt_structure(to, data::BroadeningDevice)
    kernel = Adapt.adapt_structure(to, data.kernel)
    fwhm = Adapt.adapt_structure(to, data.fwhm)
    BroadeningDevice(kernel, fwhm)
end

function (b::BroadeningDevice)(ϵ, ω)
    b.kernel(ω - ϵ)
end

function _broaden(data, bands_data, disp, energies, kernel)
    iq = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if iq > size(disp, 2)
        return
    end
    for (ib, b) in enumerate(view(disp, :, iq))
        #norm(bands.data[ib, iq]) < cutoff && continue
        for (iω, ω) in enumerate(energies)
            data[iω, iq] += kernel(b, ω) * bands_data[ib, iq]
        end
    end
    return
end

function broaden!(data::CuArray{Ret}, bands::BandIntensitiesDevice{Ret}; energies, kernel) where Ret
    energies_d = CuArray(collect(Float64, energies))
    #issorted(energies) || error("energies must be sorted")

    nω = length(energies)
    nq = size(bands.qpts.qs,1)
    (nω, nq...) == size(data) || error("Argument data must have size ($nω×$(sizestr(bands.qpts)))")

    #asdf = norm.(vec(bands.data))
    #cutoff = 1e-12 * Statistics.quantile(asdf, 0.95)

    kernel_d = BroadeningDevice(kernel)
    gpu_kernel = CUDA.@cuda launch=false _broaden(data, bands.data, bands.disp, energies_d, kernel_d)
    config = launch_configuration(gpu_kernel.fun)
    threads = Base.min(nq, config.threads)
    blocks = cld(nq, threads)
    gpu_kernel(data, bands.data, bands.disp, energies_d, kernel_d; threads=threads, blocks=blocks)

    return data
end

function Sunny.broaden(bands::BandIntensitiesDevice; energies, kernel)
    data = CUDA.zeros(eltype(bands.data), length(energies), size(bands.qpts.qs)...)
    broaden!(data, bands; energies, kernel)
    return IntensitiesDevice(bands.crystal, bands.qpts, collect(Float64, energies), data)
end
