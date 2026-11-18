# Spin wave calculations of the √3×√3 kagome antiferromagnet, as in the
# tutorial examples/spinw_tutorials/SW08_sqrt3_kagome_AFM.jl, but without
# graphics, for systems without a display. Instead of SpinWaveTheorySpiral, the
# 120° order is repeated into a 3×3×1 magnetic supercell of 27 sites, and
# SpinWaveTheory calculates its spectrum. The path and powder-average
# intensities are calculated on the host and on the first functional GPU
# backend (CUDA, AMDGPU, Metal, else the KernelAbstractions CPU backend), and the
# speedup and relative difference are printed. Run with threads for the host
# calculations, e.g.,
#
#     julia --project=<env> -t 16 examples/extra/bench_kagome_sqrt3_device.jl
#
# where <env> has Sunny, KernelAbstractions, and the GPU package, as well as MKL
# (x86_64) or AppleAccelerate (macOS) for `load_blas_for_threading`.

using Sunny
load_blas_for_threading()

# ### Spin system

# The chemical cell of a kagome lattice with spacegroup 147 (P-3), and
# nearest neighbor antiferromagnetic exchange.

units = Units(:meV, :angstrom)
latvecs = lattice_vectors(6, 6, 40, 90, 90, 120)
cryst = Crystal(latvecs, [[1/2, 0, 0]], 147)
sys = System(cryst, [1 => Moment(s=1, g=2)], :dipole)
J = 1.0
set_exchange!(sys, J, Bond(2, 3, [0, 0, 0]))

# The energy minimizing magnetic structure, for which nearest-neighbor spins
# are at 120° angles, has propagation wavevector k = (-1/3, -1/3, 0). Repeating
# it as a spiral into a 3×3×1 supercell gives a commensurate magnetic cell of
# 27 sites, suitable for SpinWaveTheory.

set_dipole!(sys, [cos(0), sin(0), 0], (1, 1, 1, 1))
set_dipole!(sys, [cos(0), sin(0), 0], (1, 1, 1, 2))
set_dipole!(sys, [cos(2π/3), sin(2π/3), 0], (1, 1, 1, 3))
k = [-1/3, -1/3, 0]
axis = [0, 0, 1]
sys_enlarged = repeat_periodically_as_spiral(sys, (3, 3, 1); k, axis)

# Each site participates in 4 bonds with energy ``J\cos(2π/3)``. Factor of 1/2
# avoids double counting.

@assert energy_per_site(sys_enlarged) ≈ (4/2)*J*cos(2π/3)

# ### Spin wave theory

# The supercell method gives rise to ghost modes in the dispersion that have
# zero intensity, but otherwise agrees with the spiral method of the tutorial.

swt = SpinWaveTheory(sys_enlarged; measure=ssf_perp(sys_enlarged))

# A path through ``𝐪``-space, and a powder average over 200 shells of 200
# points each. The intensities are dominated by a flat band at zero energy
# transfer.

qs = [[-1/2, 0, 0], [0, 0, 0], [1/2, 1/2, 0]]
path = q_space_path(cryst, qs, 400)
radii = range(0, 2.5, 200) # (1/Å)
energies = range(0, 2.5, 200)
kernel = gaussian(fwhm=0.05)

# ### Calculations on a GPU

using KernelAbstractions
try @eval using CUDA catch end
try @eval using AMDGPU catch end
try @eval using Metal catch end
backend = if @isdefined(CUDA) && CUDA.functional()
    CUDABackend()
elseif @isdefined(AMDGPU) && AMDGPU.functional()
    ROCBackend()
elseif @isdefined(Metal) && Metal.functional()
    MetalBackend()
else
    KernelAbstractions.CPU()
end
swt_d = to_device(swt; backend)
const KAExt = Base.get_extension(Sunny, :KAExt)

# Compare the times of the calculations, each the best of three runs after a
# warm-up. With `output=:device`, the average over each shell runs on the
# device, and only one spectrum per shell is copied to the host.

besttime(f) = (f(); minimum(@elapsed(f()) for _ in 1:3))
function powder_host()
    powder_average(cryst, radii, 200) do qs
        intensities(swt, qs; energies, kernel, threaded=true)
    end
end
function powder_device()
    powder_average(cryst, radii, 200; batch_size=50) do qs
        intensities(swt_d, qs; energies, kernel)
    end
end
function powder_device_avg()
    powder_average(cryst, radii, 200; batch_size=50) do qs
        intensities(swt_d, qs; energies, kernel, output=:device)
    end
end
relerr(a, b) = sqrt(sum(abs2, a - b) / sum(abs2, b))

# The intensities of the flat band diverge as its energy vanishes, so near zero
# energy they are set by the `regularization` of the Hamiltonian. In Float32,
# e.g., on Metal, `to_device` raises the regularization from 1e-8 to about
# 7e-7, which changes these intensities by order one. Away from zero energy,
# the intensities agree to the precision of the device. The relative difference
# is also reported for energies above 0.2 meV, four times the FWHM of the
# kernel.

high = energies .> 0.2

println(Sunny.nsites(swt.sys), " sites (n = ", 2Sunny.nbands(swt), ") on ", backend, " in ", KAExt.precision_of(swt_d),
        ", eigensolver ", repr(swt_d.eigensolver), ", ", Threads.nthreads(), " host threads")
for (name, f_host, f_device) in (
        ("Path", () -> intensities(swt, path; energies, kernel, threaded=true), () -> intensities(swt_d, path; energies, kernel)),
        ("Powder average", powder_host, powder_device),
        ("  on device", powder_host, powder_device_avg))
    res_h = f_host()
    res_dev = f_device()
    t_host = besttime(f_host)
    t_device = besttime(f_device)
    println(rpad(name, 16), "host ", round(t_host; sigdigits=3), " s, device ", round(t_device; sigdigits=3),
            " s, speedup ", round(t_host / t_device; digits=1), "×, relative difference ",
            round(relerr(res_dev.data, res_h.data); sigdigits=2), " (above 0.2 meV ",
            round(relerr(res_dev.data[high, :], res_h.data[high, :]); sigdigits=2), ")")
end
