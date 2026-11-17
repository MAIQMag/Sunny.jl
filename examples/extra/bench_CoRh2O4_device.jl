# Spin wave calculations of CoRh₂O₄, as in the tutorial
# examples/01_LSWT_CoRh2O4.jl, but without graphics, for systems without a
# display. The path and powder-average intensities are calculated on the host
# and on the first functional GPU backend (CUDA, AMDGPU, Metal, else the
# KernelAbstractions CPU backend), and the speedup and relative difference are
# printed. Run with threads for the host calculations, e.g.,
#
#     julia --project=<env> -t 16 examples/extra/bench_CoRh2O4_device.jl
#
# where <env> has Sunny, KernelAbstractions, and the GPU package, as well as MKL
# (x86_64) or AppleAccelerate (macOS) for `load_blas_for_threading`.

using Sunny
load_blas_for_threading()

# ### Units system

# The [`Units`](@ref Units) object selects reference energy and length scales,
# and uses these to provide physical constants. For example, `units.K` returns
# one kelvin as 0.086 meV, where the Boltzmann constant is implicit.

units = Units(:meV, :angstrom)
;#hide

# ### Crystal cell
#
# A crystallographic cell may be loaded from a `.cif` file, or specified from
# atom positions and types.
#
# Start by defining the shape of the conventional chemical cell. CoRh₂O₄ has
# cubic spacegroup 227 (Fd-3m). Its lattice constants are 8.5 Å and the cell
# angles are 90°. With this information, [`lattice_vectors`](@ref) constructs a
# 3×3 matrix `latvecs`. Columns of `latvecs` define the lattice vectors ``(𝐚_1,
# 𝐚_2, 𝐚_3)`` in the global Cartesian coordinate system. Conversely, columns
# of `inv(latvecs)` define the global Cartesian axes ``(\hat{x}, \hat{y},
# \hat{z})`` in components of the lattice vectors.

a = 8.5031 # (Å)
latvecs = lattice_vectors(a, a, a, 90, 90, 90)

# Construct a [`Crystal`](@ref) cell from spacegroup 227 in the ITA standard
# setting. Cobalt atoms belong to Wyckoff 8a, which is the diamond cubic
# lattice.

positions = [[1/8, 1/8, 1/8]]
cryst = Crystal(latvecs, positions, 227; types=["Co"])

# ### Spin system

# A [`System`](@ref) will define the spin model. Each cobalt atom carries
# quantum spin ``s = 3/2``, with a ``g``-factor of 2. Specify this
# [`Moment`](@ref) data for cobalt atom 1. By symmetry, the same moment data
# also applies to cobalt atoms 2, 3, ... 7. The option `:dipole` indicates a
# traditional model type, for which quantum spin is modeled as a dipole
# expectation value.

sys = System(cryst, [1 => Moment(s=3/2, g=2)], :dipole)

# Ge et al. demonstrated that inelastic neutron scattering data for CoRh₂O₄ is
# well modeled by antiferromagnetic nearest neighbor exchange, `J = 0.63` meV.
# Call [`set_exchange!`](@ref) with the bond that connects atom 1 to atom 3, and
# has zero displacement between chemical cells. Consistent with the symmetries
# of spacegroup 227, this interaction will be propagated to all other
# nearest-neighbor bonds.

J = +0.63 # (meV)
set_exchange!(sys, J, Bond(2, 3, [0, 0, 0]))

# ### Optimizing spins

# To search for the ground state, call [`randomize_spins!`](@ref) and
# [`minimize_energy!`](@ref) in sequence. For this problem, optimization
# converges rapidly to the expected Néel order.

randomize_spins!(sys)
minimize_energy!(sys)

# The diamond lattice is bipartite, allowing each spin to perfectly anti-align
# with its 4 nearest-neighbors. Each of these 4 bonds contribute ``-J s^2`` to
# the total energy. Two sites participate in each bond, so the energy per site
# is ``-2 J s^2``. Check this by calling [`energy_per_site`](@ref).

@assert energy_per_site(sys) ≈ -2J*(3/2)^2

# ### Reshaping the magnetic cell

# The most compact magnetic cell for this Néel order is the primitive unit cell.
# Columns of the [`primitive_cell`](@ref) matrix provide the primitive lattice
# vectors as multiples of the conventional cubic lattice vectors.

shape = primitive_cell(cryst)

# Reduce the magnetic cell size using [`reshape_supercell`](@ref). Verify that
# the energy per site is unchanged after the reshaping the supercell.

sys_prim = reshape_supercell(sys, shape)
@assert energy_per_site(sys_prim) ≈ -2J*(3/2)^2

# ### Spin wave theory

# With this primitive cell, we will perform a [`SpinWaveTheory`](@ref)
# calculation of the structure factor ``\mathcal{S}(𝐪,ω)``. The measurement
# [`ssf_perp`](@ref) indicates projection of the spin structure factor
# ``\mathcal{S}(𝐪,ω)`` perpendicular to the direction of momentum transfer, as
# appropriate for unpolarized neutron scattering. The isotropic
# [`FormFactor`](@ref) for Co²⁺ dampens intensities at large ``𝐪``.

formfactors = [1 => FormFactor("Co2")]
measure = ssf_perp(sys_prim; formfactors)
swt = SpinWaveTheory(sys_prim; measure)

# Select [`lorentzian`](@ref) broadening with a full-width at half-maximum
# (FWHM) of 0.8 meV. 

kernel = lorentzian(fwhm=0.8)

# Define a [`q_space_path`](@ref) that connects high-symmetry points in
# reciprocal space. The ``𝐪``-points are given in reciprocal lattice units
# (RLU) for the _original_ cubic cell. For example, `[1/2, 1/2, 0]` denotes the
# sum of the first two reciprocal lattice vectors, ``𝐛_1/2 + 𝐛_2/2``. A total
# of 500 ``𝐪``-points will be sampled along the path.

qs = [[0, 0, 0], [1/2, 0, 0], [1/2, 1/2, 0], [0, 0, 0]]
path = q_space_path(cryst, qs, 500)

# Calculate single-crystal scattering [`intensities`](@ref) along this path, for
# energies between 0 and 6 meV.

energies = range(0, 6, 300)
res = intensities(swt, path; energies, kernel, kT=0, threaded=true)

# Sometimes experimental data is only available as a powder average, i.e., as an
# average over all possible crystal orientations. Use [`powder_average`](@ref)
# to simulate these intensities. Each ``𝐪``-magnitude defines a spherical shell
# in reciprocal space. Consider 200 radii from 0 to 3 inverse angstroms and
# collect 2000 random samples per spherical shell. As configured, this
# calculation completes in about two seconds. Had we used the conventional cubic
# cell, the calculation would be an order of magnitude slower.

radii = range(0, 3, 200) # (1/Å)
res = powder_average(cryst, radii, 2000) do qs
    intensities(swt, qs; energies, kernel, kT=0, threaded=true)
end

# This result can be compared to experimental neutron scattering data
# from Fig. 5 of [Ge et al.](https://doi.org/10.1103/PhysRevB.96.064413)
# ```@raw html
# <img width="95%" src="https://raw.githubusercontent.com/SunnySuite/Sunny.jl/main/docs/src/assets/CoRh2O4_intensity.jpg">
# ```

# ### Calculations on a GPU
#
# Loading [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl)
# enables `to_device`, which copies the spin wave model to a GPU. Load
# the package for your GPU, e.g., CUDA for NVIDIA, AMDGPU for AMD, or Metal for
# Apple silicon. Here we use the first one that is installed and functional,
# and otherwise the CPU backend of KernelAbstractions.

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

# The device model `swt_d` replaces `swt` in calls to [`intensities`](@ref).
# Inside [`powder_average`](@ref), the option `batch_size` passes the points of
# several shells to each call, which gives the GPU more work at once.

res_d = intensities(swt_d, path; energies, kernel)
res_powder_d = powder_average(cryst, radii, 2000; batch_size=50) do qs
    intensities(swt_d, qs; energies, kernel)
end

# The results agree with those on the host to within the precision of the
# device, which is `Float32` for Metal and `Float64` otherwise. Compare the
# times of both calculations, each the best of three runs after a warm-up, to
# the serial host calculations above.

besttime(f) = (f(); minimum(@elapsed(f()) for _ in 1:3))
function powder_host()
    powder_average(cryst, radii, 2000) do qs
        intensities(swt, qs; energies, kernel, kT=0, threaded=true)
    end
end
function powder_device()
    powder_average(cryst, radii, 2000; batch_size=50) do qs
        intensities(swt_d, qs; energies, kernel)
    end
end
relerr(a, b) = sqrt(sum(abs2, a - b) / sum(abs2, b))

res_path = intensities(swt, path; energies, kernel)
res_powder = powder_host()
for (name, f_host, f_device, res_h, res_dev) in (
        ("Path", () -> intensities(swt, path; energies, kernel, kT=0, threaded=true), () -> intensities(swt_d, path; energies, kernel), res_path, res_d),
        ("Powder average", powder_host, powder_device, res_powder, res_powder_d))
    t_host = besttime(f_host)
    t_device = besttime(f_device)
    println(rpad(name, 16), "host ", round(t_host; sigdigits=3), " s, device ", round(t_device; sigdigits=3),
            " s, speedup ", round(t_host / t_device; digits=1), "×, relative difference ",
            round(relerr(res_dev.data, res_h.data); sigdigits=2))
end


# ### What's next?
#
# * For more spin wave calculations of this type, browse the [SpinW tutorials
#   ported to Sunny](@ref "SW01 - FM Heisenberg chain").
# * Spin wave theory neglects thermal fluctuations of the magnetic order. The
#   [next CoRh₂O₄ tutorial](@ref "2. Landau-Lifshitz dynamics of CoRh₂O₄ at
#   finite *T*") demonstrates how to sample spins in thermal equilibrium and
#   measure correlations from the classical spin dynamics.
# * Sunny also offers features that go beyond the dipole approximation of a
#   quantum spin via the theory of SU(_N_) coherent states. This can be
#   especially useful for systems with strong single-ion anisotropy, as
#   demonstrated in the [FeI₂ tutorial](@ref "3. Multi-flavor spin wave
#   simulations of FeI₂").
