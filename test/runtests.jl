using TestItemRunner

# TOML is in the test dependencies only as a workaround for
# https://github.com/JuliaGPU/AMDGPU.jl/issues/1222, where ROCm_Runtime loads
# TOML without declaring it, so `Pkg.test` fails. Remove TOML from
# test/Project.toml once that is fixed.

# Testitems tagged :gpu run only if SUNNY_TEST_GPU names their backend, e.g.
# "CUDA" for testitems also tagged :cuda, or "Metal" for those tagged :metal
const test_gpu = Symbol(lowercase(get(ENV, "SUNNY_TEST_GPU", "")))

@run_package_tests filter=test -> !(:skip in test.tags) && (!(:gpu in test.tags) || test_gpu in test.tags)
