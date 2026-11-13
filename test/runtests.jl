using TestItemRunner

# Testitems tagged :gpu run only if SUNNY_TEST_GPU names their backend, e.g.
# "CUDA" for testitems also tagged :cuda, or "Metal" for those tagged :metal
const test_gpu = Symbol(lowercase(get(ENV, "SUNNY_TEST_GPU", "")))

@run_package_tests filter=test -> !(:skip in test.tags) && (!(:gpu in test.tags) || test_gpu in test.tags)
