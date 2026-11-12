using TestItemRunner

# Testitems tagged :gpu run only if SUNNY_TEST_GPU names a backend, e.g. "CUDA"
const test_gpu = !isempty(get(ENV, "SUNNY_TEST_GPU", ""))

@run_package_tests filter=test -> !(:skip in test.tags) && (test_gpu || !(:gpu in test.tags))
