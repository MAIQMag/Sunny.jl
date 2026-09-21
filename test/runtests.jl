using TestItemRunner
using CUDA

# Determine which tags to target
if CUDA.functional()
    @info "CUDA detected. Running full test suite including GPU."
    CUDA.versioninfo()
    @run_package_tests filter=test->!(:skip in test.tags)
else
    @warn "CUDA not available. Running CPU tests only."
    @run_package_tests filter=test -> !(:skip in test.tags) && !(:cuda in test.tags)
end

