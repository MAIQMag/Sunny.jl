# AMDGPU-specific hooks for KAExt. There is not yet a
# `Sunny.vendor_bogoliubov!` method, so KAExt falls back to its :jacobi or
# :host eigensolvers.
module KAAMDGPUExt

using AMDGPU: AMDGPU
using Sunny: Sunny

Sunny.device_free_memory(::AMDGPU.ROCBackend) = AMDGPU.free()

end
