--[[
cuda_stub.lua
Registers torch.CudaTensor / torch.CudaStorage as plain aliases of
torch.FloatTensor / torch.FloatStorage, so a checkpoint saved with raw
CudaTensor/CudaStorage objects (not just cudnn.* layers) can be
deserialized on a machine with no cutorch/CUDA at all. The on-disk
binary layout for a Storage's data section is a flat array of its
element type (float), independent of the Cuda/Float label, so this is
safe for inference-only loading.
Usage: require this BEFORE torch.load(...), same pattern as cudnn_stub.
--]]
require 'torch'

if not torch.getmetatable('torch.CudaStorage') then
    torch.class('torch.CudaStorage', 'torch.FloatStorage')
end
if not torch.getmetatable('torch.CudaTensor') then
    torch.class('torch.CudaTensor', 'torch.FloatTensor')
end

print('[cuda_stub] Registered CPU-compatible stand-ins for: torch.CudaTensor, torch.CudaStorage')
