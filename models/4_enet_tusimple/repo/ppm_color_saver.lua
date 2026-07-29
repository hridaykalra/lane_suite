--[[
ppm_color_saver.lua

Pure-Lua binary PPM (P6) writer for 3-channel RGB float tensors,
same rationale and pattern as pgm_safe_saver.lua (P5 grayscale writer):
this Windows Torch7 build has no libpng, so image.save() can't write
PNGs. This bypasses that entirely with a hand-rolled P6 writer -- no
external DLL needed.

Input contract: tensor must be 3 x H x W, float, values in [0, 1]
(matches image.* tensor convention used everywhere else in this repo).

Usage:
    local save_ppm_color = require 'ppm_color_saver'
    save_ppm_color(path, tensor)   -- tensor: 3 x H x W float [0,1]
]]

local function save_ppm_color(filename, tensor)
   assert(tensor:dim() == 3, "save_ppm_color expects a 3 x H x W tensor, got dim=" .. tensor:dim())
   assert(tensor:size(1) == 3, "save_ppm_color expects 3 channels (RGB), got size(1)=" .. tensor:size(1))

   local clamped = tensor:clone():clamp(0, 1)
   local C, H, W = clamped:size(1), clamped:size(2), clamped:size(3)

   -- Convert C x H x W -> H x W x C (interleaved RGB), then to bytes.
   local interleaved = clamped:permute(2, 3, 1):contiguous()
   local byte_tensor = interleaved:mul(255):add(0.5):floor():byte()

   local file = torch.DiskFile(filename, 'w')
   file:binary()

   -- P6 header: magic, width, height, maxval, each followed by a single '\n'
   local header = string.format("P6\n%d %d\n255\n", W, H)
   for i = 1, #header do
      file:writeByte(header:byte(i))
   end

   local storage = byte_tensor:storage()
   file:writeByte(storage)
   file:close()
end

return save_ppm_color
