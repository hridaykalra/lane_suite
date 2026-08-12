-- ppm_safe_saver.lua
-- Pure-Lua binary PPM (P6) writer, bypassing libpng.dll which is
-- absent from this Windows Torch7 build (same missing-codec issue
-- as PNG/PPM reading, just on the write side).
-- Input: a 3D FloatTensor (3 x H x W), values in [0,1].
-- Output: a binary P6 PPM file, 8-bit RGB.

local function save_ppm_safe(filename, tensor)
  assert(tensor:dim() == 3, 'save_ppm_safe: expected a 3D (3 x H x W) tensor')
  assert(tensor:size(1) == 3, 'save_ppm_safe: expected 3 channels (RGB)')
  local height, width = tensor:size(2), tensor:size(3)

  local clamped = tensor:clone():clamp(0, 1):mul(255):add(0.5):floor()

  -- PPM byte order is interleaved per-pixel: R,G,B,R,G,B,...
  -- Our tensor is C x H x W (planar), so permute to H x W x C first.
  local interleaved = clamped:permute(2, 3, 1):contiguous()
  local bytes = interleaved:byte()

  local f = torch.DiskFile(filename, 'w')
  f:binary()
  f:quiet()

  local header = string.format('P6\n%d %d\n255\n', width, height)
  for i = 1, #header do
    f:writeChar(header:byte(i))
  end

  local flat = bytes:view(-1)
  local storage = flat:storage()
  f:writeByte(storage)
  f:close()
end

return save_ppm_safe
