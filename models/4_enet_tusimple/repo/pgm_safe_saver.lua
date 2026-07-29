-- pgm_safe_saver.lua
-- Pure-Lua binary PGM (P5) writer, bypassing libpng.dll which is
-- absent from this Windows Torch7 build (same missing-codec issue
-- as PNG reading, just on the write side).
-- Input: a 2D FloatTensor (H x W), values in [0,1].
-- Output: a binary P5 PGM file, 8-bit grayscale.

local function save_pgm_safe(filename, tensor)
  assert(tensor:dim() == 2, 'save_pgm_safe: expected a 2D (H x W) tensor')
  local height, width = tensor:size(1), tensor:size(2)

  local clamped = tensor:clone():clamp(0, 1):mul(255):add(0.5):floor()
  local bytes = clamped:byte()

  local f = torch.DiskFile(filename, 'w')
  f:binary()
  f:quiet()

  local header = string.format('P5\n%d %d\n255\n', width, height)
  for i = 1, #header do
    f:writeChar(header:byte(i))
  end

  local flat = bytes:contiguous():view(-1)
  local storage = flat:storage()
  f:writeByte(storage)
  f:close()
end

return save_pgm_safe
