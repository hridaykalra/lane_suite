local function load_ppm_safe(filename)
  local f = torch.DiskFile(filename, 'r')
  f:binary()
  f:quiet()
  local function read_token()
    local chars = {}
    local c
    while true do
      c = f:readChar()
      if c == 35 then
        while c ~= 10 do c = f:readChar() end
      elseif c == 32 or c == 9 or c == 10 or c == 13 then
      else
        break
      end
    end
    while c and c ~= 32 and c ~= 9 and c ~= 10 and c ~= 13 do
      table.insert(chars, string.char(c))
      c = f:readChar()
    end
    return table.concat(chars)
  end
  local magic = read_token()
  assert(magic == 'P6', 'load_ppm_safe: only binary P6 PPM supported, got ' .. tostring(magic))
  local width = tonumber(read_token())
  local height = tonumber(read_token())
  local maxval = tonumber(read_token())
  assert(maxval == 255, 'load_ppm_safe: only maxval=255 supported, got ' .. tostring(maxval))
  local numBytes = width * height * 3
  local storage = torch.ByteStorage(numBytes)
  local bytesRead = f:readByte(storage)
  assert(bytesRead == numBytes, string.format('load_ppm_safe: expected %d pixel bytes, got %d', numBytes, bytesRead))
  f:close()
  local raw = torch.ByteTensor(storage, 1, torch.LongStorage{height, width, 3})
  local img = raw:float():div(255)
  img = img:permute(3, 1, 2):contiguous()
  return img
end

return load_ppm_safe
