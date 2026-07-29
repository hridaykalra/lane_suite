--[[
infer.lua  (model 3: ENet-SAD, CULane, Lua Torch7)

Standalone inference script, same pattern as models 4 and 5:
does NOT go through main.lua/testLane.lua's framework (both hard-require
'cunn'/'cudnn' unconditionally, and testLane.lua also needs a DataLoader
+ list-file setup rather than a plain folder of images). Instead this
loads the checkpoint directly and runs a plain forward pass per image.

GPU support: auto-detects 'cutorch', falls back to CPU otherwise.

Checkpoint loading uses a manual DiskFile reader with longSize(8) --
the checkpoint was saved on Linux (8-byte C longs) and this Windows
Torch7 build defaults to 4-byte longs, which desyncs the byte stream
and throws "table index is nil" / "unknown object" if read normally.

Image I/O: this Windows Torch7 build has no libjpeg/libpng, only
libppm -- and libppm.dll itself misparses well-formed PPM files (same
long/size-desync bug family). So:
  - Input images are converted jpg -> ppm by shelling out to Python
    (jpg_to_ppm.py, Pillow) before this script reads anything.
  - Images are then read with a pure-Lua PPM reader (ppm_safe_loader).
  - Output is a colored overlay (original photo + lanes drawn in
    distinct colors) saved via a pure-Lua PPM writer (ppm_safe_saver),
    then converted ppm -> jpg by shelling out to Python again
    (ppm_to_jpg_3.py) at the very end.

Run with:  luajit.exe infer.lua
]]

require 'torch'
require 'nn'
require 'image'
require 'paths'

torch.setdefaulttensortype('torch.FloatTensor')

-- ==========================================================
-- GPU auto-detection
-- ==========================================================
local USE_GPU = false

local cutorch_ok = pcall(require, 'cutorch')
if cutorch_ok then
   local cunn_ok = pcall(require, 'cunn')
   local cudnn_ok = pcall(require, 'cudnn')
   if cunn_ok and cutorch.getDeviceCount() > 0 then
      USE_GPU = true
      if cudnn_ok then
         cudnn.fastest = true
         cudnn.benchmark = true
      end
   end
end

if not USE_GPU then
   -- Register CPU-compatible stand-ins for cudnn.* classes so torch.load()
   -- can deserialize this GPU-trained checkpoint without a real GPU/cudnn.
   require 'cudnn_stub'
end

-- Pure-Lua PPM reader/writer (bypasses broken libppm.dll on this build)
local load_ppm_safe = require 'ppm_safe_loader'
local save_ppm_safe = require 'ppm_safe_saver'

-- ==========================================================
-- Auto jpg<->ppm conversion (Torch's libjpeg/libpng are missing
-- on this Windows build, so we shell out to Python/Pillow instead).
-- ==========================================================
local function script_dir()
   local str = debug.getinfo(1, "S").source:sub(2)
   return str:match("(.*[/\\])") or "./"
end

local THIS_DIR    = script_dir()
local PROJECT_ROOT = THIS_DIR .. "../../../"   -- models/3_enet_culane/repo/ -> project root

local PYTHON_EXE        = "python"
local JPG_TO_PPM_SCRIPT  = PROJECT_ROOT .. "jpg_to_ppm.py"
local PPM_TO_JPG_SCRIPT  = THIS_DIR .. "postprocess_ppm_to_jpg.py"

local function run_python_step(script_path, label)
   print('Running ' .. label .. ' ...')
   local cmd = string.format('""%s" "%s""', PYTHON_EXE, script_path)
   local ok = os.execute(cmd)
   if not ok then
      print('  WARNING: ' .. label .. ' may have failed (os.execute returned non-success).')
   end
end

-- ==========================================================
-- Configuration
-- ==========================================================
local INPUT_DIR    = PROJECT_ROOT .. "common_input"
local OUTPUT_DIR   = THIS_DIR .. "output"
local WEIGHTS_PATH = THIS_DIR .. "../weights/ENet-label-new.t7"

local RESIZE_W, RESIZE_H = 976, 208   -- matches datasets/lane.lua ScaleWH(976, 208)
local MEAN = { 0.3598, 0.3653, 0.3662 }
local STD  = { 0.2573, 0.2663, 0.2756 }

local NUM_LANES = 4
local EXIST_THRESHOLD = 0.5

local IMAGE_EXTENSIONS = { ppm = true }  -- jpg/png disabled: no codec on this build; pre-converted by jpg_to_ppm.py

-- ==========================================================
-- Helpers
-- ==========================================================

local function banner()
   print(string.rep("=", 60))
   print("        ENet-SAD Lane Detection -- CULane (Torch7)")
   print(string.rep("=", 60))
   if USE_GPU then
      print("GPU detected -- running on CUDA device 0 (" .. cutorch.getDeviceCount() .. " device(s) available)")
   else
      print("No usable GPU/cutorch found -- running on CPU")
   end
   print("")
end

local function list_images(dir)
   local files = {}
   for name in paths.iterfiles(dir) do
      local ext = name:match("^.+%.(%a+)$")
      if ext and IMAGE_EXTENSIONS[ext:lower()] then
         table.insert(files, name)
      end
   end
   table.sort(files)
   return files
end

local function ensure_dir(dir)
   if not paths.dirp(dir) then
      paths.mkdir(dir)
   end
end

local function load_model()
   assert(paths.filep(WEIGHTS_PATH), "Checkpoint not found: " .. WEIGHTS_PATH)
   print("Loading checkpoint (this can take a little while)...")

   local function load_with_long_fix(filename)
      local file = torch.DiskFile(filename, 'r')
      file:binary()
      file:referenced(true)
      file:longSize(8)
      file:littleEndianEncoding()
      local object = file:readObject()
      file:close()
      return object
   end

   local model = load_with_long_fix(WEIGHTS_PATH)

   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   if USE_GPU then
      model:cuda()
   else
      model:float()
   end
   model:evaluate()

   print("Model loaded successfully.\n")
   return model
end

local function preprocess(img)
   local original_h, original_w = img:size(2), img:size(3)

   local resized = image.scale(img, RESIZE_W, RESIZE_H, 'bicubic')

   for c = 1, 3 do
      resized[c]:add(-MEAN[c]):div(STD[c])
   end

   local batch = resized:view(1, 3, RESIZE_H, RESIZE_W)

   if USE_GPU then
      batch = batch:cuda()
   end

   return batch, original_h, original_w
end

local function cleanup_ppm_files(dir)
   local removed, failed = 0, 0
   for name in paths.iterfiles(dir) do
      local ext = name:match("^.+%.(%a+)$")
      if ext and ext:lower() == "ppm" then
         local full_path = dir .. "/" .. name
         local ok = os.remove(full_path)
         if ok then
            removed = removed + 1
         else
            failed = failed + 1
            print("  WARNING: could not delete stale ppm: " .. full_path)
         end
      end
   end
   print(string.format("Cleaned up %d .ppm file(s) from %s (%d failed)", removed, dir, failed))
end
-- ==========================================================
-- Main
-- ==========================================================

local function main()
   banner()
   run_python_step(JPG_TO_PPM_SCRIPT, 'jpg -> ppm conversion')

   local ok, model = pcall(load_model)
   if not ok then
      print("FATAL ERROR loading model:")
      print(model)
      return
   end

   ensure_dir(OUTPUT_DIR)

   local images = list_images(INPUT_DIR)
   print("Input Folder : " .. INPUT_DIR)
   print("Output Folder: " .. OUTPUT_DIR)
   print("Images Found : " .. #images)
   print("")

   if #images == 0 then
      print("No supported images were found in that folder.")
      return
   end

   local processed, failed = 0, 0

   for _, name in ipairs(images) do
      local start_time = os.clock()
      local image_path = INPUT_DIR .. "/" .. name
      print("Processing : " .. name)

      local load_ok, img = pcall(function()
         return load_ppm_safe(image_path)
      end)

      if not load_ok then
         print("  Could not read image. Skipping.\n")
         failed = failed + 1
      else
         local run_ok, err = pcall(function()
            local input_batch, original_h, original_w = preprocess(img)

            local output = model:forward(input_batch)

            -- Model outputs a table: {scoremap (1 x 5 x H x W), exist (1 x 4)}
            -- Channel 1 = background, channels 2-5 = lane 1-4.
            local scoremap = output[1]
            local exist = output[2]

            local softmax = nn.SpatialSoftMax()
            if USE_GPU then
               softmax:cuda()
            end
            local probs = softmax:forward(scoremap)[1]:float()   -- 5 x H x W

            local exist_vals
            if exist:dim() == 2 then
               exist_vals = exist[1]:float()
            else
               exist_vals = exist:float()
            end

            local stem = name:gsub("%.%a+$", "")
            local exist_flags = {}

            -- Colored RGB overlay: original image with each detected lane
            -- drawn in a distinct color.
            local LANE_COLORS = {
               {1, 0, 0},   -- lane 1: red
               {0, 1, 0},   -- lane 2: green
               {0, 0, 1},   -- lane 3: blue
               {1, 1, 0},   -- lane 4: yellow
            }
            local LANE_PIXEL_TOPK_FRACTION = 0.0015

            local overlay = img:clone()

            for lane_idx = 1, NUM_LANES do
               exist_flags[lane_idx] = (exist_vals[lane_idx] > EXIST_THRESHOLD) and "1" or "0"

               if exist_flags[lane_idx] == "1" then
                  local lane_prob = probs[lane_idx + 1]
                  local lane_prob_full = image.scale(lane_prob, original_w, original_h, 'bilinear')
                  local flat = lane_prob_full:view(-1)
                  local numel = flat:size(1)
                  local k = math.max(50, math.floor(numel * LANE_PIXEL_TOPK_FRACTION))
                  local sorted = torch.sort(flat, 1, true)
                  local rank_threshold = sorted[k]
                  local mask = lane_prob_full:gt(rank_threshold)

                  local color = LANE_COLORS[((lane_idx - 1) % #LANE_COLORS) + 1]
                  for c = 1, 3 do
                     overlay[c]:maskedFill(mask, color[c])
                  end
               end
            end

            local overlay_path = OUTPUT_DIR .. "/" .. stem .. "_overlay.ppm"
            save_ppm_safe(overlay_path, overlay)

            local exist_path = OUTPUT_DIR .. "/" .. stem .. "_exist.txt"
            local f = assert(io.open(exist_path, "w"))
            f:write(table.concat(exist_flags, " "))
            f:close()

            local elapsed = os.clock() - start_time
            print(string.format("  Saved RGB overlay + exist.txt for %s  (%.2fs)\n", name, elapsed))
         end)

         if run_ok then
            processed = processed + 1
         else
            print("  ERROR while processing " .. name .. ":")
            print("  " .. tostring(err))
            print("")
            failed = failed + 1
         end
      end
   end

   print(string.rep("=", 60))
   print(string.format("Done. Processed: %d  Failed: %d  Total: %d", processed, failed, #images))
   print(string.rep("=", 60))
   run_python_step(PPM_TO_JPG_SCRIPT, 'ppm -> jpg conversion')
   cleanup_ppm_files(INPUT_DIR)
end

main()