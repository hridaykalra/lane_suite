--[[
infer.lua  (model 4: ENet-SAD, TuSimple, Lua Torch7)

Same pattern as models 3 and 5 -- standalone script bypassing
main.lua/testLaneE.lua's framework (GPU-only, DataLoader/list-file
based). GPU auto-detected the same way; falls back to CPU otherwise.

TuSimple differs from the CULane version (model 3) in two ways:
  - 6 lane classes instead of 4 (opt.nLane = 6 in the original repo)
  - resize target is 640x368 instead of 976x208

Output format mirrors the original repo's own testLaneE.lua: one
probability-map PNG per lane (up to 6), plus a text file listing which
lanes the model considers "present".

Preprocessing (resize, mean/std normalize) copied exactly from
datasets/laneE.lua's LaneDataset:preprocess().

Run with:  luajit.exe infer.lua
]]

require 'torch'
require 'nn'
require 'image'
require 'paths'

torch.setdefaulttensortype('torch.FloatTensor')

-- ==========================================================
-- GPU auto-detection (identical approach to models 3 and 5)
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
   -- No real cudnn available (no GPU) -- register CPU-compatible
   -- stand-ins for the cudnn.* classes so torch.load() can still
   -- deserialize this GPU-trained checkpoint.
   require 'cudnn_stub'
end

-- libppm.dll misparses well-formed PPM files on this Windows Torch7
-- build (same long/size-desync bug family as the checkpoint loader
-- issue) -- use a pure-Lua binary-PPM reader instead, unconditional
-- of GPU/CPU mode since this has nothing to do with cudnn.
local load_ppm_safe = require 'ppm_safe_loader'

-- ==========================================================
-- Auto jpg<->ppm conversion (Torch's libjpeg/libpng are missing
-- on this Windows build, so we shell out to Python/Pillow instead).
-- ==========================================================
local function script_dir()
   local str = debug.getinfo(1, "S").source:sub(2)
   return str:match("(.*[/\\])") or "./"
end

local THIS_DIR    = script_dir()
local PROJECT_ROOT = THIS_DIR .. "../../../"   -- models/4_enet_tusimple/repo/ -> project root

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

local save_pgm_safe = require 'pgm_safe_saver'
local save_ppm_safe = require 'ppm_safe_saver'

-- ==========================================================
-- Configuration (edit these 3 if your folder layout changes)
-- ==========================================================
local INPUT_DIR    = PROJECT_ROOT .. "common_input"
local OUTPUT_DIR   = THIS_DIR .. "output"
local WEIGHTS_PATH = THIS_DIR .. "../weights/ENet_trained.t7"

local RESIZE_W, RESIZE_H = 640, 368   -- matches datasets/laneE.lua ScaleWH(640, 368)
local MEAN = { 0.3598, 0.3653, 0.3662 }  -- exact values from datasets/laneE.lua
local STD  = { 0.2573, 0.2663, 0.2756 }

local NUM_LANES = 6           -- matches opt.nLane default in the original repo
local EXIST_THRESHOLD = 0.5   -- matches the repo's own testLaneE.lua

local IMAGE_EXTENSIONS = { ppm = true }  -- jpg/png/bmp disabled: libjpeg/libpng missing on this build; jpgs are pre-converted to ppm by jpg_to_ppm.py

-- ==========================================================
-- Helpers
-- ==========================================================

local function banner()
   print(string.rep("=", 60))
   print("        ENet-SAD Lane Detection -- TuSimple (Torch7)")
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

   -- Crop to the network's target aspect ratio (640:368 ~= 1.739) before
   -- resizing, instead of squashing the whole image. Squashing distorts
   -- road/lane geometry (especially badly on square 3072x3072 phone
   -- photos), which is likely why the model produced near-zero, scattered
   -- confidence on real-world images despite lanes being clearly visible.
   -- Crop from the top (sky/trees -- no lane info there); keep full width
   -- and the road-heavy bottom portion.
   local target_aspect = RESIZE_W / RESIZE_H
   local crop_h = math.floor(original_w / target_aspect)
   if crop_h > original_h then
      crop_h = original_h
   end
   local crop_y_offset = original_h - crop_h
   local cropped = img[{ {}, {crop_y_offset + 1, original_h}, {1, original_w} }]

   local resized = image.scale(cropped, RESIZE_W, RESIZE_H, 'bicubic')

   for c = 1, 3 do
      resized[c]:add(-MEAN[c]):div(STD[c])
   end

   local batch = resized:view(1, 3, RESIZE_H, RESIZE_W)

   if USE_GPU then
      batch = batch:cuda()
   end

   return batch, original_h, original_w, crop_h, crop_y_offset
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
         if image_path:lower():match('%.ppm$') then
            return load_ppm_safe(image_path)
         else
            return image.load(image_path, 3, 'float')
         end
      end)

      if not load_ok then
         print("  Could not read image. Skipping.\n")
         failed = failed + 1
      else
         local run_ok, err = pcall(function()
            local input_batch, original_h, original_w, crop_h, crop_y_offset = preprocess(img)

            local output = model:forward(input_batch)

            -- Model outputs a table: {scoremap (1 x 7 x H x W), exist}.
            -- scoremap is batched (leading dim = 1). exist may or may
            -- not carry that same leading batch dim depending on how
            -- this checkpoint's branch was built -- confirmed via
            -- earlier standalone testing that exist comes back as a
            -- flat 6-value vector, NOT a 1x6 batched tensor, so we
            -- handle both shapes defensively instead of assuming one.
            local scoremap = output[1]
            local exist = output[2]

            local softmax = nn.SpatialSoftMax()
            if USE_GPU then
               softmax:cuda()
            end
            local probs = softmax:forward(scoremap)[1]:float()   -- (NUM_LANES+1) x H x W

            local exist_vals
            if exist:dim() == 2 then
               exist_vals = exist[1]:float()   -- batched: 1 x NUM_LANES -> NUM_LANES
            else
               exist_vals = exist:float()      -- already unbatched: NUM_LANES
            end

            local stem = name:gsub("%.%a+$", "")
            local exist_flags = {}

-- Colored RGB overlay: original image with each detected lane drawn
-- in a distinct color, instead of separate black-and-white
-- probability maps per lane.
local LANE_COLORS = {
   {1, 0, 0},   -- lane 1: red
   {0, 1, 0},   -- lane 2: green
   {0, 0, 1},   -- lane 3: blue
   {1, 1, 0},   -- lane 4: yellow
   {1, 0, 1},   -- lane 5: magenta
   {0, 1, 1},   -- lane 6: cyan
}
local LANE_PIXEL_TOPK_FRACTION = 0.0015  -- highlight only the top 0.15% most-confident pixels per lane (rank-based, robust to low/noisy peaks) 

local overlay = img:clone()

for lane_idx = 1, NUM_LANES do
   exist_flags[lane_idx] = (exist_vals[lane_idx] > EXIST_THRESHOLD) and "1" or "0"

   if exist_flags[lane_idx] == "1" then
      local lane_prob = probs[lane_idx + 1]   -- +1 skips background channel
      local lane_prob_full = image.scale(lane_prob, original_w, original_h, 'bilinear')
      local flat = lane_prob_full:view(-1)
      local numel = flat:size(1)
      local k = math.max(50, math.floor(numel * LANE_PIXEL_TOPK_FRACTION))
      local sorted = torch.sort(flat, 1, true)  -- descending
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
