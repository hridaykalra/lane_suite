--[[
infer_folder.lua  (model 9: SCNN, Torch7, VGG-16, CULane)

Standalone inference script, same pattern as models 3/4/5. No CLI
args -- paths hardcoded below, project convention.

Requires the CPU-converted checkpoint from convert_to_cpu.lua
(vgg_SCNN_DULR_w9_cpu.t7) -- NOT the original cudnn-only checkpoint.
That conversion strips cudnn.* layers entirely, so unlike model 5 this
script does NOT need cuda_stub/cudnn_stub -- nothing in this file (or
in the converted checkpoint) makes an unconditional cudnn.xxx call.

GPU support: auto-detects 'cutorch'/'cunn', falls back to CPU.

Checkpoint loading uses a manual DiskFile reader with longSize(8) --
same Linux-vs-Windows long-size desync fix as models 3/4/5, needed
because convert_to_cpu.lua will be run on a Linux GPU machine and the
resulting .t7 gets read here on Windows.

Image I/O: jpg -> ppm (Python/Pillow) before reading, pure-Lua PPM
read/write during processing, ppm -> jpg (Python/Pillow) at the end --
same pattern as models 3/4/5, since this Windows Torch7 build has no
libjpeg/libpng and libppm.dll misparses well-formed PPM files.

Run with:  luajit.exe infer_folder.lua
]]

require 'torch'
require 'nn'
require 'image'
require 'paths'

io.stdout:setvbuf('line')   -- make sure "Processing : X" prints immediately, not after a buffer flush --
                             -- otherwise a slow-but-working CPU run can look identical to a hang

torch.setdefaulttensortype('torch.FloatTensor')

-- ==========================================================
-- CPU threading
-- ==========================================================
-- Torch defaults to a single CPU thread unless told otherwise. On a
-- VGG-16 backbone at the full 800x288 SCNN input size, single-threaded
-- convolution is genuinely slow (easily 1-3+ min for the first image)
-- -- not stuck, just doing a lot of single-core work.
local NUM_THREADS = tonumber(os.getenv('NUMBER_OF_PROCESSORS')) or 4
torch.setnumthreads(NUM_THREADS)
print("Using " .. NUM_THREADS .. " CPU thread(s) for inference (set NUMBER_OF_PROCESSORS to override)\n")

-- ==========================================================
-- GPU auto-detection
-- ==========================================================
local USE_GPU = false

local cutorch_ok = pcall(require, 'cutorch')
if cutorch_ok then
   local cunn_ok = pcall(require, 'cunn')
   if cunn_ok and cutorch.getDeviceCount() > 0 then
      USE_GPU = true
   end
end
-- No cuda_stub/cudnn_stub needed here: the converted checkpoint has no
-- cudnn.* modules left in it, and nothing below makes an unconditional
-- cudnn.xxx call outside the `if USE_GPU` branches.

-- Pure-Lua PPM reader/writer (bypasses broken libppm.dll on this build)
local load_ppm_safe = require 'ppm_safe_loader'
local save_ppm_safe = require 'ppm_safe_saver'

-- ==========================================================
-- Auto jpg<->ppm conversion
-- ==========================================================
local function script_dir()
   local str = debug.getinfo(1, "S").source:sub(2)
   return str:match("(.*[/\\])") or "./"
end

local THIS_DIR     = script_dir()
local PROJECT_ROOT = THIS_DIR .. "../../../"   -- models/9_scnn_culane/repo/ -> project root

local PYTHON_EXE       = "python"   -- assumes Python is on PATH
local JPG_TO_PPM_SCRIPT = PROJECT_ROOT .. "jpg_to_ppm.py"     -- shared, lives at project root
local PPM_TO_JPG_SCRIPT = THIS_DIR .. "postprocess_ppm_to_jpg.py"  -- per-model copy, like model 5

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
local WEIGHTS_PATH = THIS_DIR .. "../weights/vgg_SCNN_DULR_w9_cpu.t7"

local NET_W, NET_H = 800, 288          -- fixed network input size (from training)
local EXIST_THRESH = 0.4
local LANE_PROB_THRESH = 0.3           -- same threshold as original getLane.m
local NUM_ROW_ANCHORS = 18
local CULANE_REF_H = 590               -- reference height the 18 row anchors were tuned on

local MEAN = { 0.3598, 0.3653, 0.3662 }
local STD  = { 0.2573, 0.2663, 0.2756 }

local LANE_COLORS = {
   {0, 1, 0},   -- lane 1: green   (PPM is RGB, values 0..1 here, not 0..255)
   {0, 0, 1},   -- lane 2: blue
   {1, 0, 0},   -- lane 3: red
   {1, 1, 0},   -- lane 4: yellow
}

local IMAGE_EXTENSIONS = { ppm = true }  -- jpg/png disabled: no codec on this build; pre-converted by jpg_to_ppm.py

-- ==========================================================
-- Helpers
-- ==========================================================

local function banner()
   print(string.rep("=", 60))
   print("        SCNN -- CULane, VGG-16 (Torch7)")
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

local function load_model()
   assert(paths.filep(WEIGHTS_PATH), "Checkpoint not found: " .. WEIGHTS_PATH)
   print("Loading checkpoint (this can take a little while)...")

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
   -- img: 3 x H x W float tensor, values 0..1 (from load_ppm_safe)
   local original_h, original_w = img:size(2), img:size(3)

   -- Letterbox: scale to FIT inside NET_W x NET_H while preserving aspect
   -- ratio, then pad the remainder -- instead of directly squishing the
   -- image to 800x288, which distorts anything that isn't already
   -- CULane's native ~2.78:1 dashcam aspect ratio.
   local scale = math.min(NET_W / original_w, NET_H / original_h)
   local scaled_w = math.max(1, math.floor(original_w * scale + 0.5))
   local scaled_h = math.max(1, math.floor(original_h * scale + 0.5))
   local pad_left = math.floor((NET_W - scaled_w) / 2)
   local pad_top  = math.floor((NET_H - scaled_h) / 2)

   local resized = image.scale(img, scaled_w, scaled_h, 'bicubic')

   -- Pad with the per-channel mean color (not black): after normalization
   -- this becomes ~0, i.e. neutral, instead of a sharp artificial dark
   -- border that could itself trigger spurious edge activations.
   local canvas = torch.FloatTensor(3, NET_H, NET_W)
   for c = 1, 3 do
      canvas[c]:fill(MEAN[c])
   end
   canvas[{ {}, { pad_top + 1, pad_top + scaled_h }, { pad_left + 1, pad_left + scaled_w } }] = resized

   local normed = canvas:clone()
   for c = 1, 3 do
      normed[c]:add(-MEAN[c]):div(STD[c])
   end
   local batch = normed:view(1, 3, NET_H, NET_W)
   if USE_GPU then
      batch = batch:cuda()
   end
   -- Return everything needed to map network-space coords back to the
   -- original image: pad offsets + the single scale factor (uniform in
   -- both axes, since we preserved aspect ratio).
   return batch, original_h, original_w, scale, pad_left, pad_top
end

-- Reimplementation of tools/prob2lines/getLane.m -- operates directly on
-- a NET_H x NET_W float probability map (0..1) instead of a saved
-- 0..255 png, so no /255 division needed here.
local function get_lane_points(probMap, origW, origH, scale, padLeft, padTop)
   local points = {}
   local validCount = 0
   for i = 1, NUM_ROW_ANCHORS do
      local rowNet = math.floor(NET_H - (i - 1) * 20 / CULANE_REF_H * NET_H)
      rowNet = math.max(1, math.min(NET_H, rowNet))
      local row = probMap[{rowNet, {}}]
      local value, idx = row:max(1)
      value = value[1]
      idx = idx[1]
      if value > LANE_PROB_THRESH then
         -- Invert the letterbox transform: subtract the pad offset, then
         -- divide by the single uniform scale factor (not separate W/H
         -- ratios like before, since aspect ratio is now preserved).
         table.insert(points, {
            x = (idx - padLeft) / scale,
            y = (rowNet - padTop) / scale,
         })
         validCount = validCount + 1
      else
         table.insert(points, nil)
      end
   end
   if validCount < 2 then
      return {}
   end
   return points
end

local function draw_point(img, x, y, color)
   -- img is C x H x W float [0,1]
   local H, W = img:size(2), img:size(3)
   local ix, iy = math.floor(x + 0.5), math.floor(y + 0.5)
   for dy = -3, 3 do
      for dx = -3, 3 do
         local px, py = ix + dx, iy + dy
         if px >= 1 and px <= W and py >= 1 and py <= H then
            img[1][py][px] = color[1]
            img[2][py][px] = color[2]
            img[3][py][px] = color[3]
         end
      end
   end
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

   local softmax = nn.SpatialSoftMax()
   if USE_GPU then softmax = softmax:cuda() end

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
            local input_batch, origH, origW, scale, padLeft, padTop = preprocess(img)

            io.write("  Running forward pass (CPU, single image can take a while)...")
            io.flush()
            local output = model:forward(input_batch)
            print(" done.")
            local scoremap = output[1]     -- 1 x 5 x H x W (background + 4 lanes)
            local existRaw = output[2]     -- 1 x 4

            local probs = softmax:forward(scoremap):float()[1]   -- 5 x H x W
            local exist = existRaw:float()[1]                     -- 4

            -- DEBUG: show raw exist scores and per-lane peak probability,
            -- so we can tell whether nothing is drawn because scores are
            -- genuinely low (calibration) or because something upstream
            -- is producing dead/wrong output.
            io.write("  exist = [")
            for laneIdx = 1, 4 do
               io.write(string.format("%.3f", exist[laneIdx]))
               if laneIdx < 4 then io.write(", ") end
            end
            io.write("]  (threshold " .. EXIST_THRESH .. ")\n")
            for laneIdx = 1, 4 do
               local probMap = probs[laneIdx + 1]
               local peak = probMap:max()
               print(string.format("    lane %d peak prob = %.3f (threshold %.2f)", laneIdx, peak, LANE_PROB_THRESH))
            end

            local stem = name:gsub("%.%a+$", "")
            local overlay = img:clone()
            local linesPath = OUTPUT_DIR .. "/" .. stem .. "_lines.txt"
            local fp = io.open(linesPath, 'w')

            local DEBUG_IGNORE_EXIST = true   -- TEMP: draw all lanes regardless of exist score, to isolate the bug

            for laneIdx = 1, 4 do
               if DEBUG_IGNORE_EXIST or exist[laneIdx] > EXIST_THRESH then
                  local probMap = probs[laneIdx + 1]   -- channel 1 is background
                  if laneIdx == 1 then
                     -- DEBUG: find where the global peak actually sits, to confirm
                     -- whether it's outside the sampled row-anchor band entirely
                     local flatMax, flatIdx = probMap:view(-1):max(1)
                     local flatIdxVal = flatIdx[1]
                     local peakRow = math.floor((flatIdxVal - 1) / NET_W) + 1
                     local peakCol = ((flatIdxVal - 1) % NET_W) + 1
                     print(string.format("    lane 1 global peak %.3f is at netRow=%d, netCol=%d (anchors only cover netRow 122-288)",
                        flatMax[1], peakRow, peakCol))
                  end
                  local points = get_lane_points(probMap, origW, origH, scale, padLeft, padTop)
                  print(string.format("    lane %d: points found = %d", laneIdx, #points))
                  if #points > 0 then
                     for _, pt in ipairs(points) do
                        if pt then
                           draw_point(overlay, pt.x, pt.y, LANE_COLORS[laneIdx])
                           fp:write(string.format('%d %d ', math.floor(pt.x), math.floor(pt.y)))
                        end
                     end
                     fp:write('\n')
                  end
               end
            end
            fp:close()

            local overlay_path = OUTPUT_DIR .. "/" .. stem .. "_overlay.ppm"
            save_ppm_safe(overlay_path, overlay)

            local elapsed = os.clock() - start_time
            print(string.format("  Saved overlay + lines for %s  (%.2fs)\n", name, elapsed))
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
