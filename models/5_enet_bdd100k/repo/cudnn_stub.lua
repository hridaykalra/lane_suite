--[[
cudnn_stub.lua

Fake/stub registration of the cudnn.* Torch classes actually found in
the ENet-SAD checkpoints (cudnn.SpatialConvolution, cudnn.ReLU,
cudnn.SpatialSoftMax), so that torch.load() can deserialize a
GPU-trained checkpoint on a machine with NO cudnn / NO CUDA / NO GPU
at all.

How this works: Torch7's file deserializer (torch/File.lua) only
needs *some* Lua class registered under the exact saved class name
(e.g. "cudnn.SpatialConvolution") to reconstruct an object -- it
creates a blank instance of that class and copies the saved fields
(weight, bias, kW, kH, etc.) onto it. It does NOT call the
constructor and does NOT validate methods at load time. So by
registering cudnn.SpatialConvolution as a *subclass* of the plain
nn.SpatialConvolution (which shares the exact same field layout),
the loaded object behaves identically to a real nn.SpatialConvolution
for forward-pass purposes -- no GPU, no cudnn library, no CUDA
Toolkit needed anywhere.

Usage: require this BEFORE torch.load(...), instead of requiring the
real 'cudnn' package.

    require 'nn'
    require 'cudnn_stub'
    local model = torch.load('ENet_trained.t7')
--]]

require 'torch'
require 'nn'

-- Make sure the top-level 'cudnn' table exists, since we're not
-- loading the real package.
cudnn = cudnn or {}

----------------------------------------------------------------------
-- cudnn.SpatialConvolution -> alias of nn.SpatialConvolution
-- Field layout (weight, bias, kW, kH, dW, dH, padW, padH, nInputPlane,
-- nOutputPlane) is identical between the two, so a plain subclass
-- with no overrides is sufficient and gives byte-identical forward
-- pass behaviour to the plain CPU nn version.
----------------------------------------------------------------------
do
    local SpatialConvolution, parent = torch.class(
        'cudnn.SpatialConvolution', 'nn.SpatialConvolution'
    )
    -- No overrides needed: inherits updateOutput etc. from
    -- nn.SpatialConvolution directly. This mirrors what Torch's own
    -- cudnn.convert() does internally when going the other direction.
end

----------------------------------------------------------------------
-- cudnn.ReLU -> custom updateOutput, NOT a plain alias.
--
-- nn.ReLU is actually implemented as nn.Threshold with self.threshold
-- and self.val set to 0 -- but those fields are set in nn.Threshold's
-- __init constructor, which is SKIPPED during deserialization (only
-- fields present in the saved checkpoint get copied onto the blank
-- object). The real cudnn.ReLU almost certainly only saves
-- `inplace`, not `threshold`/`val`, so subclassing nn.ReLU directly
-- leaves those nil and crashes inside Threshold_updateOutput.
--
-- Fix: give cudnn.ReLU its own trivial updateOutput that doesn't
-- depend on any saved fields at all -- max(x, 0) needs no parameters.
----------------------------------------------------------------------
do
    local ReLU, parent = torch.class('cudnn.ReLU', 'nn.Module')

    function ReLU:updateOutput(input)
        self.output = self.output or input.new()
        self.output:resizeAs(input)
        self.output:copy(input)
        self.output:cmax(0)
        return self.output
    end

    function ReLU:updateGradInput(input, gradOutput)
        error('cudnn_stub.ReLU: backward pass not implemented (inference-only stub)')
    end
end

----------------------------------------------------------------------
-- cudnn.SpatialSoftMax -> NOT a plain alias.
--
-- IMPORTANT: cudnn.SpatialSoftMax computes softmax across the
-- CHANNEL dimension at each spatial location (i.e. for every pixel
-- (h,w), softmax over the C channel values). This is different from
-- plain nn.SpatialSoftMax / nn.SoftMax, which normalize differently.
-- Aliasing this one to the wrong nn class would load without error
-- but silently produce WRONG numbers. So we implement updateOutput
-- ourselves with the correct per-pixel, cross-channel softmax math.
--
-- Input shape assumed: (N, C, H, W) or (C, H, W) -- standard Torch7
-- spatial tensor layout used throughout this codebase.
----------------------------------------------------------------------
do
    local SpatialSoftMax, parent = torch.class('cudnn.SpatialSoftMax', 'nn.Module')

    function SpatialSoftMax:updateOutput(input)
        local dim
        if input:dim() == 4 then
            dim = 2   -- (N, C, H, W) -- channel is dim 2 (1-indexed)
        elseif input:dim() == 3 then
            dim = 1   -- (C, H, W)
        else
            error('cudnn_stub.SpatialSoftMax: expected 3D or 4D input, got ' .. input:dim() .. 'D')
        end

        -- Numerically stable softmax along the channel dimension:
        -- subtract the per-pixel max before exponentiating.
        local maxVal = input:max(dim)
        local shifted = input - maxVal:expandAs(input)
        local expInput = torch.exp(shifted)
        local sumExp = expInput:sum(dim)
        self.output = expInput:cdiv(sumExp:expandAs(expInput))

        return self.output
    end

    function SpatialSoftMax:updateGradInput(input, gradOutput)
        -- Not needed for inference-only use, but defined defensively
        -- so nothing breaks if some code path calls backward().
        error('cudnn_stub.SpatialSoftMax: backward pass not implemented (inference-only stub)')
    end
end

print('[cudnn_stub] Registered CPU-compatible stand-ins for: '
    .. 'cudnn.SpatialConvolution, cudnn.ReLU, cudnn.SpatialSoftMax')

return cudnn
