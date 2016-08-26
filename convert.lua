require 'nn';
require 'cunn';
require 'cudnn';
local t2c=require 'torch2caffe.lib'
local trans = require 'torch2caffe.transforms'

--require 'LocalNormalization'
Test={}
g_SLOW=""

-- Figure out the path of the model and load it
local path = arg[1]
local basename = paths.basename(path, 't7b')
local ext = path:match("^.+(%..+)$")
local model = nil
if ext == '.t7' then 
    model = torch.load(path)
    model2 = torch.load(path)
elseif ext == '.txt' then
    error('wrong model')
else
    assert(false, "We assume models end in either .t7b or .txt")
end


local function adapt_spatial_dropout(net)
    for i = 1, #net.modules do
        local c = net:get(i)
        local t = torch.type(c)
        if c.modules then
            adapt_spatial_dropout(c)
        elseif t == 'nn.SpatialDropout' then
            local m = nn.Dropout(c.p, false,true)
            local found = false
            -- Ignore groups
            net.modules[i] = m
            for j = i,1,-1 do
                local block_type = torch.type(net:get(j))
                if block_type == 'nn.SpatialConvolution'
                    or block_type == 'nn.Linear' then
                 --or block_type == 'nn.SpatialBatchNormalization' then
                    net.modules[j].weight:mul(1 - c.p)
                    if net.modules[j].bias then
                        net.modules[j].bias:mul(1 - c.p)
                    end
                    found = true
                    break
                end
            end
            if not found then
                error('SpatialDropout module cannot find weight to scale')
            end 
        end
    end
end

local function g_t2c_preprocess(model,opts)
  model = cudnn.convert(model, nn)
  model=nn.utils.recursiveType(model, 'torch.FloatTensor')
  for _, layer in pairs(model:findModules('nn.SpatialBatchNormalization')) do
    if layer.save_mean==nil then
      layer.save_mean = layer.running_mean
      layer.save_std = layer.running_var
      layer.save_std:pow(-0.5)
    end
    --layer.train = true
  end
  adapt_spatial_dropout(model)
  return model
end

if model.net then
    model = model.net
    model2 = model2.net
end

model=g_t2c_preprocess(model, opts)

local function check(module, module2,input_dims)
    module:apply(function(m) m:evaluate() end)
    local opts = {
            prototxt = string.format('%s.prototxt', basename),
            caffemodel = string.format('%s.caffemodel', basename),
            inputs={{name="data", input_dims=input_dims}},
    }
    t2c.convert(opts, module)
    t2c.compare(opts, module2)
    return opts
end


check(model, model2, {1,3,224,224})

