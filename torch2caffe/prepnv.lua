require 'nn'
require 'cunn'
require 'cudnn'

local trans = require 'torch2caffe.transforms'

local function adapt_conv1(layer)
  local std = torch.FloatTensor({0.229, 0.224, 0.225}) * 255
  local sz = layer.weight:size()
  sz[2] = 1
  layer.weight = layer.weight:cdiv(std:view(1,3,1,1):repeatTensor(sz))
  local tmp = layer.weight:clone()
  tmp[{{}, 1, {}, {}}] = layer.weight[{{}, 3, {}, {}}]
  tmp[{{}, 3, {}, {}}] = layer.weight[{{}, 1, {}, {}}]
  layer.weight = tmp:clone()
end

local function adapt_spatial_dropout(net)
  -- does not support recursive sequential(dropout)
  --print (model)
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


g_t2c_preprocess = function(model, opts)
    if model.net then
        model = model.net
    end
    model = cudnn.convert(model, nn)
    model = trans.fold_batch_normalization_layers(model, opts)
    model=nn.utils.recursiveType(model, 'torch.FloatTensor')
    for _, layer in pairs(model:findModules('nn.SpatialBatchNormalization')) do
        if 1 then
            layer.save_mean = layer.running_mean
            layer.save_std = layer.running_var
            layer.save_std:pow(-0.5)
        end
        --layer.train = true
    end
    --adapt_conv1(model.modules[1])
    adapt_spatial_dropout(model)
    return model
end



