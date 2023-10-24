import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.modules.pooling import _MaxPoolNd, _AdaptiveAvgPoolNd, _AvgPoolNd
from torch.autograd import Function
from enum import Enum

class QuantizerImpl(Function):

    @staticmethod
    def forward(ctx, x, scale, bound):
        scaled_x = x / scale
        ctx.bound = bound
        ctx.save_for_backward(scaled_x)
        return scaled_x.round_().clamp_(-bound, +bound) * scale

    @staticmethod
    def backward(ctx, grad):
        scaled_x = ctx.saved_tensors[0]
        bound    = ctx.bound
        zero = grad.new_zeros(1)
        grad = torch.where((scaled_x > -bound) | (scaled_x < +bound), grad, zero)
        return grad, None, None

class Algorithm:
    def __init__(self, name, **kwargs):
        self.name = name
        self.keys = kwargs
        self.keys["name"] = name
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __repr__(self):
        pack_args = []
        for key in self.keys:
            pack_args.append(f"{key}={self.keys[key]}")
        return "(" + ", ".join(pack_args) + ")"

class QuantizationStyle(Enum): # 继承Enum 更好的安全性、for能直接迭代、更好的可读性
    PerTensor  = 0
    PerChannel = 1

class QuantizationMethod:
    def __init__(self, bits:int, algo:Algorithm, dim:int, style:QuantizationStyle, once=False):
        self.bits = bits
        self.style = style
        self.algo = algo
        self.dim = dim
        self.once = once
        self.bitbound = int(2**(self.bits - 1) - 1)

    @staticmethod
    def per_tensor(bits:int, algo:Algorithm):
        return QuantizationMethod(bits, algo, 0, QuantizationStyle.PerTensor)
    
    @staticmethod
    def per_channel(bits:int, algo:Algorithm, dim:int, once:bool=True):
        return QuantizationMethod(bits, algo, dim, QuantizationStyle.PerChannel, once)

    def __repr__(self):
        return f"(style={self.style.name}, bits={self.bits}, algo={self.algo}, dim={self.dim}, once={self.once})"


class CalibMax:
    def __init__(self, method : QuantizationMethod):
        self.collect_datas = []
        self.method = method

    def collect(self, x):
        if self.method.style == QuantizationStyle.PerChannel:
            if len(self.collect_datas) > 0 and self.method.once:
                return
            
            x = x.abs()
            reduce_shape = list(range(len(x.shape)-1, -1, -1))
            del reduce_shape[reduce_shape.index(self.method.dim)]

            for i in reduce_shape:
                if x.shape[i] > 1:
                    x = torch.max(x, dim=i, keepdim=True)[0]

            self.collect_datas.append(x)
        else:
            self.collect_datas.append(x.abs().max())

    def compute(self):
        if self.method.style == QuantizationStyle.PerChannel:
            return self.collect_datas[0] / self.method.bitbound
        else:
            return torch.cat(self.collect_datas).mean() / self.method.bitbound


class CalibHistogram:
    def __init__(self, method : QuantizationMethod):
        self.collect_datas = None
        self.method = method

    def collect(self, x):
        if self.method.style == QuantizationStyle.PerChannel:
            raise NotImplementedError("Not implemented")
        else:
            x    = x.float().abs()
            xmax = x.max().item()
            if self.collect_datas is None:
                hist = torch.histc(x, self.method.algo.bins, min=0, max=xmax)
                self.collect_datas = hist, xmax / self.method.algo.bins, xmax, self.method.algo.bins
            else:
                prev_hist, width, prev_xmax, prev_bins = self.collect_datas
                new_xmax = max(prev_xmax, xmax)
                new_bins = max(prev_bins, int(math.ceil(xmax / width)))

                hist = torch.histc(x, new_bins, min=0, max=new_xmax)
                hist[:prev_hist.numel()] += prev_hist
                self.collect_datas = hist, width, new_xmax, new_bins

    def compute(self):
        if self.method.style == QuantizationStyle.PerChannel:
            raise NotImplementedError("Not implemented")
        
        hist, width, xmax, num_bins = self.collect_datas
        device    = hist.device
        centers   = torch.linspace(width / 2, xmax - width / 2, num_bins, device=device)
        start_bin = 128
        scaled_centers = centers[start_bin:] / self.method.bitbound
        mses      = torch.zeros(len(centers) - start_bin)
        centers   = centers.unsqueeze(1)
        hist      = hist.unsqueeze(1)
        scaled_centers = scaled_centers.unsqueeze(0)
        quant_centers = (centers / scaled_centers).round().clamp_(-self.method.bitbound, +self.method.bitbound) * scaled_centers
        mses = ((quant_centers - centers)**2 * hist).mean(0)
        index = torch.argmin(mses).item() + start_bin
        return torch.tensor(centers[index] / self.method.bitbound, dtype=torch.float32, device=device)


class Quantizer(nn.Module):
    def __init__(self, method : QuantizationMethod):
        super().__init__()

        self.enable   = False
        self.collect  = False # 调用collect的setter
        self.method   = method
        self.use_torch_quantize = False

    @property
    def collect(self):
        return self._collect

    @collect.setter # collect属性的设置方法
    def collect(self, new_value):
        self._collect = new_value
        if new_value:
            if self.method.algo.name == "max":
                self.collector = CalibMax(self.method)
            elif self.method.algo.name == "histogram":
                self.collector = CalibHistogram(self.method)
        else:
            self.collector = None

    def compute(self):
        assert self.collector is not None, "self.collector is None, please run collect data first."
        self.register_buffer("_scale", self.collector.compute())

    def forward(self, x):
        if not self.enable:
            return x

        if self._collect:
            self.collector.collect(x.detach())
            return x

        if self.use_torch_quantize:
            if self.method.style == QuantizationStyle.PerTensor:
                scale_sequeeze = self._scale.detach()
                return torch.fake_quantize_per_tensor_affine(x, scale_sequeeze, 0, -self.method.bitbound - 1, self.method.bitbound)
            elif self.method.style == QuantizationStyle.PerChannel:
                scale_sequeeze = self._scale.view(self._scale.numel()).detach()
                return torch.fake_quantize_per_channel_affine(
                    x, scale_sequeeze, scale_sequeeze.new_zeros(scale_sequeeze.shape, dtype=torch.int32), self.method.dim, -self.method.bitbound - 1, self.method.bitbound)
        return QuantizerImpl.apply(x, self._scale, self.method.bitbound)

    def __repr__(self):
        scale = None
        if hasattr(self, "_scale"):
            if self.method.style == QuantizationStyle.PerChannel:
                scale = f"scale=[min={self._scale.min().item()}, max={self._scale.max().item()}]"
            else:
                scale = f"scale={self._scale.item()}"
        return f"Quantizer({scale}, enable={self.enable}, collect={self.collect}, method={self.method})"


class QuantizerLinker(object):
    def __init__(self, model:nn.Module):
        self.model = model

    def foreach(self, fn):
        if isinstance(self.model, Quantizer):
            fn(self.model)

        for name, m in self.model.named_modules():
            if isinstance(m, Quantizer):
                fn(m)
        return self

    def __getattribute__(self, name: str):
        if name == "model":     return object.__getattribute__(self, "model")

        foreach = object.__getattribute__(self, "foreach")
        if name == "enable":    return foreach(lambda m: setattr(m, "enable", True))
        if name == "disable":   return foreach(lambda m: setattr(m, "enable", False))
        if name == "collect":   return foreach(lambda m: setattr(m, "collect", True))
        if name == "uncollect": return foreach(lambda m: setattr(m, "collect", False))
        if name == "compute":   return foreach(lambda m: m.compute())
        if name == "export":    return foreach(lambda m: setattr(m, "use_torch_quantize", True))
        if name == "unexport":  return foreach(lambda m: setattr(m, "use_torch_quantize", False))


class PTQCollect:
    def __init__(self, model):
        self.linker = QuantizerLinker(model)

    def __enter__(self):
        self.linker.enable.collect
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_value is not None:
            raise exc_value
        
        self.linker.compute.uncollect


class ExportQuantONNX:
    def __init__(self, model):
        self.linker = QuantizerLinker(model)

    def __enter__(self):
        self.linker.export
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_value is not None:
            raise exc_value
        
        self.linker.unexport


default_input_quantize_method  = QuantizationMethod.per_tensor(8, Algorithm("histogram", bins=2048))
default_weight_quantize_method = QuantizationMethod.per_channel(8, Algorithm("max"), 0)

class QuantConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_quantizer_  = Quantizer(default_input_quantize_method)
        self.weight_quantizer_ = Quantizer(default_weight_quantize_method)
    
    def forward(self, x):
        x = self.input_quantizer_(x)
        w = self.weight_quantizer_(self.weight)
        return self._conv_forward(x, w, self.bias)

class QuantLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_quantizer_  = Quantizer(default_input_quantize_method)
        self.weight_quantizer_ = Quantizer(default_weight_quantize_method)
    
    def forward(self, x):
        x = self.input_quantizer_(x)
        w = self.weight_quantizer_(self.weight)
        return F.linear(x, w, self.bias)

class QuantAdd(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_quantizer_  = Quantizer(default_input_quantize_method)
    
    def forward(self, a, b):
        return torch.add(self.input_quantizer_(a), self.input_quantizer_(b))

class QuantAdaptiveAvgPoolNd(_AdaptiveAvgPoolNd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_quantizer_  = Quantizer(default_input_quantize_method)

class QuantAdaptiveAvgPool1d(QuantAdaptiveAvgPoolNd, nn.AdaptiveAvgPool1d):
    def forward(self, x):
        return F.adaptive_avg_pool1d(self.input_quantizer_(x), self.output_size)
    
class QuantAdaptiveAvgPool2d(QuantAdaptiveAvgPoolNd, nn.AdaptiveAvgPool2d):
    def forward(self, x):
        return F.adaptive_avg_pool2d(self.input_quantizer_(x), self.output_size)

class QuantAdaptiveAvgPool3d(QuantAdaptiveAvgPoolNd, nn.AdaptiveAvgPool3d):
    def forward(self, x):
        return F.adaptive_avg_pool3d(self.input_quantizer_(x), self.output_size) 


class QuantMaxPoolNd(_MaxPoolNd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_quantizer_  = Quantizer(default_input_quantize_method)
    
class QuantMaxPool1d(QuantMaxPoolNd, nn.MaxPool1d): 
    def forward(self, x):
        x = self.input_quantizer_(x)
        return F.max_pool1d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)
    
class QuantMaxPool2d(QuantMaxPoolNd, nn.MaxPool2d): 
    def forward(self, x):
        x = self.input_quantizer_(x)
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)
    
class QuantMaxPool3d(QuantMaxPoolNd, nn.MaxPool3d): 
    def forward(self, x):
        x = self.input_quantizer_(x)
        return F.max_pool3d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)
    

class QuantAvgPoolNd(_AvgPoolNd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_quantizer_  = Quantizer(default_input_quantize_method)

class QuantAvgPool1d(QuantAvgPoolNd, nn.AvgPool1d): 
    def forward(self, x):
        x = self.input_quantizer_(x)
        return F.avg_pool1d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
    
class QuantAvgPool2d(QuantAvgPoolNd, nn.AvgPool2d): 
    def forward(self, x):
        x = self.input_quantizer_(x)
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
    
class QuantAvgPool3d(QuantAvgPoolNd, nn.AvgPool3d): 
    def forward(self, x):
        x = self.input_quantizer_(x)
        return F.avg_pool3d(x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)

modules_map = [
    (torch.nn, "Conv2d", QuantConv2d),
    (torch.nn, "Linear", QuantLinear),
    (torch.nn, "AdaptiveAvgPool1d", QuantAdaptiveAvgPool1d),
    (torch.nn, "AdaptiveAvgPool2d", QuantAdaptiveAvgPool2d),
    (torch.nn, "AdaptiveAvgPool3d", QuantAdaptiveAvgPool3d),
    (torch.nn, "MaxPool1d", QuantMaxPool1d),
    (torch.nn, "MaxPool2d", QuantMaxPool2d),
    (torch.nn, "MaxPool3d", QuantMaxPool3d),
    (torch.nn, "AvgPool1d", QuantAvgPool1d),
    (torch.nn, "AvgPool2d", QuantAvgPool2d),
    (torch.nn, "AvgPool3d", QuantAvgPool3d),
]

def add_replace_module(module, name, target):
    global modules_map
    modules_map.append((module, name, target))


class QuantReplacement:
    def __init__(self, enable=True):
        self.old_instance = []
        self.enable = enable

    def __enter__(self):
        if self.enable:
            for m, name, target in modules_map:
                self.old_instance.append((m, name, getattr(m, name)))
                setattr(m, name, target)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_value is not None:
            raise exc_value
        
        for m, name, target in self.old_instance:
            setattr(m, name, target)
        