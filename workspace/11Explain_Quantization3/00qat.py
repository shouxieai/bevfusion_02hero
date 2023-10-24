import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.autograd import Function
from enum import Enum

class QuantizerImpl(Function):

    @staticmethod
    def forward(ctx, x, scale, bound):
        scaled_x = x / scale
        ctx.bound = bound
        ctx.save_for_backward(scaled_x)
        # scale = x.max() / 127
        # x / scale = x / x.max() * 127
        return scaled_x.round().clamp(-bound, +bound) * scale

    @staticmethod
    def backward(ctx, grad):
        scaled_x = ctx.saved_tensors[0]
        bound    = ctx.bound
        zero = grad.new_zeros(1)
        grad = torch.where((scaled_x < -bound) | (scaled_x > +bound), grad, zero)
        return grad, None, None

class Algorithm:
    def __init__(self, name, **kwargs):
        self.name = name
        self.keys = kwargs
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __repr__(self):
        pack_args = []
        for key in self.keys:
            pack_args.append(f"{key}={self.keys[key]}")
        return ", ".join(pack_args)

class QuantizationStyle(Enum):
    PerTensor  = 0
    PerChannel = 1

class QuantizationMethod:
    def __init__(self, bits:int, algo:Algorithm, dim:int, style:QuantizationStyle, once=False):
        self.bits = bits
        self.style = style
        self.algo = algo
        self.dim = dim
        self.once = once

    @property
    def bitbound(self):
        return float(2**(self.bits - 1) - 1)

    @staticmethod
    def per_tensor(bits:int, algo:Algorithm):
        return QuantizationMethod(bits, algo, -1, QuantizationStyle.PerTensor)
    
    @staticmethod
    def per_channel(bits:int, algo:Algorithm, dim:int, once:bool=True):
        return QuantizationMethod(bits, algo, dim, QuantizationStyle.PerChannel, once)

    def __repr__(self):
        return f"(style={self.style.name}, bits={self.bits}, algo={self.algo}, dim={self.dim}, once={self.once})"


class Collector:
    def __init__(self, method : QuantizationMethod):
        self.method = method
        if method.algo.name == "max":
            self.collect_datas = []
            self.worker = self.collect_max
            self.compute_scale = self.compute_scale_max
        elif method.algo.name == "histogram":
            self.collect_datas = []
            self.worker = self.collect_histogram
            self.compute_scale = self.compute_scale_histogram

    def __call__(self, x):
        self.worker(x)

    def collect_max(self, x):
        if self.method.style == QuantizationStyle.PerChannel:
            if len(self.collect_datas) > 0 and self.method.once: # 如果已经收集了数据（self.collect_datas不为空）并且once属性为True，则提前退出函数，不再收集数据。
                return
            
            x = x.abs()
            reduce_shape = list(range(len(x.shape)-1, -1, -1)) # 逆序的维度索引。如果x是4维 bchw  那么reduce_shape=[3, 2, 1, 0]
            del reduce_shape[reduce_shape.index(self.method.dim)] # 从reduce_shape列表中删除self.method.dim指定的维度。这是因为我们不想在这个特定的维度上进行缩减。

            for i in reduce_shape: # per_channel算最大值
                if x.shape[i] > 1: # 例如x.shape为[1, 3, 224, 224]   当i=0时， x.shape[0]=1  等于1了还算个啥最大值。就跳过了
                    x = torch.max(x, dim=i, keepdim=True)[0]

            self.collect_datas.append(x) # 添加最大值到collect_datas里
        else:
            self.collect_datas.append(x.abs().max().item())# pertensor直接往列表中存最大值

    def compute_scale_max(self, device):
        if self.method.style == QuantizationStyle.PerChannel:
            return self.collect_datas[0] / self.method.bitbound #从列表中取数据。除以127 得到scale
        else: # pertensor是直接把列表中多个batch的最大值求个平均？？，再
            return (torch.tensor(self.collect_datas, dtype=torch.float32).mean() / self.method.bitbound).to(device)

    def collect_histogram(self, x):
        if self.method.style == QuantizationStyle.PerChannel:
            raise NotImplementedError("Not implemented")
        else:
            # hist, edges = torch.histogram(x.abs(), self.method.algo.bins)
            # self.collect_datas.append()
            raise NotImplementedError("Not implemented")

    def compute_scale_histogram(self, device):
        if self.method.style == QuantizationStyle.PerChannel:
            raise NotImplementedError("Not implemented")
        else: 
            raise NotImplementedError("Not implemented")
            # print(self.collect_datas[0])
            # return (torch.tensor(self.collect_datas, dtype=torch.float32).mean() / self.method.bitbound).to(device)


class Quantizer(nn.Module):
    def __init__(self, method : QuantizationMethod):
        super().__init__()

        self.enable    = False
        self.collect   = False
        self.method    = method
        self.collector = Collector(method)

    def compute_scale(self):
        self.register_buffer("_scale", torch.empty(1))
        self._scale = self.collector.compute_scale(self._scale.device)

    def forward(self, x):
        if not self.enable:
            return x

        if self.collect:
            self.collector(x)
            return x

        return QuantizerImpl.apply(x, self._scale, self.method.bitbound)

    def __repr__(self):
        scale = None
        if hasattr(self, "_scale"):
            if self.method.style == QuantizationStyle.PerChannel:
                scale = f"scale=[min={self._scale.min().item()}, max={self._scale.max().item()}]"
            else:
                scale = f"scale={self._scale.item()}"
        return f"Quantizer({scale}, enable={self.enable}, collect={self.collect}, method={self.method})"


class QuantConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_quantizer_  = Quantizer(QuantizationMethod.per_tensor(8, Algorithm("max")))
        self.weight_quantizer_ = Quantizer(QuantizationMethod.per_channel(8, Algorithm("max"), 1))
    
    def forward(self, x):
        x = self.input_quantizer_(x)
        w = self.weight_quantizer_(self.weight)
        return self._conv_forward(x, w, self.bias)


def set_quantizer_state(model: nn.Module, enable, collect):
    for name, m in model.named_modules():
        if isinstance(m, Quantizer):
            m.enable = enable
            m.collect = collect

def compute_scale(model: nn.Module):
    for name, m in model.named_modules():
        if isinstance(m, Quantizer):
            m.compute_scale()

torch.manual_seed(1)
nn.Conv2d = QuantConv2d
model = models.resnet50(pretrained=True).eval()
set_quantizer_state(model, enable=True, collect=True)

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    model(x)

compute_scale(model)
set_quantizer_state(model, enable=True, collect=False)

with torch.no_grad():
    y1 = model(x)

set_quantizer_state(model, enable=False, collect=False)

with torch.no_grad():
    y2 = model(x)

print(model)
print(f"Max absolute error: {(y1 - y2).abs().max():.5f}")
print(y1.argmax(), y2.argmax())