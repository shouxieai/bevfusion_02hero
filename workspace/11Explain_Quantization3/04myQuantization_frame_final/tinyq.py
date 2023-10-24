import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules.pooling import _MaxPoolNd, _AdaptiveAvgPoolNd, _AvgPoolNd
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.autograd import Function
from enum import Enum
from typing import Callable, List, Optional
import numpy as np
import inspect
import os
import warnings

g_logger_verbose = False

def set_verbose(enable=True):
    global g_logger_verbose
    g_logger_verbose = enable

def log(*msg):
    if g_logger_verbose:
        stack = inspect.stack()[1]
        name = os.path.basename(stack.filename)
        msg = " ".join(msg)
        print(f"[{name}:{stack.lineno}]: {msg}")

def obtain_sparsity_mask(weight, N=2, M=4):

    if len(weight.shape) == 2:
        O, I = weight.shape
        weight = weight.detach().reshape(-1, M)
        index  = torch.argsort(weight.abs(), dim=1)[:, :int(M-N)]
        mask = torch.ones(weight.shape, device=weight.device, dtype=weight.dtype)
        return mask.scatter_(dim=1, index=index, value=0).reshape(O, I)

    O, I, H, W = weight.shape
    weight = weight.detach().permute(0, 2, 3, 1).reshape(-1, M)
    index  = torch.argsort(weight.abs(), dim=1)[:, :int(M-N)]

    mask = torch.ones(weight.shape, device=weight.device, dtype=weight.dtype)
    mask = mask.scatter_(dim=1, index=index, value=0).reshape(O, H, W, I)
    return mask.permute(0, 3, 1, 2).contiguous()


class QuantizerOnlyImpl(Function):
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
        return grad, None, None, None


class SparsifyOnlyImpl(Function):
    @staticmethod
    def forward(ctx, x, coeff):
        mask = obtain_sparsity_mask(x)
        ctx.coeff = coeff
        ctx.mask  = mask
        ctx.save_for_backward(x)
        return x * mask

    @staticmethod
    def backward(ctx, grad):
        return grad + ctx.coeff * (1 - ctx.mask) * ctx.saved_tensors[0], None


class SparsifyAndQuantImpl(Function):
    @staticmethod
    def forward(ctx, x, scale, bound, coeff):
        mask = obtain_sparsity_mask(x)
        scaled_x = x / scale
        ctx.bound = bound
        ctx.coeff = coeff
        ctx.save_for_backward(scaled_x, x * (1 - mask))
        return scaled_x.round_().clamp_(-bound, +bound) * scale * mask

    @staticmethod
    def backward(ctx, grad):
        scaled_x, xmask = ctx.saved_tensors
        bound    = ctx.bound
        zero = grad.new_zeros(1)
        grad = torch.where((scaled_x > -bound) | (scaled_x < +bound), grad, zero)
        return grad + ctx.coeff * xmask, None, None, None, None


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
        self.bitbound = int(2**(self.bits - 1) - 1)

    @staticmethod
    def per_tensor(bits:int, algo:Algorithm):
        return QuantizationMethod(bits, algo, 0, QuantizationStyle.PerTensor)
    
    @staticmethod
    def per_channel(bits:int, algo:Algorithm, dim:int, once:bool=True):
        return QuantizationMethod(bits, algo, dim, QuantizationStyle.PerChannel, once)

    def __repr__(self):
        return f"[style={self.style.name}, bits={self.bits}, algo={self.algo}, dim={self.dim}, once={self.once}]"


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

    def compute(self, eps=1 / (1 << 24)):
        if self.method.style == QuantizationStyle.PerChannel:
            return (self.collect_datas[0] / self.method.bitbound).clamp(eps)
        else:
            return (torch.stack(self.collect_datas).mean() / self.method.bitbound).clamp(eps)


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

    def compute(self, eps=1 / (1 << 24)):
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
        return (centers[index] / self.method.bitbound).clamp(eps)


class Quantizer(nn.Module):
    def __init__(self, method : QuantizationMethod):
        super().__init__()

        self.enable   = True
        self.collect  = False
        self.method   = method
        self.use_torch_quantize = False
        self.sparsity = False
        self.quant    = True
        self.sparsity_coeff = 2e-4

    @property
    def collect(self):
        return self._collect

    @collect.setter
    def collect(self, new_value):
        self._collect = new_value
        if new_value:
            if self.method.algo.name == "max":
                self.collector = CalibMax(self.method)
            elif self.method.algo.name == "histogram":
                self.collector = CalibHistogram(self.method)
        else:
            self.collector = None

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.register_buffer("_scale", state_dict[f"{prefix}_scale"].cuda())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

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
            assert x.dtype == torch.float32, f"Failed to export onnx for {x.dtype}(not supported). Please convert the model to float32 first."
            if self.method.style == QuantizationStyle.PerTensor:
                return torch.fake_quantize_per_tensor_affine(x, self._scale.item(), 0, -self.method.bitbound - 1, self.method.bitbound)
            elif self.method.style == QuantizationStyle.PerChannel:
                scale_sequeeze = self._scale.view(self._scale.numel())
                return torch.fake_quantize_per_channel_affine(
                    x, scale_sequeeze, scale_sequeeze.new_zeros(scale_sequeeze.shape, dtype=torch.int32), self.method.dim, -self.method.bitbound - 1, self.method.bitbound)

        if self.quant and self.sparsity:
            return SparsifyAndQuantImpl.apply(x, self._scale.to(x.dtype), self.method.bitbound, self.sparsity_coeff)
        elif self.quant:
            return QuantizerOnlyImpl.apply(x, self._scale.to(x.dtype), self.method.bitbound)
        elif self.sparsity:
            return SparsifyOnlyImpl.apply(x, self.sparsity_coeff)
        else:
            return x

    def __repr__(self):
        scale = None
        if hasattr(self, "_scale"):
            if self.method.style == QuantizationStyle.PerChannel:
                scale = f"scale=[min={self._scale.min().item()}, max={self._scale.max().item()}]"
            else:
                scale = f"scale={self._scale.item()}"
        return f"Quantizer({scale}, enable={self.enable}, collect={self.collect}, method={self.method}, quant={self.quant}, sparsity={self.sparsity})"


def enable_model_sparsity(self, model):
    def enable_sparsity(name, module):
        if isinstance(module, QTypeInputAndWeight):
            shape_str = " x ".join(list(map(str, module.weight.shape)))
            if module.weight.size(1) % 4 != 0:
                log(f"Ingore sparsity for {name}, {shape_str}, due to m.weight.size(1)[{int(module.weight.size(1))} % 4 != 0]")
                return

            shape = list(module.weight.shape)
            if len(shape) == 2:
                shape += [1, 1]

            RS = np.prod(shape[2:])
            if RS > 32:
                log(f"Ingore sparsity for {name}, {shape_str} due to RS [{RS}] > 32")
                return

            CRS = np.prod(shape[1:])
            if RS > 1 and CRS < 512 or RS == 1 and CRS < 4096:
                log(f"Ingore sparsity for {name}, {shape_str} due to RS[{RS}] > 1 and CRS[{CRS}] < 512 or RS == 1 and CRS < 4096")
                return

            log(f"Enable sparsity: {name}, {shape_str}")
            module.weight_quantizer_.sparsity = True

    enable_sparsity("", model)
    for name, m in model.named_modules():
        enable_sparsity(name, m)

    return self

class linker(object):
    def __init__(self, model:nn.Module):
        self.model = model

    def apply(self, fn):
        if isinstance(self.model, Quantizer):
            fn(self.model)

        for name, m in self.model.named_modules():
            if isinstance(m, Quantizer):
                fn(m)
        return self

    def __getattribute__(self, name: str):
        if name == "model":     return object.__getattribute__(self, "model")

        apply = object.__getattribute__(self, "apply")
        if name == "enable":    return apply(lambda m: setattr(m, "enable", True))
        if name == "disable":   return apply(lambda m: setattr(m, "enable", False))
        if name == "collect":   return apply(lambda m: setattr(m, "collect", True))
        if name == "uncollect": return apply(lambda m: setattr(m, "collect", False))
        if name == "compute":   return apply(lambda m: m.compute())
        if name == "export":    return apply(lambda m: setattr(m, "use_torch_quantize", True))
        if name == "unexport":  return apply(lambda m: setattr(m, "use_torch_quantize", False))
        if name == "sparsity":    return enable_model_sparsity(self, self.model)
        if name == "unsparsity":   return apply(lambda m: setattr(m, "sparsity", False))
        if name == "quant":    return apply(lambda m: setattr(m, "quant", True))
        if name == "unquant":   return apply(lambda m: setattr(m, "quant", False))
        raise AttributeError(f"Can not found attribute: {name}")


class collect(torch.no_grad):
    def __init__(self, model):
        super().__init__()
        self.linker = linker(model)
        
    def __enter__(self):
        super().__enter__()
        self.prev_training = self.linker.model.training
        self.linker.model.eval()
        self.linker.enable.collect
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_value is not None:
            raise exc_value
        
        super().__exit__(exc_type, exc_value, exc_traceback)
        self.linker.compute.uncollect

        if self.prev_training:
            self.linker.model.train()


class early:
    def __init__(self, dataloader, num_iter):
        self.dataloader = dataloader
        self.num_iter = min(num_iter, len(dataloader))
    
    def __len__(self):
        return self.num_iter
        
    def __iter__(self):
        for i, obj in enumerate(self.dataloader):
            yield obj

            if i+1 >= self.num_iter:
                break
            

class will_export:
    def __init__(self, model):
        self.linker = linker(model)

    def __enter__(self):
        self.linker.export
        warnings.filterwarnings("ignore")
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_value is not None:
            raise exc_value
        
        self.linker.unexport


PER_TENSOR_HISTOGRAM_8BITS = QuantizationMethod.per_tensor(8, Algorithm("histogram", bins=2048))
PER_CHANNEL_MAX_8BITS      = QuantizationMethod.per_channel(8, Algorithm("max"), 0)

class QTypeInputAndWeight:
    def init_quantizer(self):
        self.input_quantizer_  = Quantizer(PER_TENSOR_HISTOGRAM_8BITS)
        self.weight_quantizer_ = Quantizer(PER_CHANNEL_MAX_8BITS)

class QTypeInputOnly:
    def init_quantizer(self):
        self.input_quantizer_  = Quantizer(PER_TENSOR_HISTOGRAM_8BITS)

class QuantConvNd(_ConvNd, QTypeInputAndWeight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_quantizer()

class QuantConv1d(QuantConvNd, nn.Conv1d):
    def forward(self, x):
        return self._conv_forward(self.input_quantizer_(x), self.weight_quantizer_(self.weight), self.bias)

class QuantConv2d(QuantConvNd, nn.Conv2d):
    def forward(self, x):
        return self._conv_forward(self.input_quantizer_(x), self.weight_quantizer_(self.weight), self.bias)

class QuantConv3d(QuantConvNd, nn.Conv3d):
    def forward(self, x):
        return self._conv_forward(self.input_quantizer_(x), self.weight_quantizer_(self.weight), self.bias)

class QuantConvTransposeNd(_ConvTransposeNd, QTypeInputAndWeight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_quantizer()
    
    def _conv_forward(self, fn, x, output_size: Optional[List[int]] = None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        x = self.input_quantizer_(x)
        w = self.weight_quantizer_(self.weight)
        output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)  # type: ignore[arg-type]
        return fn(x, w, self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)

class QuantConvTranspose1d(QuantConvTransposeNd, nn.ConvTranspose1d):
    def forward(self, x, output_size: Optional[List[int]] = None):
        return super()._conv_forward(F.conv_transpose1d, x, output_size)

class QuantConvTranspose2d(QuantConvTransposeNd, nn.ConvTranspose2d):
    def forward(self, x, output_size: Optional[List[int]] = None):
        return super()._conv_forward(F.conv_transpose2d, x, output_size)

class QuantConvTranspose3d(QuantConvTransposeNd, nn.ConvTranspose3d):
    def forward(self, x, output_size: Optional[List[int]] = None):
        return super()._conv_forward(F.conv_transpose3d, x, output_size)

class QuantLinear(nn.Linear, QTypeInputAndWeight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_quantizer()
    
    def forward(self, x):
        x = self.input_quantizer_(x)
        w = self.weight_quantizer_(self.weight)
        return F.linear(x, w, self.bias)

class QuantAdd(nn.Module, QTypeInputOnly):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_quantizer()
    
    def forward(self, a, b):
        return torch.add(self.input_quantizer_(a), self.input_quantizer_(b))

class QuantAdaptiveAvgPoolNd(_AdaptiveAvgPoolNd, QTypeInputOnly):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_quantizer()

class QuantAdaptiveAvgPool1d(QuantAdaptiveAvgPoolNd, nn.AdaptiveAvgPool1d):
    def forward(self, x):
        return F.adaptive_avg_pool1d(self.input_quantizer_(x), self.output_size)
    
class QuantAdaptiveAvgPool2d(QuantAdaptiveAvgPoolNd, nn.AdaptiveAvgPool2d):
    def forward(self, x):
        return F.adaptive_avg_pool2d(self.input_quantizer_(x), self.output_size)

class QuantAdaptiveAvgPool3d(QuantAdaptiveAvgPoolNd, nn.AdaptiveAvgPool3d):
    def forward(self, x):
        return F.adaptive_avg_pool3d(self.input_quantizer_(x), self.output_size) 


class QuantMaxPoolNd(_MaxPoolNd, QTypeInputOnly):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_quantizer()
    
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
    

class QuantAvgPoolNd(_AvgPoolNd, QTypeInputOnly):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_quantizer()

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
    (torch.nn, "Conv1d", QuantConv1d),
    (torch.nn, "Conv2d", QuantConv2d),
    (torch.nn, "Conv3d", QuantConv3d),
    (torch.nn, "ConvTranspose1d", QuantConvTranspose1d),
    (torch.nn, "ConvTranspose2d", QuantConvTranspose2d),
    (torch.nn, "ConvTranspose3d", QuantConvTranspose3d),
    (torch.nn, "Linear", QuantLinear),
    (torch.nn, "AdaptiveAvgPool1d", QuantAdaptiveAvgPool1d),
    (torch.nn, "AdaptiveAvgPool2d", QuantAdaptiveAvgPool2d),
    (torch.nn, "AdaptiveAvgPool3d", QuantAdaptiveAvgPool3d),
    (torch.nn, "AvgPool1d", QuantAvgPool1d),
    (torch.nn, "AvgPool2d", QuantAvgPool2d),
    (torch.nn, "AvgPool3d", QuantAvgPool3d),
]

def add_replace_module(module, name, target):
    global modules_map
    modules_map.append((module, name, target))


class modules_replacement:
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
        

def replace_modules(model: nn.Module, ignore_proxy: [Callable, List[str]]=None):

    preper_modules_map = []
    for m, name, target in modules_map:
        preper_modules_map.append([getattr(m, name), target])

    select_modules = []
    for target, target_module in model.named_modules():
        if ignore_proxy is not None:
            if isinstance(ignore_proxy, callable):
                if ignore_proxy(target):
                    continue
            elif isinstance(ignore_proxy, list) or isinstance(ignore_proxy, tuple) or isinstance(ignore_proxy, set) or isinstance(ignore_proxy, dict):
                if target in ignore_proxy:
                    continue
            else:
                raise NotImplementedError(f"Unsupport ignore proxy {ignore_proxy}")

        for old_cls, new_cls in preper_modules_map:
            if isinstance(target_module, old_cls):
                select_modules.append([target_module, target, new_cls])
                break

    for target_module, target, new_cls in select_modules:
        quant_module = new_cls.__new__(new_cls)
        for k, val in vars(target_module).items():
            setattr(quant_module, k, val)
        
        quant_module.init_quantizer()
        atoms = target.split(".")
        parent = model.get_submodule(".".join(atoms[:-1]))
        item  = atoms[-1]
        setattr(parent, item, quant_module)
    return model