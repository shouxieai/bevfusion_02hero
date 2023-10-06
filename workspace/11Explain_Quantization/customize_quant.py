import torch
import torchvision
from pytorch_quantization import tensor_quant, quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from typing import List, Callable, Union, Dict

# 取消量化
class disable_quantization:
    def __init__(self, model):
        self.model = model
    
    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled
    
    def __enter__(self):
        self.apply(True)
    
    def __exit__(self, *args, **kwargs):
        self.apply(False)

# 应用量化
class enable_quantization:
    def __init__(self, model):
        self.model = model
    
    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled
    
    def __enter__(self):
        self.apply(True)
    
    def __exit__(self, *args, **kwargs):
        self.apply(False)

# 查看量化器信息
def quantizer_state(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            print(name, module)

def transfer_torch_to_quantization(nninstance:torch.nn.Module, quantmodule):
    # 创建一个新的量化模块对象
    quant_instance = quantmodule.__new__(quantmodule)
    # 将原来模块中的属性通过键值对赋值到新的量化模块对象中
    for k, val in vars(nninstance).items():
        setattr(quant_instance, k, val)
    
    # 对量化模块对象中插入量化器
    def __init__(self):
        # 插入输入量化器
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            quant_desc_input = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True)
            self.init_quantizer(quant_desc_input)

            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        # 插入输入量化器权重量化器 (quant_nn_utils.QuantMixin)
        else:
            quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
            self.init_quantizer(quant_desc_input, quant_desc_weight)
            
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True
    
    __init__(quant_instance)
    return quant_instance


def replace_to_quantization_module(model:torch.nn.Module, ignore_policy:Union[str, List[str], Callable] = None):
    module_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)
        # 通过原始的module的id来查找量化后的module
        module_dict[id(module)] = entry.replace_mod
    
    def recursive_and_replace_module(model, prefix=""):
        for name in model._modules:
            submodule = model._modules[name]
            path = name if prefix == "" else prefix + "." + name
            print(path)
            recursive_and_replace_module(submodule, prefix=path)
            
            submodule_id = id(type(submodule))
            if submodule_id in module_dict:
                model._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])
    
    recursive_and_replace_module(model)


model = torchvision.models.resnet50()
model.cuda()
# quantizer_state(model)
replace_to_quantization_module(model)


inputs = torch.randn(1, 3, 224, 224, device='cuda')
quant_nn.TensorQuantizer.use_fb_fake_quant = True
torch.onnx.export(model, inputs, 'cus_quant_resnet50.onnx', opset_version=13)
