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
    pass

def replace_to_quantization_module(model:torch.nn.Module, ignore_policy:Union[str, List[str], Callable] = None):
    pass


quant_modules.initialize()
model = torchvision.models.resnet50()
model.cuda()
# quantizer_state(model)

disable_quantization(model.conv1).apply()
enable_quantization(model.conv1).apply()

inputs = torch.randn(1, 3, 224, 224, device='cuda')
quant_nn.TensorQuantizer.use_fb_fake_quant = True
torch.onnx.export(model, inputs, 'quant_resnet50_conv1_enable.onnx', opset_version=13)
