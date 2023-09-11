import torch
import torchvision
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn

# 初始化量化模块，用于替换模型中的模块
quant_modules.initialize()

model = torchvision.models.resnet50()
# model.cuda()

inputs = torch.randn(2, 3, 224, 224)
quant_nn.TensorQuantizer.use_fb_fake_quant = True
torch.onnx.export(model, inputs, "test_quant_resnet50.onnx", opset_version=13)
