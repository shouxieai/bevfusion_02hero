import torch
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor

class QuantMultiAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
    
    def forward(self, x, y, z):
        return self._input_quantizer(x) + self._input_quantizer(y) + self._input_quantizer(z)

class QuantMultiMulAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
        self._weight_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
    
    def forward(self, x, y, z, w_x, w_y, w_z):
        return self._input_quantizer(x) * self._weight_quantizer(w_x) + \
                self._input_quantizer(y) * self._weight_quantizer(w_y) + \
                self._input_quantizer(z) * self._weight_quantizer(w_z)

quant_nn.TensorQuantizer.use_fb_fake_quant = True

model = QuantMultiAdd()
model1 = QuantMultiMulAdd()
# model.cuda()
# model1.cuda()
# inputs_a = torch.randn(1, 3, 224, 224, device='cuda')
# inputs_b = torch.randn(1, 3, 224, 224, device='cuda')
# inputs_c = torch.randn(1, 3, 224, 224, device='cuda')
# w_a = torch.randn(64, 3, 224, 224, device='cuda')
# w_b = torch.randn(64, 3, 224, 224, device='cuda')
# w_c = torch.randn(64, 3, 224, 224, device='cuda')
inputs_b = torch.randn(1, 3, 224, 224)
inputs_a = torch.randn(1, 3, 224, 224)
inputs_c = torch.randn(1, 3, 224, 224)
w_a = torch.randn(64, 3, 224, 224)
w_b = torch.randn(64, 3, 224, 224)
w_c = torch.randn(64, 3, 224, 224)

torch.onnx.export(model, (inputs_a, inputs_b, inputs_c), "multi_add.onnx", opset_version=13, \
    input_names=['x', 'y', 'z'], output_names=['output'])
torch.onnx.export(model1, (inputs_a, inputs_b, inputs_c, w_a, w_b, w_c), "multi_mul_add.onnx", opset_version=13, \
    input_names=['x', 'y', 'z', 'w_x', 'w_y', 'w_z'], output_names=['output'])
