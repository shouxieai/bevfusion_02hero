import torch
from pytorch_quantization import tensor_quant

torch.manual_seed(123456)
x = torch.rand(10)
# 模拟对称量化后反量化的结果
fake_x = tensor_quant.fake_tensor_quant(x, x.abs().max())
print(x)
print(fake_x)
int8_x, scale = tensor_quant.tensor_quant(x, x.abs().max())
print(f"scale: {scale}\nint8_x: {int8_x}")
