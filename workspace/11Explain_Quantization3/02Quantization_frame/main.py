import torch
import torch.nn as nn
import torchvision.models as models
from tiny_quant import QuantReplacement, PTQCollect, QuantizerLinker, ExportQuantONNX

torch.manual_seed(1)
x = torch.randn(1, 3, 224, 224).cuda()

# 方式1：基于修改nn模块的方式
with QuantReplacement():
    model = models.resnet50(pretrained=True).eval().cuda()

# print(model)

with PTQCollect(model):
    with torch.no_grad():
        for i in range(10):
            print(i)
            x = torch.randn(1, 3, 224, 224).cuda()
            model(x)

with torch.no_grad():
    y1 = model(x)

QuantizerLinker(model).disable
with torch.no_grad():
    y2 = model(x)

# print(model)
print(f"Max absolute error: {(y1 - y2).abs().max():.5f}")
print(y1.argmax(), y2.argmax())

print(f"Export onnx...")
with ExportQuantONNX(model):
    torch.onnx.export(model, (x,), "demo.onnx", opset_version=13)