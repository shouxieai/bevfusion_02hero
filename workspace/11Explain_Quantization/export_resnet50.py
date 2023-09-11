import torch
import torchvision.models as models

model = models.resnet50()
input = torch.randn(1, 3, 224, 224)

torch.onnx.export(model, input, "resnet50.onnx")
