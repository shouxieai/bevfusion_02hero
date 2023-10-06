import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)

from pytorch_quantization import nn as quant_nn
