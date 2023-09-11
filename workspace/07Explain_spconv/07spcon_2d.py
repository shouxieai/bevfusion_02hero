
from __future__ import print_function
import argparse
import torch

import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import contextlib
import torch.cuda.amp


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = spconv.SparseSequential(
            # nn.BatchNorm1d(1),
            spconv.SubMConv2d(1, 1, 3, 1, 1,bias=False),
            # nn.ReLU(),
            spconv.ToDense()
        )

    def forward(self, x: torch.Tensor):
        x_sp = spconv.SparseConvTensor.from_dense(x.reshape(-1, 4, 4, 1))
        # print(x_sp) # SparseConvTensor[shape=torch.Size([3, 1])]
        x = self.net(x_sp)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = Net()
model.net[0].weight.data.fill_(1) # make kernel value all fill 1
model.to(device) # 较重要

data = torch.tensor([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 3]], dtype=torch.float32).to(device) # type is important
data = data.reshape(1, 4, 4, 1) # batch_size, w,h,c 因为是2d，所以暂时理解为whc(可能不严谨)。 3D的话是batch_size, x,y,z
print(data)
print(data.shape)
print(data.device)

res = model(data)
res = res.to("cpu")
print(res)
print(res.shape)

"""
tensor([[[[3., 0., 0., 0.],
          [0., 3., 0., 0.],
          [0., 0., 0., 0.],
          [0., 0., 0., 3.]]]], grad_fn=<ToCopyBackward0>)
"""