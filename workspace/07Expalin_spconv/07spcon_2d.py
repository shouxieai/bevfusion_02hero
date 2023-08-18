
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
        x = self.net(x_sp)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net()
model.net[0].weight.data.fill_(1)
model.to(device)

data = torch.tensor([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 3]], dtype=torch.float32).to(device)
data = data.reshape(1, 4, 4, 1)

res = model(data)
res = res.to("cpu")
print(res)

"""
tensor([[[[3., 0., 0., 0.],
          [0., 3., 0., 0.],
          [0., 0., 0., 0.],
          [0., 0., 0., 3.]]]], grad_fn=<ToCopyBackward0>)
"""