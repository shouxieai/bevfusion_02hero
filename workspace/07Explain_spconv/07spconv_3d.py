from spconv.pytorch.utils import PointToVoxel, gather_features_by_pc_voxel_id
# import spconv.pytorch as spconv
# from spconv.pytorch import functional as Fsp
# from torch import nn
# from spconv.pytorch.utils import PointToVoxel
# from spconv.pytorch.hash import HashTable

import spconv.pytorch as spconv
from torch import nn
import torch
import numpy as np

# gen = PointToVoxel(
#     vsize_xyz=[0.1, 0.1, 0.1], 
#     coors_range_xyz=[-80, -80, -2, 80, 80, 6], 
#     num_point_features=3, 
#     max_num_voxels=5000, 
#     max_num_points_per_voxel=5)
# pc = np.random.uniform(-10, 10, size=[1000, 3])
# pc_th = torch.from_numpy(pc)
# voxels, coords, num_points_per_voxel = gen(pc_th, empty_mean=True)

gen = PointToVoxel(
    vsize_xyz=[1, 1, 2], 
    coors_range_xyz=[-2, -2, -1, 2, 2, 1],  # 4*4*1
    num_point_features=3, 
    max_num_voxels=20, 
    max_num_points_per_voxel=5)
pc = np.array([[0, 0, 0], [0, 1, 1], [0, 3, 3]], dtype=np.float64)
pc_th = torch.from_numpy(pc)
voxels, coords, num_points_per_voxel = gen(pc_th, empty_mean=True)

print()

# class ExampleNet(nn.Module):
#     def __init__(self, shape):
#         super().__init__()
#         self.net = spconv.SparseSequential(
#             spconv.SparseConv3d(32, 64, 3, 2, indice_key="cp0"),
#             # spconv.SparseInverseConv3d(64, 32, 3, indice_key="cp0"), # need provide kernel size to create weight
#         )
#         self.shape = shape

#     def forward(self, features, coors, batch_size):
#         coors = coors.int()
#         x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
#         return self.net(x)