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
import torch.nn.functional as F

#1.0 官方案例，稍加修改，演示体素坐标计算。
# gen = PointToVoxel(
#     vsize_xyz=[0.1, 0.1, 0.1], 
#     coors_range_xyz=[-80, -80, -2, 80, 80, 2], 
#     num_point_features=3, 
#     max_num_voxels=5000, 
#     max_num_points_per_voxel=1)
# pc = np.random.uniform(-10, 10, size=[1, 3])
# pc_th = torch.from_numpy(pc)
# voxels, coords, num_points_per_voxel = gen(pc_th, empty_mean=True)
# print("1")
# ''' # 公式 pc / torch.tensor([0.1, 0.1, 0.1]) + torch.tensor([800, 800, 20]) 
# tensor([[[ 9.5872, -9.5767, -0.3381]]])
# tensor([[ 16, 704, 895]], dtype=torch.int32)
# torch.Size([1, 1, 3])
# array([[ 9.58723467, -9.576721  , -0.33808557]])
# tensor([[895.8723, 704.2328,  16.6191]], dtype=torch.float64)
# '''




# 2.0 把boardmix上我们的例子，想象成3个点，如何变成4*4*1的格子
import torch
import numpy as np
from spconv.pytorch.utils import PointToVoxel, gather_features_by_pc_voxel_id

# # Initialize PointToVoxel
# gen = PointToVoxel(
#     vsize_xyz=[1, 1, 1],
#     coors_range_xyz=[0, 0, 0, 4, 4, 1], # 格子范围 # 4*4*1
#     num_point_features=4, #需要与点对应 X,Y,Z,value
#     max_num_voxels=20,
#     max_num_points_per_voxel=1
# )

# # Initialize point cloud
# # Here, the x, y, and z coordinates are taken from the target coords.
# # The feature values 5, 6, and 10 are added as an additional dimension.
# pc = np.array([
#     [0, 0, 0, 1],
#     [1, 1, 0, 2],
#     [3, 3, 0, 3]
# ], dtype=np.float32)



gen = PointToVoxel(
    vsize_xyz=[1, 1, 2],
    coors_range_xyz=[-2, -2, -1, 2, 2, 1], # 格子范围 # 4*4*1
    num_point_features=4, #需要与点对应 X,Y,Z,value
    max_num_voxels=20,
    max_num_points_per_voxel=1
)
pc = np.array([
    [-2, -2, 0, 1],
    [-1, -1, 0, 2],
    [1, 1, 0, 3]
], dtype=np.float32)


# Convert to PyTorch tensor
pc_th = torch.from_numpy(pc)

# Generate voxels and coords
voxels, coords, num_points_per_voxel = gen(pc_th, empty_mean=False)

print("Voxels (Features): ", voxels)
print("Coords: ", coords) # 生成的值是ZYX

# ZYX -> XYZ
coords = torch.flip(coords, [1])
# 手动增加batch_size
coords = F.pad(coords, (1, 0), mode="constant", value=0)

# voxels 对1这个维度求均值，即max_num_points_per_voxel这个维度求均值
# 对应代码158行
voxels = voxels.mean(dim=1)

print("Voxels (Features): ", voxels.shape)
print("Coords: ", coords) # # 生成的值是ZYX
x_sp = spconv.SparseConvTensor(voxels[:, [3]], coords, torch.tensor([4, 4, 1]), batch_size=1)

# 变成密集型
print(x_sp.dense())
print(x_sp.dense().shape)

print(x_sp.dense().squeeze_())

