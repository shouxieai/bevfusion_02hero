import torch
import ext_modules_name_a


# feats = torch.ones(2)
# points = torch.zeros(2)
# out = ext_modules_name.trilinear_interpolation02(feats, points)
# print(out) # RuntimeError: feats must be a CUDA tensor # 这就是检查起作用了


feats = torch.ones(2).to("cuda:4")
points = torch.zeros(2).to("cuda:4")
out = ext_modules_name_a.trilinear_interpolation_a(feats, points)
print(out) # tensor([1., 1.], device='cuda:4')