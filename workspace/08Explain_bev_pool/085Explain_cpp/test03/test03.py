import torch
import ext_modules_name
import time
# =====体验检查=====
# feats = torch.ones(2)
# points = torch.zeros(2)
# out = ext_modules_name.trilinear_interpolation(feats, points)
# print(out) # RuntimeError: feats must be a CUDA tensor # 这就是检查起作用了

# =====体验跑通======
# feats = torch.ones(2).to("cuda:4")
# points = torch.zeros(2).to("cuda:4")
# out = ext_modules_name.trilinear_interpolation_plus(feats, points)
# print(out) # tensor([1., 1.], device='cuda:4')


#=====体验完整调用三线性插值
# 1. python版本的三线性插值代码
def trilinear_interpolation_py(feats, points):
    """
    Inputs:
        feats:(N, 8, F)
        points:(N, 3) local coordinates in [-1, 1]
        
    Outputs:
        feats_interp:(N, F)
    """
    u = (points[:, 0:1] + 1) / 2
    v = (points[:, 1:2] + 1) / 2
    w = (points[:, 2:3] + 1) / 2
    a = (1 - v) * (1 - w)
    b = (1 - v) * w
    c = v * (1 - w)
    d = 1 - a - b - c
    
    feats_interp = (1 - u) * (a * feats[:, 0] + 
                              b * feats[:, 1] + 
                              c * feats[:, 2] +
                              d * feats[:, 3]) + \
                        u * ( a * feats[:, 4] +
                              b * feats[:, 5] +
                              c * feats[:, 6] +
                              d * feats[:, 7])
                        
    return feats_interp



N = 65536
F = 256
feats = torch.rand(N, 8, F, device='cuda') # N个立方体，每个立方体8个点，每个点是256维特征来描述
points = torch.rand(N, 3, device='cuda') * 2 - 1 # 使得ppints的取值在 -1 到1 之间

feats2 = feats.clone()
points2 = points.clone()

t = time.time()
out_cuda = ext_modules_name.trilinear_interpolation_plus(feats, points) # cuda的结果
# print(out_cuda.shape) # 计算出来的形状 torch.Size([65536, 256])
torch.cuda.synchronize()
print(f"cuda time : {time.time() - t}")


t = time.time()
out_py = trilinear_interpolation_py(feats2, points2) # python函数的值
torch.cuda.synchronize()
print(f"python time : {time.time() - t}")

print(torch.allclose(out_cuda, out_py)) # 默认绝对误差是 1e-8，如果在误差范围内，认为两个值相同。1e-6是相对误差

print(out_cuda[0, :10])
print(out_py[0, :10])

#==================以上是完整的结果。===============================

# ==================以下是测试这种cuda扩展书写，能否自动算grad=======
# 二、如果feats，让他可以自动求导
feats = torch.rand(N, 8, F, device='cuda').requires_grad_()
print(feats.requires_grad)

# 2.1 那么python版本的双线性插值，结果仍然是课自动求导的
out_py = trilinear_interpolation_py(feats, points) # python函数的值
print(out_py.requires_grad) # True
out_cuda = ext_modules_name.trilinear_interpolation_plus(feats, points)
print(out_cuda.requires_grad) # False 说明这种cuda扩展，是没有办法自动算grad的。