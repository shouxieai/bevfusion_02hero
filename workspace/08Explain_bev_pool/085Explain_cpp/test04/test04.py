import torch
import ext_modules_name2
import time

"""
代码主要讲解torch.autograd.Function包装
"""


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


# 重点在这个函数的书写
class Trilinear_interpolation_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, points): # 1. 两个输入。 ctx是上下文 。算法假设所有points是固定的不会变。所以points不需求导
        feat_interp = ext_modules_name2.trilinear_interpolation_fw2(feats, points) # 2. 得到前向结果
        ctx.save_for_backward(feats, points) # 3. 两个输入保存到上下文中。
        return feat_interp # 4. 返回前向的结果
    
    @staticmethod
    def backward(ctx, grad_feat_interp): # 5.输入默认有loss对feat_interp的导数，作为已知参数
        feats, points = ctx.saved_tesnsors # 6.取出值
        grad_feats = ext_modules_name2.trilinear_interpolation_bw(grad_feat_interp, feats, points)
        return grad_feats, None # 7 数量forward的输入对应
    
    

if __name__ == "__main__":
    N = 65536
    F = 256
    feats = torch.rand(N, 8, F, device='cuda')
    points = torch.rand(N, 3, device='cuda') * 2 - 1 

    feats2 = feats.clone()
    points2 = points.clone()

    t = time.time()
    out_cuda = Trilinear_interpolation_cuda(feats, points) # 8.注意调用方法的方式
    torch.cuda.synchronize()
    print(f"cuda fw time : {time.time() - t}")


    t = time.time()
    out_py = trilinear_interpolation_py(feats2, points2)
    torch.cuda.synchronize()
    print(f"python fw time : {time.time() - t}")

    print(torch.allclose(out_cuda, out_py)) 

    print(out_cuda[0, :10])
    print(out_py[0, :10])

    #==================以上是完整的结果。===============================

    # 下方假设python写的是对的。所以以python为基准，验证Trilinear_interpolation_cuda写的对不对
    t = time.time()
    loss = out_py.sum()
    loss.backward()
    torch.cuda.synchronize()
    print(f"python bw time : {time.time() - t}")
    
    t = time.time()
    loss2 = out_cuda.sum()
    loss2.backward()
    torch.cuda.synchronize()
    print(f"cuda bw time : {time.time() - t}")
    
    print("二者一样") if torch.allclose(out_cuda, out_py) else print("不一样")
    
    