# 前提：要复制到MIT bevfusion的目录下。才能工作
from mmdet3d.ops.bev_pool import bev_pool_ext
from mmdet3d.ops.bev_pool.bev_pool import QuickCumsum, QuickCumsumCuda
import time

if __name__ == "__main__":
    import torch
    import pickle
    
    """
    x.pkl，是第一个sample的，QuickCumsumCuda中导出的x的数据
    geom_feats.pkl，是第一个sample的，QuickCumsumCuda中导出的x的数据
    ranks.pkl，是第一个sample的，QuickCumsumCuda中导出的x的数据
    """
    with open("x.pkl", "rb") as f:
        content = f.read()
        x = pickle.loads(content)
    x = x.to("cuda:4")
    # x.to(torch.float64).to("cuda:4") 作者的写法不能使用float64
    print(x.shape)

    with open("geom_feats.pkl", "rb") as f:
        content = f.read()
        geom_feats = pickle.loads(content)
    geom_feats = geom_feats.to("cuda:4")
    print(geom_feats.shape)

    with open("ranks.pkl", "rb") as f:
        content = f.read()
        ranks = pickle.loads(content)
    ranks = ranks.to("cuda:4")
    print(ranks.shape)
    
    t = time.time()
    res = QuickCumsum.apply(x, geom_feats, ranks)
    torch.cuda.synchronize()
    print(f"QuickCumsum花费时间:{time.time() - t}s")
    
    t = time.time()
    res = QuickCumsumCuda.apply(x, geom_feats, ranks, 1, 1, 360, 360)
    torch.cuda.synchronize()
    print(f"QuickCumsumCuda花费时间:{time.time() - t}s")
    
    
    
    