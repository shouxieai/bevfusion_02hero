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
    
    
    #==========测试两者输出是否相同===不相同=========
    # res_lss = QuickCumsum.apply(x, geom_feats, ranks)
    # res_bev = QuickCumsumCuda.apply(x, geom_feats, ranks, 1, 1, 360, 360)
    # print(res_lss[0].shape)
    # print(res_bev.shape)
    
    
    #======梯度检查=======================
    # x.requires_grad = True

    # # 不设置requires_grad=True，因为geom_feats和ranks不需要梯度。
    # x = x.to(torch.double)
    # geom_feats = geom_feats.to(torch.int64) # keep it as int64 as it's the original dtype
    # ranks = ranks.to(torch.int64) # keep it as int64

    # from torch.autograd import gradcheck

    # # 使用.apply来调用
    # res = gradcheck(QuickCumsum.apply, (x[:30], geom_feats[:30], ranks[:30]), eps=1e-3)
    # print(res)
    
    
    # =====极简版测试===
    # t = time.time()
    # res = QuickCumsum.apply(x, geom_feats, ranks)
    # torch.cuda.synchronize()
    # print(f"QuickCumsum花费时间:{time.time() - t}s")
    
    # t = time.time()
    # res = QuickCumsumCuda.apply(x, geom_feats, ranks, 1, 1, 360, 360)
    # torch.cuda.synchronize()
    # print(f"QuickCumsumCuda花费时间:{time.time() - t}s")
    
    
    # ======修改版测试=========
    # num_iterations = 10
    # warmup_iterations = 10
    # from tqdm import tqdm
    
    # # 预热 QuickCumsum
    # for _ in tqdm(range(warmup_iterations), desc="预热 QuickCumsum"):
    #     _ = QuickCumsum.apply(x, geom_feats, ranks)
    #     torch.cuda.synchronize()

    # # 测试 QuickCumsum
    # start_time = time.time()
    # for _ in tqdm(range(num_iterations), desc="测试 QuickCumsum"):
    #     _ = QuickCumsum.apply(x, geom_feats, ranks)
    #     torch.cuda.synchronize()
    # end_time = time.time()
    # avg_time_quickcumsum = (end_time - start_time) / num_iterations
    # print(f"QuickCumsum 平均花费时间: {avg_time_quickcumsum:.6f}s")

    # # 预热 QuickCumsumCuda
    # for _ in tqdm(range(warmup_iterations), desc="预热 QuickCumsumCuda"):
    #     _ = QuickCumsumCuda.apply(x, geom_feats, ranks, 1, 1, 360, 360)
    #     torch.cuda.synchronize()

    # # 测试 QuickCumsumCuda
    # start_time = time.time()
    # for _ in tqdm(range(num_iterations), desc="测试 QuickCumsumCuda"):
    #     _ = QuickCumsumCuda.apply(x, geom_feats, ranks, 1, 1, 360, 360)
    #     torch.cuda.synchronize()
    # end_time = time.time()
    # avg_time_quickcumsumcuda = (end_time - start_time) / num_iterations
    # print(f"QuickCumsumCuda 平均花费时间: {avg_time_quickcumsumcuda:.6f}s")

    # torch.cuda.empty_cache()  # 清空 CUDA 缓存

    