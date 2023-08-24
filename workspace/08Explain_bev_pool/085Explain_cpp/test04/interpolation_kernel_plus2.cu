#include <torch/extension.h>

template <typename scalar_t>
__global__ void trilinear_fw_kernel( 
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp 
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x; 
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    if(n >= feats.size(0) || f >= feats.size(2)) return ; 

    const scalar_t u = (points[n][0]+1)/2; 
    const scalar_t v = (points[n][1]+1)/2; 
    const scalar_t w = (points[n][2]+1)/2; 

    const scalar_t a = (1 - v) * (1 - w);
    const scalar_t b = (1 - v) * w;
    const scalar_t c = v * (1 - w);
    const scalar_t d = 1 - a - b - c; 
    feat_interp[n][f] = (1 - u) * (a * feats[n][0][f] + 
                                    b * feats[n][1][f] +
                                    c * feats[n][2][f] +
                                    d * feats[n][3][f]) +
                                u * (a * feats[n][4][f] +
                                     b * feats[n][5][f] +
                                     c * feats[n][6][f] +
                                     d * feats[n][7][f]);
}

torch::Tensor trilinear_fw_cu2(
    const torch::Tensor feats,
    const torch::Tensor points
){
    const int N = feats.size(0), F = feats.size(2);

    torch::Tensor feat_interp = torch::zeros({N, F}, feats.options()); 
    const dim3 threads(16, 16);
    const dim3 blocks((N + threads.x - 1) / threads.x, (F + threads.y - 1) / threads.y); 
    
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu", ([&] {
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
            );
    }));
    return feat_interp;
}


template <typename scalar_t>
__global__ void trilinear_bw_kernel(  //2.8 函数名修改，如下入参修改，同调用
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dfeat_interp, 
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> dL_dfeats 
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x; 
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    if(n >= feats.size(0) || f >= feats.size(2)) return ; 

    const scalar_t u = (points[n][0]+1)/2; 
    const scalar_t v = (points[n][1]+1)/2; 
    const scalar_t w = (points[n][2]+1)/2; 

    //一、test04主要介绍backward部分
    /*
    每个边长度为1
          f1。----------。f2
            |    | v    |
            |----。     |
            |  u        |
          f3。----------。f4
    
    f = u*v*f4 + (1 - u)*v*f3 +u*(1-v)*f2+(1-u)(1-v)*f1
    下方代码无非是推广到3维
    底下一圈4个点，乘w   上面一圈4点 乘

    1.1 重点：有哪些是需要计算偏微分的。 
    1.2 答案：这里 f1  f2 f3  f4是需要计算的。而uvw是固定的。
            注意：我们计算的是得到中心点的特征，特征由8个点插值而来。
                    而特征是需要梯度更新的。
    1.3 第一节知道，损失函数 L对 forward的输出f的倒数，会出现在backward的入参中。
        所以这里需要自己计算下 f 对于 f1\f2\f3\f4的导数
        f对f1的偏导=(1 - u)(1 - v)
        f对f2的偏导=u * (1 - v)
        f对f3的偏导=(1 - u) * v
        f对f4的偏导=u * v
       
    二、我们来修改这个trilinear_fw_kernel函数，改造为backward函数
    */
    const scalar_t a = (1 - v) * (1 - w);
    const scalar_t b = (1 - v) * w;
    const scalar_t c = v * (1 - w);
    const scalar_t d = 1 - a - b - c; 
    // 2.9 前向的计算不需要了。
    // feat_interp[n][f] = (1 - u) * (a * feats[n][0][f] + 
    //                                 b * feats[n][1][f] +
    //                                 c * feats[n][2][f] +
    //                                 d * feats[n][3][f]) +
    //                             u * (a * feats[n][4][f] +
    //                                  b * feats[n][5][f] +
    //                                  c * feats[n][6][f] +
    //                                  d * feats[n][7][f]);

    // 2.10 新增计算f1 f2 f3 f4导数部分
    /*
    根据前向的式子分析。abcd都是固定值。feat_interp 由 feats计算而来。 feats是变量。
    而feats 形状为{N,8,F}，对feat_interp 有影响的就是feats中的8个定点的F

    根据链式法则
    dL_dfeats = dL_dfeat_interp * dfeat_interp _dfeats
    即
    dL_dfeats[n][0][f] = dL_dfeat_interp[n][f] * (1 - u) * a
    dL_dfeats[n][1][f] = dL_dfeat_interp[n][f] * (1 - u) * b
    dL_dfeats[n][2][f] = dL_dfeat_interp[n][f] * (1 - u) * c
    dL_dfeats[n][3][f] = dL_dfeat_interp[n][f] * (1 - u) * d
    dL_dfeats[n][4][f] = dL_dfeat_interp[n][f] * u * a
    dL_dfeats[n][5][f] = dL_dfeat_interp[n][f] * u * b
    dL_dfeats[n][6][f] = dL_dfeat_interp[n][f] * u * c
    dL_dfeats[n][7][f] = dL_dfeat_interp[n][f] * u * d
    */

    dL_dfeats[n][0][f] = dL_dfeat_interp[n][f] * (1 - u) * a;
    dL_dfeats[n][1][f] = dL_dfeat_interp[n][f] * (1 - u) * b;
    dL_dfeats[n][2][f] = dL_dfeat_interp[n][f] * (1 - u) * c;
    dL_dfeats[n][3][f] = dL_dfeat_interp[n][f] * (1 - u) * d;
    dL_dfeats[n][4][f] = dL_dfeat_interp[n][f] * u * a;
    dL_dfeats[n][5][f] = dL_dfeat_interp[n][f] * u * b;
    dL_dfeats[n][6][f] = dL_dfeat_interp[n][f] * u * c;
    dL_dfeats[n][7][f] = dL_dfeat_interp[n][f] * u * d;
}
//2.1 名字从trilinear_fw_kernel  改为 trilinear_bw_kernel
torch::Tensor trilinear_bw_cu(
    const torch::Tensor dL_dfeat_interp,//2.2 输入会多一个已知值，即Loss对feat_interp的偏微分。
    const torch::Tensor feats,
    const torch::Tensor points
){
    const int N = feats.size(0), F = feats.size(2); 

    torch::Tensor dL_dfeats = torch::zeros({N, 8, F}, feats.options()); 
    //2.3 输出值feat_interp，改为dL_dfeat 
        //即前向时，输出 插值的结果，而反向时，输出的是Loss对feat的导数。
        //同理，输出的东西不一样，形状也不一样。{N,  F} 改为{N， 8， F}

    const dim3 threads(16, 16);

    const dim3 blocks((N + threads.x - 1) / threads.x, (F + threads.y - 1) / threads.y); 
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_bw_cu", ([&] { //2.4 函数名修改为bw
        trilinear_bw_kernel<scalar_t><<<blocks, threads>>>( //注意修改核函数名字。
            dL_dfeat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),//2.5 新增已知参数 形状{N, F}，记住是2维的。求导不改变形状
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dfeats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>() //2.6 最后一个参数从feat_interp 改为 dL_dfeats 注意更改维度为3
            );
    }));

    
    return dL_dfeats;//2.7 更改输出 从feat_interp 改为 dL_dfeats
}