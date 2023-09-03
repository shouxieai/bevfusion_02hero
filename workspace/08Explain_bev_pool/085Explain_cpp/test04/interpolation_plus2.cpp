#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA forward declarations
torch::Tensor trilinear_fw_cu2(
    const torch::Tensor feats,
    const torch::Tensor points
);

//2.10新增函数声明
torch::Tensor trilinear_bw_cu(
    const torch::Tensor dL_dfeat_interp,
    const torch::Tensor feats,
    const torch::Tensor points
);

// C++ interface
/*
1.说一下输入输出的形状：
    feats (N,8,F)  输入的是N个框，每个框8个角点， 每个点有F个特征
    points(N,3)     点的坐标，
    feat_interp(N,F) 中心点，是8个角点的三线性内插值。
*/
torch::Tensor trilinear_interpolation_fw2(
    torch::Tensor feats,
    torch::Tensor points
){
    CHECK_INPUT(feats);  //尽量每个参数输入的时候，都要检查下。
    CHECK_INPUT(points);
    return trilinear_fw_cu2(feats, points);
}

//2.11 新增函数，调用cuda函数
torch::Tensor trilinear_interpolation_bw(
    torch::Tensor dL_dfeat_interp, 
    torch::Tensor feats,
    torch::Tensor points
){
    CHECK_INPUT(dL_dfeat_interp); 
    CHECK_INPUT(feats);  
    CHECK_INPUT(points);
    return trilinear_bw_cu(dL_dfeat_interp, feats, points);
}


//绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("trilinear_interpolation_fw2", &trilinear_interpolation_fw2);
    m.def("trilinear_interpolation_bw", &trilinear_interpolation_bw); // 2.12 新增绑定
}