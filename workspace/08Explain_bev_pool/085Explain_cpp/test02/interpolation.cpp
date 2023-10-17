#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA forward declarations
torch::Tensor trilinear_fw_cu(
    torch::Tensor feats,
    torch::Tensor points
);

// C++ interface
torch::Tensor trilinear_interpolation_a(
    torch::Tensor feats,
    torch::Tensor points
){
    CHECK_INPUT(feats);  //尽量每个参数输入的时候，都要检查下。
    CHECK_INPUT(points);
    return trilinear_fw_cu(feats, points);
}


//绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("trilinear_interpolation_a", &trilinear_interpolation_a);
}s