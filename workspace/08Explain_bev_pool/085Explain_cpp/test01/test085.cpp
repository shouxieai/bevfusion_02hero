#include <torch/extension.h>

torch::Tensor cpp_test_func1(    
    torch::Tensor input
){
    torch::Tensor output = input + 1;
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("cpp_test_func1", &cpp_test_func1); 
    //第一个参数是python调用时使用的名字，一般和cpp中函数名一致
    //第二个参数是会调用cpp中哪个函数，给函数名
}