#include <torch/torch.h>
#include <iostream>
 
int main() {
  torch::Tensor tensor = torch::rand({2, 3}); //生成大小为2*3的随机数矩阵
  std::cout << tensor << std::endl;           //标准输出流打印至屏幕
}