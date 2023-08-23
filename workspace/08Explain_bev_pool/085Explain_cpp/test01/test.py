import torch
import my_package_name

input = torch.ones(2)

res = my_package_name.cpp_test_func1(input)
print(res)