import torch
import ext_modules_name

input = torch.ones(2)

res = ext_modules_name.cpp_test_func1(input)
print(res)