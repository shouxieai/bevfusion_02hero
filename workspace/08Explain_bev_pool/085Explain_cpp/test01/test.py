import torch
# import ext_modules_name
# import test01_name.aaa as ext_modules_name
from test01_name import aaa as ext_modules_name

input = torch.ones(2)

res = ext_modules_name.cpp_test_func1(input)
print(res)