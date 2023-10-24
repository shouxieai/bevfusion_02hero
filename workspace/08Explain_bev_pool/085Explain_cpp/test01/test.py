import torch
import ext_modules_name
# import test01_name.aaa as ext_modules_name
# from test01_name import aaa as ext_modules_name
# import bbb
from test01 import bbb

input = torch.ones(2)

res = bbb.cpp_test_func1(input)
print(res)