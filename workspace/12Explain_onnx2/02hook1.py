"""
module.register_forward_hook 是 PyTorch 中用于注册前向传播钩子(hook)的方法
hook方法是在模型的前向传播过程中的不同层或模块上执行的用户自定义函数
允许在模型运行时访问和操作中间输出
这对于模型解释、特征提取和其他任务非常有用
"""
import torch.nn as nn
import torch

# 目的：介绍hook，将模型中conv2的输入输出用hook打印出来
# 1. 定义模型
class myModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = self.conv1(x)
        print(f"[1]\t origin forward conv2 input.shape={x.shape}") # origin forward conv2 input.shape=torch.Size([1, 20, 28, 28])
        x = self.conv2(x)
        print(f"[2]\t origin forward conv2 output.shape={x.shape}") # origin forward conv2 output.shape=torch.Size([1, 20, 24, 24])
        return x

# 2.实例化模型
model = myModule()

# 3.准备数据
input = torch.ones(1, 1, 32, 32)

# 4.直接打印输出结果
res1 = model(input)
print(f"[3]\t forward origin res1 ={res1.shape}")

print("==================================================================")
# 5.写钩子函数
def before_hook(module, x, output):
    print(f"[hook]\t x input shape is \t\t{x[0].shape}") # 
    print(f"[hook]\t output shape is \t\t{output.shape}")
    print(f"[hook]\t 当前的module是\t\t{module}")
    print(f"[hook]\t 当前的module.__class__.__name__是\t\t{module.__class__.__name__}")
    return torch.ones(1, 1, 1, 1)

# 6.注册到self.conv2中
hook_handle = model.conv2.register_forward_hook(before_hook)

# 7.重新把input放入网络
res2 = model(input)
print(f"[4]\t forward origin res1 ={res2.shape}")

# 8.重点：注意删除钩子
hook_handle.remove()

"""
[1]      origin forward conv2 input.shape=torch.Size([1, 20, 28, 28])
[2]      origin forward conv2 output.shape=torch.Size([1, 20, 24, 24])
[3]      forward origin res1 =torch.Size([1, 20, 24, 24])
==================================================================
[1]      origin forward conv2 input.shape=torch.Size([1, 20, 28, 28])
[hook]   x input shape is               torch.Size([1, 20, 28, 28])
[hook]   output shape is                torch.Size([1, 20, 24, 24])
[hook]   当前的module是         Conv2d(20, 20, kernel_size=(5, 5), stride=(1, 1))
[hook]   当前的module.__class__.__name__是              Conv2d
[2]      origin forward conv2 output.shape=torch.Size([1, 1, 1, 1])
[4]      forward origin res1 =torch.Size([1, 1, 1, 1])
"""