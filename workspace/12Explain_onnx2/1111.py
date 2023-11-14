import torch.nn as nn
import torch
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.relu = nn.ReLU
        self.conv2 = nn.Conv2d(3, 2, 3)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x
    

def hook_forward(fn):
    fnname_list = fn.split(".")
    layer_name = eval(".".join(fnname_list[:-1])) # torch.nn.Conv2d
    fn_name = fnname_list[-1] # "forward"
    oldfn = getattr(layer_name, fn_name)
    
    def make_hook(bind_func):
        def myforward(self, x):
            y = oldfn(self, x)
            bind_func(self, x, y)
            return y
        
        setattr(layer_name, fn_name, myforward)
        return myforward
        
    return make_hook
    
   
@hook_forward("torch.nn.Conv2d.forward")
def symbolic_conv2d(self, x, y):
    print(f"==[Hook]==\t type:{type(self)} \t input_id:{id(x)} \t output_id:{id(y)} {x.shape}, {y.shape}")

model = Model().eval()
input1 = torch.zeros(1, 1, 5, 5)
a = model(input1)
print(a)
