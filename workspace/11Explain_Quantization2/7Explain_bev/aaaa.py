import torch
import torch.nn as nn
import pytorch_quantization.nn as quant_nn
import torchvision.models as models
import quantize
from pytorch_quantization import quant_modules
import debugpy
# 保证host和端口一致，listen可以只设置端口，则为localhost,否则设置成(host,port)
debugpy.listen(12347)
print('wait debugger')
debugpy.wait_for_client()
print('Debugger Attached')

class myModel2(nn.Module):
    def __init__(self):
        super(myModel2, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 3)

    def forward(self, x):
        return self.conv1(x)

model2 = myModel2()

quantize.replace_to_quantization_module(model2)
quantize.set_quantizer_fast(model2)

torch.manual_seed(0)
a = torch.randn(1, 3, 224, 224)

images = [a, a, a, a]


input1 = torch.randn((1, 3, 224, 224)).to(torch.float32).to("cuda")
model2.to(torch.float32).to("cuda")
# model2.eval()
quant_nn.TensorQuantizer.use_fb_fake_quant = True
with torch.no_grad():
    torch.onnx.export(model2, input1, "aaaa.onnx", 
                      verbose=True, 
                      input_names=["input"], 
                      output_names=["output"], 
                      opset_version=13, 
                      do_constant_folding=False)
quant_nn.TensorQuantizer.use_fb_fake_quant = False