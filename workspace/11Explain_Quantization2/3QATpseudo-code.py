

"""
QAT中是如何使用hook函数的。以及在模型前向的时候如何获取模型层的输出
并且通过伪代码，简单展示整个QAT思想。
"""
#注意！！！以下代码为 伪代码
# 1.0 介绍  
    # 正常权重的更新，需要计算预测值与真实值的loss
    # QAT模型调优的工作中，需要量化后的PTQ模型的输出值 与 模型原始的输出值 二者之间计算loss
        # QAT模型调优的目的就是为了，在模型训练中，让PTQ模型的输出 越来越接近 原始模型的输出。 这样才能更大程度的保证量化前后的模型的精度损失最小。
    
    # 因此在计算loss的过程中，需要PTQ模型的输出值 与 原始模型的输出值
        # 与hook结合，自然而然的想法是用hook，在模型前向的时候获得每一层 PTQ的输出 与 原始模型输出。计算loss，然后loss相加

# 2.0 思路： QAT：PTQ_model, origin_model ===> loss ===> update: PTQ_model

# 2.0.1 获取 PTQ_model, origin_model 网络层/模块  匹配对 layer_pairs
#以下为假定的匹配对的举例
ptq_origin_layer_pairs = [[ptq_layer0, origin_layer0], [ptq_layer1, origin_layer1], [ptq_layer2, origin_layer2], ...]

# 2.0.2 获取 PTQ的输出 与 原始模型输出
ptq_outputs = []
origin_outputs = []

# 2.0.3 定义hook函数。
def make_layer_forward_hook(module_outputs): 
    def forward_hook(module, input, output):# 上节课演示的hook。主因input是个元组。用来捕获每个层的输出。
        module_outputs.append(output) # 将捕获的ptq 或者 原始模型的层的输出，放入module_outputs
        
    return forward_hook # 返回钩子函数

## 2.0.5.1 创建空列表，储存handle，后面统一删除
remove_handle = []

# 2.0.4 注册每一层的hook
for ptq_m, ori_m in ptq_origin_layer_pairs:
    remove_handle.append(ori_m.register_forward_hook(make_layer_forward_hook(origin_outputs)))  # 原始曾注册register_forward_hook的同时。希望所有原始层的结果存在origin_outputs中
    remove_handle.append(ptq_m.register_forward_hook(make_layer_forward_hook(ptq_outputs)))     # ptq层注册register_forward_hook的同时。希望所有ptq层的结果存在origin_outputs中
  
# 2.0.6 ptq模型前向
ptq_model(imgs)

# 2.0.7 原始模型前向
origin_model(imgs)

# 2.0.8 计算ptq与origin的loss
loss = 0.
for index, (ptq_out, ori_out) in enumerate(zip(ptq_outputs, origin_outputs)):
    loss += loss_Function(torch.abs(ptq_out - ori_out)) # 一个假定的loss计算函数

    
# 2.0.5 将注册时返回的handle，删掉
# 2.0.5.2 逐个删掉
for rm in remove_handle:
    rm.remove()
    
    