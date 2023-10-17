"""
该文件是学习ptq的文件。有大量注释
"""

from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn 
from yolov7.models.yolo import Model
import torch
from pytorch_quantization.nn.modules import _utils as quant_nn_utils # 6.4.3中导入
from pytorch_quantization import calib # 6.4.3中导入

# import sys  # 导入sys模块
# sys.setrecursionlimit(26000)  # 将默认的递归深度修改为3000

import debugpy
# 保证host和端口一致，listen可以只设置端口，则为localhost,否则设置成(host,port)
debugpy.listen(12346)
print('wait debugger')
debugpy.wait_for_client()
print('Debugger Attached')

# 1.0 定义函数
def load_yolov7_model(weight, device="cpu"):
    ## 1.1 加载yolov7的模型的权重
    ckpt = torch.load(weight, map_location=device)
    
    ## 1.2 创建新的模型实例
    model = Model(cfg="cfg/training/yolov7.yaml", ch=3, nc=80).to(device)
    
    ## 1.3 加载的模型权重转换为单精度浮点数FP32。
        # 主要是因为权重可能是FP64
    state_dict = ckpt['model'].float().state_dict()
    
    ## 1.4 权重加载到模型中
    model.load_state_dict(state_dict, strict=False) # False不然有的
    return model
 
# 7.手动量化 注释掉 quant_modules.initialize()
from pytorch_quantization.tensor_quant import QuantDescriptor
from absl import logging as quant_logging
def initialize():
    # 知识点一：默认的量化器就是Max。一般给weight使用max。而input一般使用直方图histogram
    # 以下代码，就是只对input，把标定方法改为historgram
    quant_desc_input = QuantDescriptor(calib_method="histogram")
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.MaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.Linear.set_default_quant_desc_input(quant_desc_input)
    quant_logging.set_verbosity(quant_logging.ERROR)
    
#4.1 插入qdq节点
def prepare_model(weight, device):
    # 4.2 使用自动插入QDQ节点的方法
    # quant_modules.initialize() # 走到第7步时，需要注释掉,与下方代码二选一
    initialize() # 7.1使用这个  与上方代码二选一
    
    # 4.3 调用自定义的模型加载函数。加载模型
    model = load_yolov7_model(weight, device)
    model.to(torch.float32).eval()
    
    with torch.no_grad():
        model.fuse() # 合并conv层以及batchNormal层。主要为了提速
    return model # 这样自动插入qdq节点就完成了

# 11.7    
import re
def quantization_ignore_match(ignore_layer, path):
    if ignore_layer is None:
        return False
    if isinstance(ignore_layer, str) or isinstance(ignore_layer, list):
        if isinstance(ignore_layer, str):
            ignore_layer = [ignore_layer] #如果是字符串，把他转换成字符列表
        if path in ignore_layer:
            return True
        for item in ignore_layer: # 正则表达式的方式匹配。
            if re.match(item, path):
                return True
    return False  
   
# 6.3 定义递归查找函数，用于找到torch_module与quant_module的匹配关系  
# 11.5 增加ignore_layer入参
def recursive_and_replace_module(ori_model:torch.nn.Module, module_dict, ignore_layer, prefix=''):
    for name in ori_model._modules: # ori_model._modules是OrderDict类型。  name就是OrderDict的名字
        ## 6.3.1 根据name，得到具体的子模块名字。递归查找
        submodule = ori_model._modules[name]
        path = name if prefix == "" else prefix + "." + name
        recursive_and_replace_module(submodule, module_dict, ignore_layer, prefix=path) # 如果submodule已经是一个nn层了。迭代就会停止
        
        submodule_id = id(type(submodule))
        if submodule_id in module_dict:
            # 11.6 增加ignore_layer判断。如果path是属于ignore_layer里面的。则跳过
            ignored = quantization_ignore_match(ignore_layer, path)
            if ignored:
                print(f"Quantization: {path} has ignored")
                continue
            # 转换
            ori_model._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])
    

# 6.4 定义具体替换函数。将torch的层与quant的层替换
def transfer_torch_to_quantization(nn_instance:torch.nn.Module, quant_module):
    quant_instance = quant_module.__new__(quant_module) # 6.4.1 创建新的quant的实例
    for k, v in vars(nn_instance).items(): # 6.4.2将nn的模块的属性，设置到新的quant_instance实例上
        setattr(quant_instance, k, v)

    #6.4.3 量化器初始化
    def __init__(self): #必须使用这个函数，否则报错
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            quant_desc_input = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True)
            """Pop quant descriptors in kwargs
            If there is no descriptor in kwargs, the default one in quant_cls will be used

            Arguments:
            quant_cls: A class that has default quantization descriptors
            input_only: A boolean. If True, pop quant_desc_input only, not quant_desc_weight. Default false.

            Keyword Arguments:
            quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
                Quantization descriptor of input.
            quant_desc_weight: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
                Quantization descriptor of weight.
            """
            self.init_quantizer(quant_desc_input) # 内部实现self._input_quantizer = TensorQuantizer(quant_desc_input)
            
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True # 主要是为了加速
        else:
            quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
            self.init_quantizer(quant_desc_input, quant_desc_weight)
            
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True # 主要是为了加速
                self._weight_quantizer._calibrator._torch_hist = True # 主要是为了加速

    __init__(quant_instance)
    return quant_instance

# 6.1 手动插入QDQ节点
# 11.3 改造，增加入参ignore_layer。增加忽略层的功能
def replace_to_quantization_model(ori_model, ignore_layer=None):    # 6.2该函数手动替换节点。把conv2d替换为quant_conv2d
    """
    model : 没有插入QDQ节点的模型。
    """
    module_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP: # entry类型<class 'pytorch_quantization.quant_modules.quant_entry'>
        # print(type(entry))
        # print(entry.orig_mod) # <module 'torch.nn' from '/opt/conda/lib/python3.8/site-packages/torch/nn/__init__.py'>
        # print(entry.mod_name) # Conv1d
        # print(entry.replace_mod) # <class 'pytorch_quantization.nn.modules.quant_conv.QuantConv1d'>
        # a = eval("entry.orig_mod" + "." + f"{entry.mod_name}") # <class 'torch.nn.modules.conv.Conv1d'>
        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod # 记录模块id与替换的quant_module
    # 11.4 参数ignore_layer也给recursive_and_replace_module的入参
    recursive_and_replace_module(ori_model, module_dict, ignore_layer) # 使用6.3中定义的函数

    
    
# 2.0 定义函数
import collections
from yolov7.utils.datasets import create_dataloader
def perpare_val_dataset(cocodir, batch_size=4):
    dataloader, _ = create_dataloader( # 返回值是dataloader, dataset
        path=f"{cocodir}/val2017.txt",
        imgsz=640,
        batch_size=batch_size,
        opt=collections.namedtuple("Opt", "single_cls")(False), 
        # create_dataloader内会调用opt.single_cls。我的opt没有single_cls。所以用collections.namedtuple。
            # 快速封装一个。并且立刻实例化，给single_cls赋值False
        augment=False, hyp=None, rect=True, cache=False, stride=32, pad=0.5, image_weights=False
        # 验证数据，不需要增强。 rect是否使用矩形图像。
        )
    return dataloader

import yaml
def perpare_train_dataset(cocodir, batch_size=4):
    # 训练集需要增广
    # 加载增广配置文件
    with open("data/hyp.scratch.p5.yaml") as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    
    dataloader, _ = create_dataloader( 
        path=f"{cocodir}/train2017.txt", # 训练集与测试集区别在这里1
        imgsz=640,
        batch_size=batch_size,
        opt=collections.namedtuple("Opt", "single_cls")(False), 
        augment=True, hyp=hyp, rect=False, cache=False, stride=32, pad=0, image_weights=False
        # 增广改为True ， hyp设置上， pad设置为0， rect是我私自改成False的？？？
        )
    return dataloader

# 3.0 定义函数
import os
from pathlib import Path
import yolov7.test as test
def evaluate_coco(model, loader, save_dir=".", conf_thres=0.001, iou_thres=0.65):
    '''
    Super-mkdir（超级创建目录）：创建一个叶子目录和所有中间目录。
    其工作方式类似于 mkdir，不同之处在于它会创建所有不存在的中间路径段（
    不仅仅是最右边的那一个）。如果目标目录已经存在，
    当 `exist_ok` 参数为 `False` 时，
    将会抛出一个 `OSError` 异常。否则，不会抛出任何异常。这个操作是递归的。
    '''
    if save_dir and os.path.dirname(save_dir) != "": # 如果路径不存在或父目录为空
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    
    return test.test( # 返回(mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
        data="data/coco.yaml",
        save_dir=Path(save_dir), # 保存结果的路径
        dataloader=loader,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        model=model,
        is_coco=True,
        plots=False, # 是否生成结果图像
        half_precision=True, #是否使用半精度浮点数进行评估
        save_json=False # 是否用json保存结果 
    )[0][3] # 只需要map
    """
    def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False):
    """

# 8.1.1.1定义函数
def collect_stats(model, data_loader, device, num_batch=200):
    """
    num_batch:对多少个batch进行标定
    """
    model.eval()
    
    # 遍历模型，启动标定，禁用量化
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer): #遍历所有信息。直到层里有quant_nn.TensorQuantizer
            if module._calibrator is not None: #如果有校准器
                module.disable_quant() # 关闭量化
                module.enable_calib() # 开启标定
            else: # 如果没有校准器，禁用该模块
                module.disable() #直接禁用该模块
    
    # 模型前向收集信息 test
    with torch.no_grad():
        for i, datas in enumerate(data_loader):
            imgs = datas[0].to(device, non_blocking=True).float()/255.0 
            # 此处应该是yolov7自身原因，当初一个批次4张图片，datas是列表里面有4个。这里似乎只用了列表中第一条数据
            # 具体可以参考evaluate_coco中官方test()函数是如何写的
            """
            非阻塞操作
            non_blocking=True: 
            这个参数表示数据迁移操作是非阻塞的。在默认情况下（non_blocking=False），.to() 方法会等待数据完全复制到目标设备后才会继续执行后面的代码。
            但如果设置了 non_blocking=True，.to() 方法会立即返回，而数据迁移会在后台进行，这样可以提高代码的执行效率。
            """
            model(imgs)
            print(i)
            if i > num_batch:
                break
            
    # 标定完成后。开启量化，关闭标定
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

        
    
# 8.1.2.1定义函数
def compute_amax(model, device, **kwargs):
    # for循环遍历所有模块。如果是量化模块，并且具有校准器
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                # print(f"{module}\n,{module._calibrator}")
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax() # 加载，计算amax。结果保存到每个module的_amax变量中
                else:
                    module.load_calib_amax(**kwargs)                   
                module._amax = module._amax.to(device) # 计算结束后，把值放到device上
                print(F"{name:40}: {module}")
    # model.cuda() # ????

# 8.1 定义模型
def calibrate_model(model, dataloader, device):
    ## 8.1.1 收集信息
    collect_stats(model, dataloader, device, 10) # 调用8.1.1.1中定义的函数
    
    ## 8.1.2 计算动态范围  amax scale
    compute_amax(model, device, method="mse") # 调用8.1.2.1中定义的函数,传值mse，给compute_amax用。

# 9.0 定义函数
def export_ptd(model, save_file, device, dynamic_batch=False):
    input_dummy = torch.randn(1, 3, 640, 640, device=device).to(torch.float32)
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    with torch.no_grad():
        torch.onnx.export(model, input_dummy, save_file, 
                        opset_version=13, 
                        verbose=True, 
                        input_names=["input"],
                        output_names=["output"],
                        dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}} if dynamic_batch else None,
                        do_constant_folding=False)

        
    quant_nn.TensorQuantizer.use_fb_fake_quant = False       
 
# 10.1 定义检测层(层的名字)是否是量化层
def have_quantizer(layer):
    for name, module in layer.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True
# 10.2 
class disable_quantization: # 类似于上下文管理器的工作。作用：临时禁制模型该层的量化 ??为啥带()
    # 初始化
    def __init__(self, model):
        self.model = model
        
    # 具体应用的函数,关闭量化
    def apply(self, disabled=True):
        # 遍历每个模块。如果是量化器，会将disable的属性值设置为True
        for name, module in self.model.named_modules(): # 一定要写self
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self): # 进入上下文管理器的操作
        print("enter")
        self.apply(disabled=True)
        
    def __exit__(self, *args, **kwargs): # 退出上下文管理器的时候的操作，启用量化层
        print("exit")
        self.apply(disabled=False)
# 10.3       
# 重启量化     
class enable_quantization: 
    def __init__(self, model):
        self.model = model

    def apply(self, enable=True):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enable
                
    def __enter__(self): 
        print("enter")
        self.apply(disabled=True)
        
    def __exit__(self, *args, **kwargs): 
        print("exit")
        self.apply(disabled=False)       

# 10.4
# 负责保存敏感曾分析的类
import json
class SummaryTool:
    def __init__(self, file):
        self.file = file
        self.data = []
        
    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent = 4) # indent=4更容易阅读
                   
# 10.0 定义敏感曾函数
def sensitive_analysis(model:torch.nn.Module, val_dataloader, save_file):
    # 保存的文件名
    # save_file = "sensitive_analysis.json"
    summary = SummaryTool(save_file)
    
    # 10.1 for循环model的每一个quantizer层
    print(f"Sensitive analysise by each layer...")
    for i in range(0, len(model.model)):
        layer = model.model[i]
        # 判断是否是量化层
        if have_quantizer(layer): # 如果是量化层
            # 使该层的量化失效，不进行int8的量化，使用fp16进行运算。退出with时使用__exit__启用
            # disable_quantization(layer).apply()# 10.1.1定义类，这个类有类似上下文管理器的功能。进入时调用__enter__方法。退出时调用__exit__方法
            with disable_quantization(layer) as f:
                print("disable_quantization")
            
                # 计算mAP值
                ap = evaluate_coco(model, val_dataloader)
            
                # 保存这一层的精度值
                summary.append([ap, f"model.{i}"])

            # enable_quantization(layer).apply()
                
            print(f"layer {i} ap: {ap}")
        else:
            print(f"ignore model.{i} because it is {type(layer)}")
            
    # 循环计数，打印前10个影响比较大的层
    # 逻辑是我关闭了你这个层。结果ap值比较大。说明你这个层开启会掉精度。
    summary = sorted(summary.data, key=lambda x: x[0], reverse=True)
    print("Sensitive summary: ")
    for n, (ap, name) in enumerate(summary[:10]):
        print(f"Top{n}: Using fp16 {name}, ap - {ap:.5f}")
            

if __name__ == "__main__":
    # 0. 定义权重变量1
    weight = "yolov7.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    # pth_model = load_yolov7_model(weight, device)
    
    # 2. 加载验证数据
    cocodir = "coco"
    val_dataloader = perpare_val_dataset(cocodir)
    train_dataloader = perpare_train_dataset(cocodir)
    
    # 4. 自动插入QDQ节点
    model = prepare_model(weight, device=device)
    # print(model_auto_qdq)
    # print(f"====Evaluate Auto add QDQ Model....=====")
    # ap = evaluate_coco(model, dataloader)
    
    # 6. 手动插入QDQ节点
    # replace_to_quantization_model(model) # 6.2 中函数调用
    # print(pth_model)
    # print(type(pth_model))
    
    # 8. 模型标定，此时mAp会贼略微的提升。结束后模型就是ptq模型
    # 模型标定需要模型以及数据。所以需要传参
    # 重要：需要使用训练集的dataloader
    # calibrate_model(model, train_dataloader, device)
    # print(f"====Evaluate Max PTQ....=====")
    # ap = evaluate_coco(pth_model, dataloader)
    
    # 10. 敏感层分析
    # sensitive_analysis(model, val_dataloader) #使用val的dataloader
    """
    1. for循环model的每一个quantizer层
    2. 只关闭该层的量化，其余的层的量化保留
    3. 验证模型，计算ap和mAP。使用evaluate_coco()并保存精度值
    4. 验证结束。重新该层的量化。
    5. for玄幻结束，得到所有层的精度值
    6. 排序，得到前10个对精度影响比较大的层。将这些曾进行打印
    """
    # 11 
    # 如何处理敏感层分析出的结果：将影响较大的层关闭量化，使用fp16进行计算
    # 所以在进行PTQ量化之前就要进行敏感层的分析。得到影响较大的层，然后再手动插入量化节点
    # 的时候将这些影响层进行量化的关闭
    
    # 11.1 把敏感层分析出来的前10个结果拿出来。存ignore_layer中
    # ignore_layer = ["model\.105\.(.*)", "model\.22\.(.*)", "model\.26\.(.*)", "model\.56\.(.*)", "model\.0\.(.*)", 
    #                 "model\.28\.(.*)", "model\.82\.(.*)", "model\.75\.(.*)", "model\.57\.(.*)", "model\.92\.(.*)"] # 存的是正则表达式能匹配的
    # 11.2 改造replace_to_quantization_model函数的定义
    # replace_to_quantization_model(model, ignore_layer)
    # print(model)# 打印出来后，就可以发现。ignore_layer列表中的层。没有插入量化节点。
    
    # for name, module in model.named_modules():
    #     if isinstance(module, quant_nn.TensorQuantizer):
    #         print(f"{module}\n,{module._calibrator}")
    

    # 5. 查看插入QDQ与没有插入的区别
    # print(model) # 插入的
    # print(pth_model) # 原始的
    
    # 9模型导出
    # pth_model.to(device)
    # save_file = "ptq_yolov72.onnx"
    export_ptd(model, save_file, device)
    
    # 3. 验证评估
    for i in range(10):
        print(f"{i}")
        ap = evaluate_coco(model, val_dataloader)
    
    