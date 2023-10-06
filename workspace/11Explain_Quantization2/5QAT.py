"""
以下为yolov7 QAT量化代码
    # 启动命令：python ../qat.py --fp16
    # 执行过程：见2.png
        - 1. 标定的batch数量200 固定死的数值。修改需要在quantize.py中calibrate_model修改
        - 2. 关于敏感层分析。这里没有。在ptq.py有代码。这里直接使用敏感层分析的结果。即default="model\.105\.m\.(.*)"这些层不量化
        - 3. QAT微调训练在tain_dataloader中实现。跑10个epoch
        
    # 最后结果。产生 qat_yolov7.onnx 
        # 产生 Finetune.json中储存微调结果日志文件。
            #大的210行左右的微调需要修改。  ！！！！！！！  也许不重要  

"""
import torch
import os, argparse
import quantize
from torch.cuda import amp # 新增用于混合精度
import torch.optim as optim # 新增
from pytorch_quantization import nn as quant_nn
from copy import deepcopy

# import debugpy
# # 保证host和端口一致，listen可以只设置端口，则为localhost,否则设置成(host,port)
# debugpy.listen(12346)
# print('wait debugger')
# debugpy.wait_for_client()
# print('Debugger Attached')

from typing import Callable # 8.3 导入包
def run_finetune(args, model:torch.nn.Module, train_loader, val_loader,supervision_policy:Callable=None, fp16=True): # 8.2 修改入参。增加supervision_policy
    # 0.0 QAT部分前面的代码和我们模型训练的工作相似
    summary = quantize.SummaryTool("Finetune.json")
    
    # 8.4 传入模型进行深拷贝
    origin_model = deepcopy(model).eval()
    quantize.disable_quantization(origin_model).apply()
    
    # 0.1 模型、启动梯度、启动混合精度寻来你、初始化优化器、损失函数、设备、学习率策略
    model.train()
    model.requires_grad_(True) # 启动梯度计算
    
    scaler = amp.GradScaler(enabled=True) #fp16
    # 开启混合精度的训练。用于自动调整梯度的缩放比例。防止使用FP16推理时。
        # 出现梯度下溢或者上溢的问题。enabled=True表示启用混合精度
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr) # 优化器
    
    quant_lossfn = torch.nn.MSELoss() # 损失函数 主要是获得量化后模型 与 原始模型输出的 差异
    
    device = next(model.parameters()).device # model.parameters() 返回一个生成器，该生成器包含模型的所有参数（通常是张量）。next(model.parameters()) 则会返回生成器的第一个元素，即模型的第一个参数。
    
    lrschedule = { # 学习策略
        0: 1e-6,
        3: 1e-5,
        8: 1e-6
    }
    
    # 8.5 创建hook函数。
    def make_layer_forward_hook(l):
        def forward_hook(module, input_tuple, outputs):
            l.append(outputs)
        return forward_hook
    
    # 8.6 获取ptq 与 原始模型之间的匹配对
    supervision_module_pairs = []
    for (ptq_n, ptq_m), (origin_n, origin_m) in zip(model.named_modules(), origin_model.named_modules()):
        if isinstance(ptq_m, quant_nn.TensorQuantizer):
            continue
        
        if supervision_policy:
            if not supervision_policy(ptq_n, ptq_m):
                continue
            
        supervision_module_pairs.append([ptq_m, origin_m])
          
    
    # 0.2 循环epoch
    best_ap = 0
    for epoch in range(args.num_epoch):
        # 1.2.1 动态学习率
        if epoch in lrschedule: # 如果epoch符合我们lrschedule字典的键，就把优化其中的学习率改为对应的值
            learning_rate = lrschedule[epoch]
            for g in optimizer.param_groups:
                g['lr'] = learning_rate
                
        # 8.7 注册hook，将输出放入列表中
        ptq_outputs = []
        origin_outputs = []
        remove_handle = [] # 用于删除handle
        for ptq_module, origin_module in supervision_module_pairs:
            remove_handle.append(ptq_module.register_forward_hook(make_layer_forward_hook(ptq_outputs)))
            remove_handle.append(origin_module.register_forward_hook(make_layer_forward_hook(origin_outputs)))
                   
        # 0.2.2 训练
        # 训练并不需要训练每个epoch中所有数据。只需要epoch中 1/10数据 即可。
        model.train()
        
        for batch_idx, datas in enumerate(train_loader):
            if batch_idx >= args.iters:
                break
            
            images = datas[0].to(device).float() / 255.0 # 图像预处理。
            
            # 根据run_finetune函数传进来的fp16参数的值，决定是否移动fp16混合精度训练。 
            with amp.autocast(enabled=args.fp16):
                model(images) #ptq模型前向  # 为啥不接收结果？不需要最终结果。前向过程中每层的输出都通过钩子
                
                #9.0 完善模型前向
                with torch.no_grad():
                    origin_model(images) #原始模型前向
                
                # 计算量化损失
                quant_loss = 0
                
                # 10.0 完善loss计算
                for index, (ptq_out, origin_out) in enumerate(zip(ptq_outputs, origin_outputs)):
                    quant_loss += quant_lossfn(ptq_out, origin_out)
                    
                ptq_outputs.clear() # 清除方便下一次使用
                origin_outputs.clear()
                
                
            if fp16: # 如果fp16为true，则半精度训练
                scaler.scale(quant_loss).backward() # 缩放损失值，并进行反向传播
                scaler.step(optimizer)              # 更新模型参数
                scaler.update()                     # 更新缩放器状态
            else: # 否则正常梯度更新
                quant_loss.backward()
                optimizer.step()
                
            optimizer.zero_grad()
            
            print(f"QAT Finetuning {epoch + 1} / {args.num_epoch}, Loss:{quant_loss.detach().item():.5f}, LR:{learning_rate}:g")# :g科学计数法表示
        
        # 11.0 一次遍历后，移除handle
        for handle in remove_handle:
            handle.remove()
        
        # 0.2.3 模型评估。
        ap = quantize.evaluate_coco(model, val_dataloader)
        summary.append([f"QAT{epoch}", ap])
        
        if ap > best_ap:
            print(f"Save qat model to {args.qat} @ {ap:.5f}") # args.save_qat 修改为args.qat
            best_ap = ap
            # torch.save({"model":model})
            quantize.export_ptd(model, "qat_yolov7.onnx", device)
         
def get_parser():
    """
    Creates an argument parser.
    """
    parser = argparse.ArgumentParser(description='YOLOV7 quantization flow script')
    parser.add_argument('--weight', type=str, default='yolov7.pt', help='initial weights path')
    parser.add_argument('--cocodir', type=str, default='coco', help='coco dataset directory') # 数据集路径需修改。我放在yolov7项目里了
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for dataloader,default=8')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    #======================================以下为新增的训练参数=========================================
    parser.add_argument('--num_epoch', type=int, default=10, help='max epoch for finetune')
    parser.add_argument('--iters', type=int, default=200, help='iters per epoch')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for QAT Finetune')
    
    #======================================以上为新增的训练参数=========================================
    
    # parser.add_argument('--close_sensitive', action="store_true", help="close use sensitive analysis or not befor ptq") # 删除
    # parser.add_argument('--sensitive_summary', type=str, default="sensitive-summary.json", help="summary save file") # 删除
    parser.add_argument('--ignore_layers', type=str, default="model\.105\.m\.(.*)", help="regx") # 正则匹配，经敏感层分析后哪层不量化
    
    parser.add_argument('--save_ptq', type=bool, default=False, help="file")
    parser.add_argument('--ptq', type=str, default="ptq_yolov7.onnx", help="file")
    
    parser.add_argument('--save_qat', type=bool, default=False, help="file") # 新增
    parser.add_argument('--qat', type=str, default="qat_yolov7.onnx", help="file") # 新增

    parser.add_argument('--conf_thres', type=float, default=0.001, help='confidence threshold default=0.001')
    parser.add_argument('--nmsiou_thres', type=float, default=0.65, help='nms threshold')
    
    parser.add_argument('--eval_origin', action="store_true", help='do eval for origin model')
    parser.add_argument('--eval_ptq', action="store_true", help='do eval for ptq model')
    parser.add_argument('--eval_qat', action="store_true", help='do eval for qat model') # 新增
    parser.add_argument('--fp16', action="store_true", help='do eval for qat model') # 新增
    
    parser.add_argument('--eval_summary', type=str, default="ptq_summary.json", help="summary save file")# 名字从qat_summary 修改为eval_summary
    return parser

if __name__ == "__main__":
    parser = get_parser() 
    args = parser.parse_args()
    print(args)
    
    is_cuda = (args.device != "cpu") and torch.cuda.is_available()
    device = torch.device("cuda:1" if is_cuda else "cpu")
    
    # perpare model
    print("Prepare Model ....")
    model = quantize.prepare_model(args.weight, device)
    quantize.replace_to_quantization_model(model, args.ignore_layers) 
    
    # perpare dataset
    print("Prepare Dataset ....")
    val_dataloader = quantize.perpare_val_dataset(args.cocodir, batch_size=args.batch_size)
    train_dataloader = quantize.perpare_train_dataset(args.cocodir, batch_size=args.batch_size)
    
    # calibration model
    print("Begining Calibration ....")
    quantize.calibrate_model(model, train_dataloader, device)    
    
    # 日志文件
    summary = quantize.SummaryTool(args.eval_summary)
    
    if args.eval_origin:
        print("Evaluate Origin...")
        with quantize.disable_quantization(model):
            ap = quantize.evaluate_coco(model, val_dataloader, conf_thres=args.conf_thres, iou_thres=args.nmsiou_thres)
            summary.append(["Origin", ap])
    if args.eval_ptq:
        print("Evaluate PTQ...")
        ap = quantize.evaluate_coco(model, val_dataloader, conf_thres=args.conf_thres, iou_thres=args.nmsiou_thres)
        summary.append(["PTQ", ap])
        
    if args.save_ptq:
        print("Export PTQ...")
        quantize.export_ptd(model, args.ptq, device)
        
    # 8.0 判断传入的模块受否需要在QAT训练期间计算损失
    def supervision_policy():
        supervision_list = [] # 空列表，存储模型模块的id
        for item in model.model:
            supervision_list.append(id(item))
        
        supervision_stride = 1 # 间隔数
        keep_idx = list(range(0, len(model.model) - 1, supervision_stride))
        keep_idx.append(len(model.model) - 2) #传入倒数第三个模块的数值
        
        def impl(name, module):
            if id(module) not in supervision_list: # 曾经写错了
                return False
            
            idx = supervision_list.index(id(module))
            if idx in keep_idx:
                print(f"Supervision: {name} will compute loss origin model during QAT training...")
            else:
                print(f"Supervision: {name} not compute loss origin model during QAT training...")
                
            return idx in keep_idx # True/False
        
        return impl
            
        
    #===================以下正式为qat部分代码====================
    # 0.3
    print("begining Finetune ...")
    run_finetune(args, model, train_dataloader, val_dataloader, supervision_policy=supervision_policy()) # 8.1 将编写好的supervision_policy，当作入参传入
    print("QAT Finished ...")
    
