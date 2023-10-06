"""
注意：将1quantize.py的文件名字替换成
"""
import torch
import quantize # 自己的函数
import argparse # 准备使用命令行传参的方式

import debugpy
# 保证host和端口一致，listen可以只设置端口，则为localhost,否则设置成(host,port)
debugpy.listen(12346)
print('wait debugger')
debugpy.wait_for_client()
print('Debugger Attached')

def run_SensitiveAnalysis(weight, cocodir, device="cpu"):
    # perpare model
    print("Prepare Model ....")
    model = quantize.prepare_model(weight, device)
    quantize.replace_to_quantization_model(model)
    
    # perpare dataset
    print("Prepare Dataset ....")
    val_dataloader = quantize.perpare_val_dataset(cocodir)
    train_dataloader = quantize.perpare_train_dataset(cocodir)
    
    # calibration model
    print("Begining Calibration ....")
    quantize.calibrate_model(model, train_dataloader, device)
    
    # sensitive analysis and print
    print("Begining Sensitive Analysis ....")
    quantize.sensitive_analysis(model, val_dataloader, args.sensitive_summary)


def run_PTQ(args, device="cpu"):
    # perpare model
    print("Prepare Model ....")
    model = quantize.prepare_model(args.weight, device)
    quantize.replace_to_quantization_model(model, args.ignore_layers) 
    # 与敏感层分析的最大的区别。就是这里传了，敏感层分析后的args.ignore_layers
    # 这些层在量化时会被忽略，直接用fp16计算。
    
    # perpare dataset
    print("Prepare Dataset ....")
    val_dataloader = quantize.perpare_val_dataset(args.cocodir, batch_size=args.batch_size)
    train_dataloader = quantize.perpare_train_dataset(args.cocodir, batch_size=args.batch_size)
    
    # calibration model
    print("Begining Calibration ....")
    quantize.calibrate_model(model, train_dataloader, device)
    
    summary = quantize.SummaryTool(args.ptq_summary)
    
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

def get_parser():
    """
    Creates an argument parser.
    """
    parser = argparse.ArgumentParser(description='YOLOV7 quantization flow script')
    parser.add_argument('--weight', type=str, default='yolov7.pt', help='initial weights path')
    parser.add_argument('--cocodir', type=str, default='coco', help='coco dataset directory') # 数据集路径需修改。我放在yolov7项目里了
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for dataloader,default=8')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    parser.add_argument('--close_sensitive', action="store_true", help="close use sensitive analysis or not befor ptq")
    parser.add_argument('--sensitive_summary', type=str, default="sensitive-summary.json", help="summary save file")
    parser.add_argument('--ignore_layers', type=str, default="model\.105\.m\.(.*)", help="regx") # 正则匹配，经敏感层分析后哪层不量化
    
    parser.add_argument('--save_ptq', type=bool, default=False, help="file")
    parser.add_argument('--ptq', type=str, default="ptq_yolov7.onnx", help="file")

    parser.add_argument('--conf_thres', type=float, default=0.001, help='confidence threshold default=0.001')
    parser.add_argument('--nmsiou_thres', type=float, default=0.65, help='nms threshold')
    
    parser.add_argument('--eval_origin', action="store_true", help='do eval for origin model')
    parser.add_argument('--eval_ptq', action="store_true", help='do eval for ptq model')
    
    parser.add_argument('--ptq_summary', type=str, default="ptq_summary.json", help="summary save file")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    
    is_cuda = (args.device != "cpu") and torch.cuda.is_available()
    device = torch.device("cuda:0" if is_cuda else "cpu")
    
    # 敏感层分析
    if not args.close_sensitive:
        print("Sensitive Analysis...")
        run_SensitiveAnalysis(args.weight, args.cocodir, device)
    
    
    # 然后模型标定，生成PTQ
    ignore_layers = ["model.105.m.(.*)", "model.99.m.(.*)"] # 注意敏感层分析后，类似这样传值。
    args.ignore_layers = ignore_layers
    
    print("Begin PTQ...")
    run_PTQ(args, device)
    
    print("PTQ Quantization Has Finished ....")
    
    """
    敏感层分析：
    python /yolov7/ptq.py 
    
    标定好后，修改ignore_layers，使用命令
    python /yolov7/ptq.py --close_sensitive --save_ptq=True --eval_origin --eval_ptq
    """
    