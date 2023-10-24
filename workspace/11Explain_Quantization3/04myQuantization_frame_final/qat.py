from argparse import ArgumentParser

import torch
import torch.optim as optim
from torch.cuda import amp
import shutil
import os
import torchvision.transforms as T
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.imagenet import ImageFolder
from argparse import ArgumentParser
from tqdm import tqdm
from copy import deepcopy
import datetime
import tinyq
import torchvision.models as models

def evaluate(model, batch, data):
    model.eval()
    transform = T.Compose([
        T.Resize(224 + 32, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    valset = ImageFolder(os.path.join(data, "val"), transform)
    valloader = DataLoader(valset, batch, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    correct = 0
    total   = 0
    dtype = next(model.parameters()).dtype
    bar = tqdm(enumerate(valloader), total=len(valloader), desc="Evaluate")
    for ibatch, (image, target) in bar:
        B = image.size(0)

        with torch.no_grad():
            predict = model(image.cuda().to(dtype)).view(B, -1).argmax(1)
        
        correct += (predict == target.cuda()).sum()
        total += B
        accuracy = correct / total
        bar.set_description(f"Evaluate accuracy is: {accuracy:.6f}")
    
    accuracy = correct / total
    print(f"Top1 accuracy is: {accuracy:.6f}")
    return accuracy

class Logger:
    def __init__(self, file):
        self.file = file
        os.makedirs(os.path.dirname(file), exist_ok=True)

    def info(self, *args, console=True):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = " ".join([f"[Info] {now}"] + list(args))

        if console:
            print(msg)

        with open(self.file, "a+") as f:
            f.write(msg + "\n")

def doqat(ckpt, epochs, num_batch_epoch, batch, data, save, sparsity, device):

    torch.manual_seed(42)
    torch.cuda.set_device(device)

    model = models.resnet50(pretrained=True).eval().cuda()
    tinyq.replace_modules(model)
    print(model)
    model.load_state_dict(torch.load(ckpt, map_location="cpu").state_dict())

    model.float()

    if os.path.isdir(save) and save != "." and save != "..":
        print(f"Remove old folder: {save}")
        shutil.rmtree(save)

    logger = Logger(f"{save}/run.log")
    logger.info(
        "\n=======================================================\n" +
        f"Run at: \n" + 
        f" epochs = {epochs}\n" + 
        f" num_batch_epoch = {num_batch_epoch}\n" + 
        f" batch = {batch}\n" + 
        f" save = {save}\n" + 
        f" ckpt = {ckpt}\n" +
        f" sparsity = {sparsity}\n" + 
        "======================================================="
    )
    
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BILINEAR),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    trainset = ImageFolder(os.path.join(data, "train"), transform)
    trainloader = DataLoader(trainset, batch, shuffle=True, num_workers=8, pin_memory=True, drop_last=False)

    origin_model = deepcopy(model).eval()
    tinyq.linker(origin_model).disable

    model.train()
    model.requires_grad_(True)
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.requires_grad_(False)
            module.eval()

    fp16         = True
    scaler       = amp.GradScaler(enabled=fp16)
    optimizer    = optim.Adam(model.parameters(), 1e-5)
    quant_lossfn = torch.nn.MSELoss()
    device       = next(model.parameters()).device
    
    lrschedule = {
        0: 1e-5,
        3: 1e-5,
        5: 1e-6,
        13: 1e-6,
        14: 1e-7
    }

    def make_layer_forward_hook(l):
        def forward_hook(m, input, output):
            l.append(output)
        return forward_hook

    dtype = next(model.parameters()).dtype
    supervision_module_pairs = []
    for ((mname, ml), (oriname, ori)) in zip(model.named_modules(), origin_model.named_modules()):
        if isinstance(ml, tinyq.Quantizer): continue
        if not isinstance(ml, torch.nn.ReLU) and mname != "fc": continue

        logger.info(f"Add supervision: {mname}", console=False)
        supervision_module_pairs.append([ml, ori])

    for iepoch in range(epochs):

        if iepoch in lrschedule:
            learningrate = lrschedule[iepoch]
            for g in optimizer.param_groups:
                g["lr"] = learningrate

        model_outputs  = []
        origin_outputs = []
        remove_handle  = []

        for ml, ori in supervision_module_pairs:
            remove_handle.append(ml.register_forward_hook(make_layer_forward_hook(model_outputs))) 
            remove_handle.append(ori.register_forward_hook(make_layer_forward_hook(origin_outputs)))

        model.train()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.requires_grad_(False)
                module.eval()
        
        pbar = tqdm(trainloader, desc="QAT", total=num_batch_epoch)
        for ibatch, (imgs, target) in enumerate(pbar):

            if ibatch >= num_batch_epoch:
                break
            
            imgs = imgs.to(device).to(dtype)
            with amp.autocast(enabled=fp16):
                B = imgs.size(0)
                pred = model(imgs).view(B, -1)

                with torch.no_grad():
                    target = origin_model(imgs).view(B, -1).softmax(-1)

                quant_loss = 0
                for index, (mo, fo) in enumerate(zip(model_outputs, origin_outputs)):
                    quant_loss += quant_lossfn(mo, fo)

                model_outputs.clear()
                origin_outputs.clear()

                if fp16:
                    scaler.scale(quant_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    quant_loss.backward()
                    optimizer.step()
                optimizer.zero_grad()
            
            if ibatch % 100 == 0:
                logger.info(f"QAT Finetuning {iepoch + 1} / {epochs}, Loss: {quant_loss.detach().item():.5f}, LR: {learningrate:g}", console=False)
            
            pbar.set_description(f"QAT Finetuning {iepoch + 1} / {epochs}, Loss: {quant_loss.detach().item():.5f}, LR: {learningrate:g}")

        # You must remove hooks during onnx export or torch.save
        for rm in remove_handle:
            rm.remove()

        top1_accuracy = evaluate(deepcopy(model).half(), 512, data)
        save_path = f"{save}/{iepoch:03d}_{top1_accuracy:.6f}.pth"

        logger.info(f"{iepoch}. Lr is: {learningrate}, Save to: {save_path}")
        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":

    parser = ArgumentParser('evaluate imagenet accuracy')
    parser.add_argument('ckpt', type=str, help="ckpt file")
    parser.add_argument('--epochs', type=int, default=15, help="batch size")
    parser.add_argument('--num_batch_epoch', type=int, default=1000, help="batch size")
    parser.add_argument('--batch', type=int, default=64, help="finetune batch size")
    parser.add_argument('--data', type=str, default="/dset/imagenet", help="imagenet data path")
    parser.add_argument('--save', type=str, default="qat_finetune", help="save to folder")
    parser.add_argument('--sparsity', action="store_true", help="Enable sparsity")
    parser.add_argument('--device', type=str, default="cuda:0", help="device name")

    # 0.761080
    # ckpt, nofuseadd, epochs, num_batch_epoch, batch, data, save
    doqat(**parser.parse_args().__dict__)