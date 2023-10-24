from argparse import ArgumentParser

import os
import torchvision.transforms as T
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.imagenet import ImageFolder
from argparse import ArgumentParser
from tqdm import tqdm
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

def docalibration(num, batch, data, save):

    model = models.resnet50(pretrained=True).eval().cuda()
    tinyq.replace_modules(model)

    print(model)
    print(
        "=======================================================\n" +
        f"Run at: \n" + 
        f" num = {num}\n" + 
        f" batch = {batch}\n" + 
        "======================================================="
    )
    
    transform = T.Compose([
        T.Resize(224 + 32, interpolation=T.InterpolationMode.BILINEAR),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    trainset = ImageFolder(os.path.join(data, "train"), transform)
    trainloader = DataLoader(trainset, batch, shuffle=True, num_workers=8, pin_memory=True, drop_last=False)
    dtype = next(model.parameters()).dtype
    with tinyq.collect(model):
        for image, target in tqdm(tinyq.early(trainloader, num), desc="Calibing"):
            model(image.cuda().to(dtype))

    print(model)
    top1_accuracy = evaluate(model, 512, data)

    if save is None:
        save = f".ptq_{top1_accuracy:.6f}.pth"
    else:
        save = save.replace("{acc}", f"{top1_accuracy:.6f}")

    torch.save(model.state_dict(), save)
    print(f"Done, Save to: {save}")


if __name__ == "__main__":

    parser = ArgumentParser('evaluate imagenet accuracy')
    parser.add_argument('--num', type=int, default=10, help="batch size")
    parser.add_argument('--batch', type=int, default=256, help="batch size")
    parser.add_argument('--data', type=str, default="/dset/imagenet", help="imagenet data path")
    parser.add_argument('--save', type=str, help="save to")
    
    docalibration(**parser.parse_args().__dict__)