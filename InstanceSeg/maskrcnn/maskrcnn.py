import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import argparse
import time

# 定义数据集类
class CocoDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super(CocoDataset, self).__init__(root, annFile)
        self.transform = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDataset, self).__getitem__(idx)
        if self.transform is not None:
            img = self.transform(img)
        target = [{k: torch.tensor(v) for k, v in t.items()} for t in target]
        return img, target

  
# 训练函数
def train(model, 
          dataloader, 
          optimizer,
          device,
          epoch):
    model.train()
    for images, targets in dataloader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {losses.item()}")

# 推理函数
def eval(model, 
         device, 
         dataloader):
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            return outputs
        
# 可视化结果
def visualize_result(image, outputs):
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    draw = ImageDraw.Draw(Image.fromarray(image))
    for box in outputs[0]['boxes']:
        draw.rectangle(box.cpu().numpy(), outline="red", width=3)
    plt.imshow(image)
    plt.show()

   
def main():
    # Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-batch-size', default=4, type=int,
                        help='Default value is 4.')
    parser.add_argument('--train-epochs', default=2, type=int,
                        help='Default value is 2. Set 0 to this argument if you want an untrained network.')
    parser.add_argument('--infer-batch-size', default=1, type=int,
                        help='Default value is 1.')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), type=str,
                        choices=['cuda', 'cpu'], help='Device to use for training and inference.')
    parser.add_argument('--image-size', default=256, type=int,
                        help='The width and height of an image.')
    parser.add_argument('--dataset-root', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'), type=str,
                        help='Location of the dataset root directory.')
    parser.add_argument('--mode', default='both', type=str,
                        choices=['train', 'infer', 'both'], help='Mode to run: train, infer, or both.')
    parser.add_argument('--num_classes', default=91, type=int,
                            help='Class num')

    args = parser.parse_args()
    
    print(vars(args))
    
    input_size = (args.image_size, args.image_size)
    
    train_dir = os.path.join(args.dataset_root, "images/train2017")
    train_ann_file = os.path.join(args.dataset_root, "images/annotations/instances_train2017.json")
    val_dir = os.path.join(args.dataset_root, 'images/val2017')
    val_ann_file = os.path.join(args.dataset_root, 'images/annotations/instances_val2017.json')
    
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = CocoDetection(train_dir, train_ann_file, transforms=transform)
    val_dataset = CocoDetection(val_dir, val_ann_file, transforms=transform)
    
    # 数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, 
                                               shuffle=True, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.infer_batch_size, 
                                             shuffle=False, num_workers=2,collate_fn=lambda x: tuple(zip(*x)))
    
    if args.mode == "infer":
        model = maskrcnn_resnet50_fpn(pretrained=True, num_classes=args.num_classes)
    else:
        model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=args.num_classes)
    
    device = torch.device(args.device)
    model.to(device)    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    if args.mode in ['train', 'both']:
        print(f"[INFO] Start training on {args.device}.")
        start_time = time.time()
        for epoch in range(args.train_epochs):
            print(f"[INFO] Training epoch {epoch + 1}/{args.train_epochs}.")
            train(model, train_loader, optimizer, device, epoch)
        end_time = time.time()
        print(f"Training complete in {end_time - start_time} seconds.")
        
    if args.mode in ['infer', 'both']:
        print(f"[INFO] Start inference on {args.device}")
        eval(model, device, val_loader)


if __name__ == "__main__":
    main()
