import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import sys
from pathlib import Path

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models.segmentation import fcn_resnet50
from torchvision.datasets import VOCSegmentation
from tqdm import tqdm

sys.path.append("../")
from bench.evaluator import Evaluator


# Define the color map for VOC dataset
VOC_COLORMAP = [
    (0, 0, 0),        # Background
    (128, 0, 0),      # Aeroplane
    (0, 128, 0),      # Bicycle
    (128, 128, 0),    # Bird
    (0, 0, 128),      # Boat
    (128, 0, 128),    # Bottle
    (0, 128, 128),    # Bus
    (128, 128, 128),  # Car
    (64, 0, 0),       # Cat
    (192, 0, 0),      # Chair
    (64, 128, 0),     # Cow
    (192, 128, 0),    # Dining table
    (64, 0, 128),     # Dog
    (192, 0, 128),    # Horse
    (64, 128, 128),   # Motorbike
    (192, 128, 128),  # Person
    (0, 64, 0),       # Potted plant
    (128, 64, 0),     # Sheep
    (0, 192, 0),      # Sofa
    (128, 192, 0),    # Train
    (0, 64, 128)      # TV/Monitor
]

# Create a dictionary mapping from RGB color to class index
VOC_COLOR_DICT = {color: idx for idx, color in enumerate(VOC_COLORMAP)}

def voc_colormap_to_label(mask):
    mask = mask.convert("RGB")
    mask = np.array(mask)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
    for color, idx in VOC_COLOR_DICT.items():
        label_mask[np.all(mask == color, axis=-1)] = idx
    return torch.tensor(label_mask, dtype=torch.long)


# Function to train the model.
def train(model,
          train_loader,
          criterion,
          optimizer,
          device,
          epoch,
          args):
    model.train()

    # 初始化记录变量
    batch_losses = []
    batch_times = []
    epoch_loss_sum = 0.0
    epoch_start_time = time.time()

    # 使用 tqdm 包装 DataLoader
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.train_epochs}")

    for batch_idx, (images, targets) in enumerate(pbar):
        batch_start_time = time.time()

        # 数据移至设备
        images = images.to(device)
        targets = targets.squeeze(1).long().to(device)

        # 前向传播 + 计算损失
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, targets)

        # 反向传播 + 优化
        loss.backward()
        optimizer.step()

        # 记录 batch 数据
        batch_time = time.time() - batch_start_time
        batch_loss = loss.item()

        batch_losses.append(batch_loss)
        batch_times.append(batch_time)
        epoch_loss_sum += batch_loss

        # 更新 tqdm 信息（显示当前 batch 的 loss 和 it/s）
        pbar.set_postfix({
            "batch_loss": f"{batch_loss:.4f}"
        })

    # 计算 epoch 级数据
    epoch_time = time.time() - epoch_start_time
    epoch_avg_loss = epoch_loss_sum / len(train_loader)
    epoch_avg_it_s = len(train_loader) / epoch_time  # epoch 平均 it/s

    print(f"\nEpoch {epoch+1}/{args.train_epochs} | "
          f"Avg Loss: {epoch_avg_loss:.4f} | "
          f"Avg it/s: {epoch_avg_it_s:.2f} | "
          f"Time: {epoch_time:.2f}s")

    # 保存模型（按需）
    if (epoch + 1) % args.saving_interval == 0:
        print("Saving model")
        torch.save(model.state_dict(), Path(args.saved_dir) / f"fcn_b{args.train_batch_size}_ep{epoch}.pt")

# Function to evaluate the model.
def evaluate(model,
             num_classes,
             val_loader, 
             device):
    
    evaluator = Evaluator(num_classes)
    evaluator.reset()
    model.eval()

    with torch.no_grad():
        for _, batch in enumerate(val_loader):
            input, target = batch
            input = input.to(device)
            output = model(input)

            pred = output["out"].cpu().numpy()
            pred = np.argmax(pred, axis=1)

            gt = target.squeeze(1).cpu().numpy()
            evaluator.add_batch(gt, pred)

        print(evaluator.Mean_Intersection_over_Union())


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
    parser.add_argument('--num_classes', default=21, type=int,
                        help='Class num')
    parser.add_argument('--saved_dir', default="model", type=str,
                            help='Dir to save model ckpt')
    parser.add_argument('--saving_interval', default=5, type=int,
                            help='Epoch interval to save model ckpt')
    args = parser.parse_args()

    print(vars(args))
    
    model_dir = Path(args.saved_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)

    # Data preprocessing.
    input_size = (args.image_size, args.image_size)

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    target_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.Lambda(lambda img: voc_colormap_to_label(img))
    ])

    # Dataset loading.
    train_dataset = VOCSegmentation(
        root=args.dataset_root,
        year='2007',
        image_set='train',
        download=False,
        transform=transform,
        target_transform=target_transform
    )

    val_dataset = VOCSegmentation(
        root=args.dataset_root,
        year='2007',
        image_set='test',
        download=False,
        transform=transform,
        target_transform=target_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                            shuffle=True, num_workers=2, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=args.infer_batch_size,
                            shuffle=False, num_workers=2, drop_last=True)

    # FCN modeling.
    if args.mode == "infer":
        model = fcn_resnet50(pretrained=True, num_classes=args.num_classes)
    else:
        model = fcn_resnet50(pretrained=False, num_classes=args.num_classes)
    

    # Move the model to the device.
    device = torch.device(args.device)
    model.to(device)

    # Loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001,
                        momentum=0.9, weight_decay=0.0005)

    # Training and evaluation.
    if args.mode in ['train', 'both']:
        if args.train_epochs != 0:
            print(f'[INFO] Start training on {args.device}.')
            start_time = time.time()
            for epoch in range(args.train_epochs):
                print(f'[INFO] Training epoch {epoch + 1}/{args.train_epochs}.')
                train(model, train_loader, criterion, optimizer, device, epoch, args)
            end_time = time.time()
            print(f'Training completed in {end_time - start_time:.2f} seconds.')

    if args.mode in ['infer', 'both']:
        print(f'[INFO] Start inference on {args.device}.')
        evaluate(model, args.num_classes, val_loader, device)
      
if __name__ == "__main__":
    main()
