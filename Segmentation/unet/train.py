import os
import torch
import numpy as np
import cv2
import torch.optim as optim
import torchvision.transforms.functional as F
from pathlib import Path
from torch import nn
from torchvision import transforms, datasets
import argparse
from tqdm.auto import tqdm
import time

from unet import UNet

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

def train(model,
          epochs,
          batch_size,
          datasetloader,
          device,
          optimizer,
          saving_interval,
          model_path):

    model_dir = model_path.parent
    model_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        sample_cnt = 0
        epoch_start_time = time.time()
        batch_losses = []
        batch_it_s = []

        # 使用tqdm显示实时训练信息
        pbar = tqdm(datasetloader, 
                   desc=f"Epoch {epoch+1}/{epochs}")

        for step, (inputs, targets) in enumerate(pbar, start=1):
            batch_start_time = time.time()
            
            # 数据移至设备
            inputs = inputs.to(device)
            targets = targets.to(device, dtype=torch.long).squeeze()
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()

            # 计算当前batch指标
            batch_time = time.time() - batch_start_time
            current_it_s = 1 / batch_time if batch_time > 0 else 0
            current_loss = loss.item()

            # 记录batch数据
            batch_losses.append(current_loss)
            batch_it_s.append(current_it_s)
            epoch_loss_sum += current_loss * inputs.size(0)
            sample_cnt += inputs.size(0)

            # 更新进度条信息
            pbar.set_postfix({
                "batch_loss": f"{current_loss:.4f}"    
            })

        # 计算epoch级指标
        epoch_time = time.time() - epoch_start_time
        epoch_avg_loss = epoch_loss_sum / sample_cnt
        epoch_avg_it_s = len(datasetloader) / epoch_time if epoch_time > 0 else 0

        # 打印epoch总结
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Avg Loss: {epoch_avg_loss:.4f} | "
              f"Avg it/s: {epoch_avg_it_s:.2f} | "
              f"Time: {epoch_time:.2f}s")

        # 保存检查点
        if (epoch + 1) % saving_interval == 0:
            ckpt_path = model_dir / f"unet_b{batch_size}_ep{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"✔ Saved model to {ckpt_path}")

    # 训练完成后保存最终模型
    torch.save(model.state_dict(), model_path)
    print(f"✔ Final model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int,
                        help="Batch size.")
    parser.add_argument("--epochs", default=50, type=int,
                        help="Total epochs")
    parser.add_argument("--input_size", default=256, type=int,
                        help="Input size")
    parser.add_argument("--classes", type=int, default=21, 
                        help="Num of classes")
    parser.add_argument("--dataset_path", type=str, default="../data")
    parser.add_argument("--VOC_year", type=str, default="2007")
    parser.add_argument("--saved_dir", type=str, default="./model")
    args = parser.parse_args()

    print(args)

    data_folder = args.dataset_path 
    model_folder = Path(args.saved_dir)
    model_folder.mkdir(exist_ok=True)
    model_path = model_folder / f"unet_b{args.batch_size}_ep{args.epochs}.pt"
    saving_interval = 50

    epochs = args.epochs
    input_size = args.input_size
    classes = args.classes
    batch_size = args.batch_size
    year = args.VOC_year
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    target_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.Lambda(lambda img: voc_colormap_to_label(img))
    ])

    dataset = datasets.VOCSegmentation(
        data_folder,
        year=year,
        download=False,
        image_set="train",
        transform=transform,
        target_transform=target_transform
    )
    datasetloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = UNet(classes)
    model.to(device)
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    optimizer=optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train(model,
          epochs,
          batch_size,
          datasetloader,
          device,
          optimizer,
          saving_interval,
          model_path) 
    
if __name__ == "__main__":
    main()
