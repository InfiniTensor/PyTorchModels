import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.datasets import VOCSegmentation
from tqdm import tqdm


# Function to train the model.
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for images, targets in tqdm(train_loader):
        images = images.to(device)
        targets = targets.squeeze(1).long().to(device)
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


# Function to evaluate the model.
def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_time = 0.0
    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = images.to(device)
            targets = targets.squeeze(1).long().to(device)

            start_time = time.time()
            outputs = model(images)['out']
            end_time = time.time()
            total_time += (end_time - start_time)

            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
    return correct, total, total_time


# Argument parsing.
parser = argparse.ArgumentParser()
parser.add_argument("--train-batch-size", default=4, type=int,
                    help="Default value is 4.")
parser.add_argument("--train-epochs", default=2, type=int,
                    help="Default value is 2. Set 0 to this argument if you want an untrained network.")
parser.add_argument("--infer-batch-size", default=1, type=int,
                    help="Default value is 1.")
parser.add_argument("--device", type=str, default="cpu",
                    choices=["cuda", "cpu"], help="Device to use for training and inference.")
parser.add_argument("--image-size", default=256, type=int,
                    help="The width and height of an image.")
args = parser.parse_args()

print(vars(args))

# Data preprocessing.
input_size = (args.image_size, args.image_size)

transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor()
])

# Dataset loading.
train_dataset = VOCSegmentation(
    root='../data',
    year='2007',
    image_set='train',
    download=True,
    transform=transform,
    target_transform=target_transform
)

val_dataset = VOCSegmentation(
    root='../data',
    year='2007',
    image_set='val',
    download=True,
    transform=transform,
    target_transform=target_transform
)

train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                          shuffle=True, num_workers=2, drop_last=True)

val_loader = DataLoader(val_dataset, batch_size=args.infer_batch_size,
                        shuffle=False, num_workers=2, drop_last=True)

# DeepLabV3 modeling.
model = deeplabv3_resnet50(pretrained=False)
model.classifier[4] = nn.Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))

# Move the model to the device.
device = torch.device(args.device)
model.to(device)

# Loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=0.0005)

# Training and evaluation.
if args.train_epochs != 0:
    print(f"[INFO] Start training on {args.device}.")
    start_time = time.time()
    for epoch in range(args.train_epochs):
        print(f"[INFO] Training epoch {epoch + 1}/{args.train_epochs}.")
        train(model, train_loader, criterion, optimizer, device)
    end_time = time.time()
    print(f'Training completed in {end_time - start_time:.2f} seconds.')

correct, total, total_time = evaluate(model, val_loader, device)
accuracy = 100 * correct / total
print(f'Total images: {total}.')
print(f'Accuracy: {accuracy:.2f}%.')
print(
    f'Inference time per batch: {total_time / len(val_loader):.5f} seconds.')
