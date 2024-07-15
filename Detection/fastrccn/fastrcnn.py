import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Sampler
import time
import random
import argparse
import sys
from tqdm import tqdm

# Set environment variable for CUDA visibility
os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # 使用 GPU 5
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Define argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--train_batch", default=30, type=int, help="Default value of train_batch is 30.")
parser.add_argument("--train_epoch", default=2, type=int, help="Default value of train_epoch is 2. Set 0 to this argument if you want an untrained network.")
parser.add_argument("--infer_batch", default=1, type=int, help="Default value of infer_batch is 1.")
parser.add_argument("--which_device", type=str, default="cuda", choices=["mlu", "cuda", "cpu", "xpu"])
parser.add_argument("--image_size", default=224, type=int, help="The width/height of image")
parser.add_argument("--sample", default=1.0, type=float, help="The percentage of the all images which to be infer.")
args = parser.parse_args()

print(vars(args))

# Define data preprocessing and data loader
class PercentageSampler(Sampler):
    def __init__(self, data_source, percentage):
        self.data_source = data_source
        self.percentage = percentage
        self.num_samples = int(len(data_source) * percentage)

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        random.shuffle(indices)
        return iter(indices[:self.num_samples])

    def __len__(self):
        return self.num_samples

def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)

transform = transforms.Compose([
    transforms.ToTensor(),
])

class CustomCocoDetection(CocoDetection):
    def __getitem__(self, index):
        img, target = super(CustomCocoDetection, self).__getitem__(index)
        boxes = [obj['bbox'] for obj in target]
        labels = [obj['category_id'] for obj in target]

        # 过滤掉无效的 bounding box
        valid_boxes = []
        valid_labels = []
        for box, label in zip(boxes, labels):
            x_min, y_min, width, height = box
            if width > 0 and height > 0:
                valid_boxes.append([x_min, y_min, x_min + width, y_min + height])
                valid_labels.append(label)

        # 如果没有有效的 bounding box，使用占位符
        if len(valid_boxes) == 0:
            print(f"Warning: No valid bbox at index {index}")
            valid_boxes = [[0, 0, 1, 1]]
            valid_labels = [0]

        # Convert to tensors
        boxes = torch.as_tensor(valid_boxes, dtype=torch.float32)
        labels = torch.as_tensor(valid_labels, dtype=torch.int64)

        # 确保 boxes 至少是 2 维的
        if boxes.ndim == 1:
            print(f"Warning: Incorrect bbox dimension at index {index}, bbox: {boxes}")
            boxes = boxes.unsqueeze(0)

        # Create the target dictionary
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        return img, target

def get_coco_loader(root, ann_file, transform, batch_size, sample_percentage, shuffle=False):
    dataset = CustomCocoDetection(root=root, annFile=ann_file, transform=transform)
    sampler = PercentageSampler(dataset, sample_percentage)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        collate_fn=collate_fn,
        sampler=sampler if not shuffle else None
    )
    return loader

train_loader = get_coco_loader(
    root="/data1/shared/Dataset/coco/images/train2017",
    ann_file="/data1/shared/Dataset/coco/images/annotations/instances_train2017.json",
    transform=transform, 
    batch_size=args.train_batch, 
    sample_percentage=0.5,  # 设置抽样比例为 50%
    shuffle=True
)

test_loader = get_coco_loader(
    root="/data1/shared/Dataset/coco/images/val2017",
    ann_file="/data1/shared/Dataset/coco/images/annotations/instances_val2017.json",
    transform=transform, 
    batch_size=args.infer_batch, 
    sample_percentage=args.sample,
    shuffle=False
)

# Load FastRCNN model
device = torch.device(args.which_device)
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 91  # COCO dataset has 91 classes

# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# Replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Define loss function and optimizer
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Train the network
if args.train_epoch != 0:
    model.to(device)
    model.train()
    for epoch in range(args.train_epoch):
        print(f"[INFO] Training {epoch} epoch...")
        running_loss = 0.0
        for images, targets in tqdm(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: torch.as_tensor(v).to(device) for k, v in target.items()} for target in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            running_loss += losses.item()
        print(f"Epoch {epoch} loss: {running_loss / len(train_loader)}")

# Evaluate the network
model.eval()
total = 0
correct = 0
total_time = 0.0
with torch.no_grad():
    for images, targets in tqdm(test_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        start_time = time.time()
        outputs = model(images)
        end_time = time.time()
        total_time += (end_time - start_time)
        for target, output in zip(targets, outputs):
            total += len(target["labels"])
            correct += sum(p == t for p, t in zip(output["labels"], target["labels"]))
    print(f"Accuracy: {100 * correct / total}%")
    print(f"Inference time per batch: {total_time / len(test_loader)} seconds")
