import os
import torch
import argparse
import time
import numpy as np

from pathlib import Path
from unet import UNet
from torchvision import transforms, datasets
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

num_classes = 21
intersection = torch.zeros(num_classes, num_classes)
union = torch.ones(num_classes, num_classes)

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU
    
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def eval(num_classes):
    evaluator = Evaluator(num_classes)
    evaluator.reset()
    model = UNet(dimensions=num_classes).to(device)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint)
        print(f'Loaded weights from {model_path}')
    else:
        print(f'WARNING: {model_path} not found, using random weights for throughput benchmark')
    model.eval()

    total_inference_time = 0.0
    total_samples = 0
    with torch.no_grad():
        for _, batch in enumerate(datasetloader):
            input, target = batch
            input = input.to(device)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            infer_start = time.time()
            output = model(input)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            infer_end = time.time()
            total_inference_time += (infer_end - infer_start)
            total_samples += input.size(0)

            pred = output.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            gt = target.cpu().numpy()
            evaluator.add_batch(gt, pred)

        print(evaluator.Mean_Intersection_over_Union())

    # Print inference throughput and latency
    if total_samples > 0:
        avg_latency_ms = (total_inference_time / total_samples) * 1000
        throughput = total_samples / total_inference_time
        print(f'\nInference throughput: {throughput:.2f} samples/s')
        print(f'Average inference latency: {avg_latency_ms:.2f} ms/sample')
        print(f'Total inference time: {total_inference_time:.2f} s')
    if torch.cuda.is_available():
        print(f'GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
        print(f'GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB')


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu",
                    choices=["cuda", "cpu"], 
                    help="Device to use for training and inference.")
parser.add_argument("--input_size", default=256, type=int,
                    help="Input size")
parser.add_argument("--classes", type=int, default=21, 
                    help="Num of classes")
parser.add_argument("--dataset_path", type=str, default="../data")
parser.add_argument("--VOC_year", type=str, default="2007")
args = parser.parse_args()

data_folder = args.dataset_path
year = args.VOC_year
model_path = Path("model") / f"unet_b4_ep100.pt"
transform = transforms.Compose([
    transforms.Resize((args.input_size, args.input_size)),
    transforms.ToTensor(), 
    ])

target_transform = transforms.Compose([
    transforms.Resize((args.input_size, args.input_size)),
    transforms.Lambda(lambda img: voc_colormap_to_label(img))
])

dataset = datasets.VOCSegmentation(
    data_folder,
    year=year,
    download=False,
    image_set="test",
    transform=transform,
    target_transform=target_transform
)
datasetloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

if __name__ == "__main__":
    eval(args.classes)