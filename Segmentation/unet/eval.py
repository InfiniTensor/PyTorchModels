import torch
import argparse
import numpy as np

from pathlib import Path
from unet import UNet
from torchvision import transforms, datasets

def is_mlu_available():
    if hasattr(torch, 'is_mlu_available'):
        return torch.is_mlu_available()
    try:
        import torch_mlu
        return torch.mlu.is_available()
    except:
        return False

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

def eval(model, datasetloader, num_classes, device):
    evaluator = Evaluator(num_classes)
    evaluator.reset()
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for _, batch in enumerate(datasetloader):
            input, target = batch
            input = input.to(device)
            output = model(input)

            pred = output["out"].cpu().numpy() if isinstance(output, dict) else output.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            gt = target.squeeze(1).cpu().numpy()
            evaluator.add_batch(gt, pred)

        print(evaluator.Mean_Intersection_over_Union())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str,
                      default="mlu" if is_mlu_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
                      choices=['cuda', 'cpu', 'mlu'],
                      help='Device to use for training and inference.')
    parser.add_argument("--input_size", default=256, type=int,
                      help="Input size")
    parser.add_argument("--classes", type=int, default=21,
                      help="Num of classes")
    parser.add_argument("--dataset_path", type=str, default="../data")
    parser.add_argument("--VOC_year", type=str, default="2007")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to model weights file")
    args = parser.parse_args()

    try:
        if args.device == "mlu":
            args.device = torch.device("mlu:0")
            test_tensor = torch.tensor([1.0]).to(args.device)
            print(f"Successfully initialized MLU device: {args.device}")
        else:
            args.device = torch.device("cpu")
            print("Using CPU")
    except Exception as e:
        print(f"Device initialization failed: {e}")
        args.device = torch.device("cpu")
        print("Fall back to CPU")

    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(), 
    ])

    target_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.Lambda(lambda img: voc_colormap_to_label(img))
    ])

    dataset = datasets.VOCSegmentation(
        root="/dataset",
        year=args.VOC_year,
        download=False,
        image_set="val",
        transform=transform,
        target_transform=target_transform
    )
    datasetloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # 加载模型
    model = UNet(dimensions=args.classes)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 执行评估
    eval(model, datasetloader, args.classes, args.device)

if __name__ == "__main__":
    main()
