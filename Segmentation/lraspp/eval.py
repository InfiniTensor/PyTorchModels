import torch
import argparse
import numpy as np
import sys

from pathlib import Path
from torchvision import transforms, datasets
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

sys.path.append("../")
from bench.evaluator import Evaluator

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


def eval(model,
         num_classes,
         val_loader,
         device):
    evaluator = Evaluator(num_classes)
    evaluator.reset()
    model = model.to(device)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                        choices=["cuda", "cpu"], 
                        help="Device to use for training and inference.")
    parser.add_argument("--input_size", default=256, type=int,
                        help="Input size")
    parser.add_argument("--dataset_path", type=str, default="../data")
    parser.add_argument("--VOC_year", type=str, default="2007")
    parser.add_argument("--num_classes", type=int, default=21)
    args = parser.parse_args()

    print(vars(args))

    data_folder = args.dataset_path
    year = args.VOC_year

    # Prepare data
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

    # Prepare model
    model = lraspp_mobilenet_v3_large(pretrained=True, num_classes=args.num_classes) 
    
    print(f'[INFO] Start inference on {args.device}.')
    eval(model, args.num_classes, datasetloader, args.device)

if __name__ == "__main__":
    main()