import numpy as np
from PIL import Image
import torch
import argparse

from unet import UNet
from torchvision import transforms, utils, datasets
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval(num_classes):
    model = UNet(dimensions=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input, _ = batch
            output = model(input).detach()

    return

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu",
                    choices=["cuda", "cpu"], 
                    help="Device to use for training and inference.")
parser.add_argument("--input_size", default=256, type=int,
                    help="Input size")
parser.add_argument("--classes", type=int, default=22, 
                    help="Num of classes")
args = parser.parse_args()

data_folder = "data"
model_path = "model/unet-voc.pt"

transform = transforms.Compose([transforms.Resize((args.input_size, args.input_size)), transforms.ToTensor(), transforms.Grayscale()])
dataset = datasets.VOCSegmentation(
    data_folder,
    year="2007",
    download=True,
    image_set="train",
    transform=transform,
    target_transform=transform,
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

if __name__ == "__main__":
    eval(args.classes)

