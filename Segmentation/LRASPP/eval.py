import torch
import argparse
from torchvision import transforms, datasets
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval(num_classes):
    model = lraspp_mobilenet_v3_large(weights=None, num_classes=classes)
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input, _ = batch
            input = input.to(device)
            model(input)
    return

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu",
                    choices=["cuda", "cpu"], 
                    help="Device to use for training and inference.")
parser.add_argument("--input_size", default=256, type=int,
                    help="Input size")
parser.add_argument("--classes", type=int, default=21, 
                    help="Num of classes")
parser.add_argument("--dataset_path", type=str, default="./data")
parser.add_argument("--VOC_year", type=str, default="2007")
args = parser.parse_args()

input_size = args.input_size
classes = args.classes
data_folder = args.dataset_path
year = args.VOC_year
model_path = "./model/lraspp-voc.pt"


transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),                     
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

target_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor()
])

dataset = datasets.VOCSegmentation(
    data_folder,
    year=year,
    download=False,
    image_set="train",
    transform=transform,
    target_transform=target_transform,
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

if __name__ == "__main__":
    eval(args.classes)
