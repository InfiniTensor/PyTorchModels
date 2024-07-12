import os
import torch
import torch.optim as optim
from pathlib import Path
from torch import nn
from torchvision import transforms, datasets
import argparse

from unet import UNet

def train(model, epochs):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        losses = []
        for i, batch in enumerate(datasetloader):
            input, target = batch
            input = input.to(device)
            target = target.type(torch.LongTensor).to(device)
            if input.shape[0] < 2:
                continue
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target.squeeze())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(sum(losses) /len(losses))
        if (epoch + 1) % saving_interval == 0:
            print("Saving model")

        torch.save(model.state_dict(), model_path)
    torch.save(model.state_dict(), model_path)
    return


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=4, type=int,
                    help="Batch size.")
parser.add_argument("--epochs", default=50, type=int,
                    help="Total epochs")
parser.add_argument("--input_size", default=256, type=int,
                    help="Input size")
parser.add_argument("--classes", type=int, default=22, 
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
saving_interval = 10

epochs = args.epochs
input_size = args.input_size
classes = args.classes
batch_size = args.batch_size
year = args.VOC_year
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.Grayscale(),
    transforms.ToTensor(), 
    ])
dataset = datasets.VOCSegmentation(
    data_folder,
    year=year,
    download=False,
    image_set="train",
    transform=transform,
    target_transform=transform,
)
datasetloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = UNet(classes)
model.to(device)
if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
optimizer = optim.RMSprop(
    model.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9
)

if __name__ == "__main__":
    train(model, epochs)
