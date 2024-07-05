import os
import torch
import torch.optim as optim
from pathlib import Path
from torch import nn
from torchvision import transforms, datasets
import argparse
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

def train(epochs):
    model = lraspp_mobilenet_v3_large(weights=None, num_classes=classes)
    model.to(device)
    optimizer = optim.RMSprop(
        model.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        losses = []
        for i, batch in enumerate(datasetloader):
            input, target = batch
            input = input.to(device)
            target = target.squeeze(1).long().to(device)
            if input.shape[0] < 2:
                continue
            optimizer.zero_grad()
            output = model(input)['out']
            loss = criterion(output, target)
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
parser.add_argument("--classes", type=int, default=21, 
                    help="Num of classes")
parser.add_argument("--dataset_path", type=str, default="./data")
parser.add_argument("--VOC_year", type=str, default="2007")
args = parser.parse_args()


data_folder = "./data"
model_folder = Path("./model")
model_folder.mkdir(exist_ok=True)
model_path = "./model/lraspp-voc.pt"
saving_interval = 10

epochs = args.epochs
input_size = args.input_size
classes = args.classes
batch_size = args.batch_size
year = args.VOC_year
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.Resize((input_size, input_size)),
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
datasetloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    train(epochs)
