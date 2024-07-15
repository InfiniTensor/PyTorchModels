import os
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch import nn
from torchvision import transforms, datasets
import argparse
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

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
    criterion = nn.CrossEntropyLoss()
    model_dir = model_path.parent
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
            torch.save(model.state_dict(), model_dir / f"lraspp_b{batch_size}_ep{epoch}.pt")
    torch.save(model.state_dict(), model_path)
    return

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
    model_path = model_folder / f"lraspp_b{args.batch_size}_ep{args.epochs}.pt"
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )])

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
        target_transform=target_transform,
    )
    datasetloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = lraspp_mobilenet_v3_large(weights=None, num_classes=classes)
    model.to(device)
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    optimizer = optim.RMSprop(
        model.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9
    )
    
    train(model,
          epochs,
          batch_size,
          datasetloader,
          device,
          optimizer,
          saving_interval,
          model_path,
    )
    
if __name__ == "__main__":
    main()
