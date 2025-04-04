import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import torchvision.transforms as transforms

from data_utils import DatasetFromFolder
from model import Net
from psnrmeter import PSNRMeter

# 自定义简单的 AverageMeter
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def add(self, value):
        self.sum += value
        self.count += 1

    def value(self):
        return self.sum / self.count if self.count > 0 else 0


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    meter_loss = AverageMeter()
    meter_psnr = PSNRMeter()
    for data, target in tqdm(train_loader, desc="Training"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        meter_loss.add(loss.item())
        meter_psnr.add(output, target)

    return meter_loss.value(), meter_psnr.value()


def validate(model, val_loader, criterion, device):
    model.eval()
    meter_loss = AverageMeter()
    meter_psnr = PSNRMeter()
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            meter_loss.add(loss.item())
            meter_psnr.add(output, target)

    return meter_loss.value(), meter_psnr.value()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Super Resolution')
    parser.add_argument('--upscale_factor', default=3, type=int, help='super resolution upscale factor')
    parser.add_argument('--num_epochs', default=100, type=int, help='super resolution epochs number')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    train_set = DatasetFromFolder(
        'data/train', upscale_factor=UPSCALE_FACTOR,
        input_transform=transforms.ToTensor(), target_transform=transforms.ToTensor()
    )
    val_set = DatasetFromFolder(
        'data/val', upscale_factor=UPSCALE_FACTOR,
        input_transform=transforms.ToTensor(), target_transform=transforms.ToTensor()
    )
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net(upscale_factor=UPSCALE_FACTOR).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    print('# parameters:', sum(param.numel() for param in model.parameters()))

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        # Train
        train_loss, train_psnr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"[Train] Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f} dB")

        # Validate
        val_loss, val_psnr = validate(model, val_loader, criterion, device)
        print(f"[Val] Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f} dB")

        # Save checkpoint
        torch.save(model.state_dict(), f'epochs/epoch_{UPSCALE_FACTOR}_{epoch}.pt')

        # Step scheduler
        scheduler.step()