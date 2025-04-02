import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
import argparse  # 导入 argparse 用于命令行参数解析

def parse_args():
    """
    解析命令行参数并设置默认值
    """
    parser = argparse.ArgumentParser(description="Train SSD300 model on VOC dataset")

    # Data parameters
    parser.add_argument('--data', type=str, default='./data', help="Folder containing data files")
    parser.add_argument('--keep_difficult', type=bool, default=True, help="Whether to use difficult objects in dataset")

    # Model parameters
    parser.add_argument('--n_classes', type=int, default=len(label_map), help="Number of different types of objects")
    parser.add_argument('--device', type=str, default="cuda", choices=["cpu", "cuda"], help="Device to use for training")

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--iterations', type=int, default=120000, help="Number of iterations to train")
    parser.add_argument('--workers', type=int, default=4, help="Number of workers for data loading")
    parser.add_argument('--print_freq', type=int, default=200, help="Frequency of printing training status")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--decay_lr_at', type=int, nargs='*', default=[80000, 100000], help="Decay learning rate at these many iterations")
    parser.add_argument('--decay_lr_to', type=float, default=0.1, help="Fraction to decay learning rate to")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="Weight decay")
    parser.add_argument('--grad_clip', type=float, default=None, help="Clip gradients if necessary")

    # Checkpoint parameters
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to model checkpoint, None if none")

    return parser.parse_args()

def main():
    args = parse_args()  # 获取命令行参数

    cudnn.benchmark = True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Initialize model or load checkpoint
    if args.checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=args.n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * args.lr}, {'params': not_biases}],
                                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    else:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move model to device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(args.data,
                                     split='train',
                                     keep_difficult=args.keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=args.workers,
                                               pin_memory=True)

    # Calculate total number of epochs to train and the epochs to decay learning rate at
    epochs = args.iterations // (len(train_dataset) // 32)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in args.decay_lr_at]

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, args.decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              args=args)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch, args):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if args.grad_clip is not None:
            clip_gradient(optimizer, args.grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()
