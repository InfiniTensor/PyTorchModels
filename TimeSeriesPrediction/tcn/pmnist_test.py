import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from utils import data_generator
from model import TCN
import numpy as np
import argparse
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

root = '../data/mnist'
batch_size = args.batch_size
n_classes = 10
input_channels = 1
seq_length = int(784 / input_channels)
epochs = args.epochs
steps = 0

print(args)
train_loader, test_loader = data_generator(root, batch_size)

permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)

if args.cuda:
    model.cuda()
    permute = permute.cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

def train(ep):
    global steps
    train_loss = 0
    model.train()
    
    epoch_start_time = time.time()
    total_batches = len(train_loader)
    batch_times = []
    
    # 使用tqdm包装train_loader
    train_bar = tqdm(enumerate(train_loader), total=total_batches, 
                    desc=f'Epoch {ep}', leave=True)
    
    for batch_idx, (data, target) in train_bar:
        batch_start_time = time.time()
        
        if args.cuda: 
            data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, seq_length)
        if args.permute:
            data = data[:, :, permute]
        data, target = Variable(data), Variable(target)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        
        train_loss += loss.item()
        steps += seq_length
        
        # 计算batch耗时
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        
        # 更新进度条信息
        train_bar.set_postfix({
            'Batch_loss': f'{loss.item():.6f}',
        })
        
    # 计算epoch统计信息
    epoch_time = time.time() - epoch_start_time
    avg_loss = train_loss / total_batches
    avg_it_per_sec = total_batches / epoch_time
    
    # 打印epoch统计信息
    print(f"\nEpoch {ep:3d} | Avg Loss: {avg_loss:.4f} | "
          f"Avg it/s: {avg_it_per_sec:.2f} | Time: {epoch_time:.2f}s\n")

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return test_loss


if __name__ == "__main__":
    for epoch in range(1, epochs+1):
        train(epoch)
        test()
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
