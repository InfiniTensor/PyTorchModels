import os
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import copy
import numpy as np
import argparse
import logging
from typing import Optional, Tuple
from lstm import CustomLSTM
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_directions = 1
        self.lstm = CustomLSTM(self.input_size,self.hidden_size,self.num_layers,batch_first=True)
        self.linear = nn.Linear(self.hidden_size,self.output_size)
 
    def forward(self,inputseq):
        h_0 = torch.randn(self.num_directions*self.num_layers,self.batch_size,self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        output,_ = self.lstm(inputseq,(h_0,c_0))
        pred = self.linear(output[:,-1,:])
        return pred
def process_data(data,N):
    XY=[]
    for i in range(len(data) - N):
        X = []
        Y = []
        for j in range(N):
            X.append(data.iloc[i+j, 1])
        Y.append(data.iloc[i + N, 1])
        X = torch.FloatTensor(X).view(-1,1)
        Y = torch.FloatTensor(Y)
        XY.append((X,Y))
    return XY
 
class MyDataset(Dataset):
    def __init__(self,data):
        self.data = data
 
    def __getitem__(self, item):
        return self.data[item]
 
    def __len__(self):
        return len(self.data)
 
def data_loader(data,N,batch_size,shuffle):
    seq = process_data(data, N)
    seq_set = MyDataset(seq)
    seq = DataLoader(dataset=seq_set,batch_size=batch_size,shuffle=shuffle,drop_last=True)
    return seq_set,seq
def read_data(filename):
    data = pd.read_csv(filename,skiprows=1)
    data.head(5)
    L = data.shape[0]
    logger.info("data的尺寸为：{}".format(data.shape))
    # logger.info("文件中有nan行共{}�?.format(data.isnull().sum(axis=1)))
    return data,L

def train_proc(para_dict, train_data, val_data):
    input_size = para_dict["input_size"]
    hidden_size = para_dict["hidden_size"]
    num_layers = para_dict["num_layers"]
    output_size = para_dict["output_size"]
    batch_size = para_dict["batch_size"]
    lr = para_dict["lr"]
    epoch = para_dict["epoch"]
    model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size)
    model.to(device)
    
    # 优化器保存当前的状态，并可以进行参数的更新
    if para_dict["optimizer"] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr)
    if para_dict["loss_function"] == 'mse':
        loss_function = nn.MSELoss()

    best_model = None
    min_val_loss = float('inf')
    train_loss = []
    val_loss = []

    for epoch_idx in range(epoch):
        epoch_start_time = time.time()
        train_loss_tmp = 0
        val_loss_tmp = 0
        total_batches = len(train_data)
        
        # 训练
        model.train()
        train_bar = tqdm(enumerate(train_data), total=total_batches, 
                        desc=f'Epoch {epoch_idx+1}/{epoch}', leave=True)
        
        batch_times = []
        for step, curdata in train_bar:
            batch_start_time = time.time()
            
            seq, label = curdata
            seq = seq.to(device)
            label = label.to(device)
            
            # 计算网络输出
            y_pred = model(seq)
            # 计算损失
            loss = loss_function(y_pred, label)
            train_loss_tmp += loss.item()
            
            # 计算梯度和反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算batch耗时
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            it_per_sec = 1.0 / batch_time if batch_time > 0 else 0
            
            # 更新tqdm进度条信息
            train_bar.set_postfix({
                'Batch_loss': f'{loss.item():.5f}'
            })
            
            #if step % 10 == 0:
            #    logger.info(f"epoch={epoch_idx} step={step}/{total_batches}: loss = {loss.item():05f}")

        # 计算epoch统计信息
        epoch_time = time.time() - epoch_start_time
        avg_loss = train_loss_tmp / total_batches
        avg_it_per_sec = total_batches / epoch_time
        
        # 打印epoch统计信息
        print(f"\nEpoch {epoch_idx+1:3d}/{epoch} | Avg Loss: {avg_loss:.4f} | "
              f"Avg it/s: {avg_it_per_sec:.2f} | Time: {epoch_time:.2f}s")
        
        # 验证
        model.eval()
        val_bar = tqdm(val_data, total=len(val_data), desc='Validating', leave=False)
        for seq, label in val_bar:
            seq = seq.to(device)
            label = label.to(device)
            with torch.no_grad():
                y_pred = model(seq)
                loss = loss_function(y_pred, label)
                val_loss_tmp += loss.item()
                val_bar.set_postfix({'val_loss': f'{loss.item():.5f}'})

        # 最优模型
        if val_loss_tmp < min_val_loss:
            min_val_loss = val_loss_tmp
            best_model = copy.deepcopy(model)
            torch.save({'models': best_model.state_dict()}, 
                      os.path.join(para_dict["modelpara_path"], 'lstm_best.pt'))
        
        # 损失保存
        train_loss_tmp /= len(train_data)
        val_loss_tmp /= len(val_data)
        train_loss.append(train_loss_tmp)
        val_loss.append(val_loss_tmp)
        logger.info(f"Val: epoch={epoch_idx+1:3d}/{epoch}: train_loss = {train_loss_tmp:05f}, val_loss = {val_loss_tmp:05f}")

        # 保存模型
        if epoch_idx % para_dict['save_step'] == 0:
            state = {'models': model.state_dict()}
            torch.save(state, os.path.join(para_dict["modelpara_path"], 
                      f'lstm_epoch_{epoch_idx}.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM')
    parser.add_argument('--dataset', default='complete_data.csv', type=str,help='dataset path')
    parser.add_argument('--input_size', default=1, type=int,help='')
    parser.add_argument('--output_size', default=1, type=int,help='')
    parser.add_argument('--epoch', default=200, type=int,help='train epoch')
    parser.add_argument('--batch_size', default=512, type=int,help='train batch_size')
    parser.add_argument('--hidden_size', default=64, type=int,help='')
    parser.add_argument('--num_layers', default=3, type=int,help='')
    parser.add_argument('--seq_len', default=24, type=int,help='')
    parser.add_argument('--modelpara_path', default='./checkpoints', type=str,help='model output path')
    parser.add_argument('--loss_function', default='mse', type=str,help='loss function')
    parser.add_argument('--optimizer', default='Adam', type=str,help='optimizer')
    parser.add_argument('--lr', default=0.0001, type=float,help='learning rate')
    parser.add_argument('--save_step', default=10, type=int,help='model save step')
    args = parser.parse_args()
    para_dict = vars(args)

    if not os.path.exists(para_dict["modelpara_path"]):
        os.makedirs(para_dict["modelpara_path"])

    device="cuda"
    torch.cuda.set_device(0)

    data,L = read_data(para_dict['dataset'])
    min_val = min(data.iloc[:,1])
    max_val = max(data.iloc[:,1])
    data.iloc[:, 1] = (data.iloc[:,1]-min_val)/(max_val-min_val)
 
    train_pro = 0.9
    val_pro = 0.95
    test_pro = 1
    train = data.loc[:len(data)*train_pro,:]
    val = data.loc[len(data)*train_pro+1:len(data)*val_pro,:]

    logger.info("训练集的大小为{}".format(train.shape))
    logger.info("验证集的大小为{}".format(val.shape))
 
    batch_size = para_dict["batch_size"]
    N = para_dict["seq_len"]
    train_data_set,train_data = data_loader(train,N,batch_size,True)
    logger.info('训练数据导入完毕')
    val_data_set,val_data = data_loader(val, N,batch_size,True)
    logger.info('验证数据导入完毕')
 
    logger.info('开始训练')
    train_proc(para_dict,train_data,val_data)
    logger.info('训练完毕！！！！')
