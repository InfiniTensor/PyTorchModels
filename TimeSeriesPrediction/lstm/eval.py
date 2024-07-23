import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import numpy as np
import argparse
import logging
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
        self.lstm = nn.LSTM(self.input_size,self.hidden_size,self.num_layers,batch_first=True)
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


def test_proc(para_dict,test_data,min_val,max_val):
    input_size=para_dict["input_size"]
    hidden_size = para_dict["hidden_size"]
    num_layers = para_dict["num_layers"]
    output_size = para_dict["output_size"]
    batch_size = para_dict["batch_size"]
    path = para_dict["model_path"]
    model = LSTM(input_size,hidden_size,num_layers,output_size,batch_size)
    model.to(device)
    logger.info("loading models ......")
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
 
    pred = []#list
    labels = []
    for curdata in test_data:
        seq, label = curdata
        seq = seq.to(device)
        label = label.to(device)
        with torch.no_grad():
            y_pred = model(seq)
        for j in range(len(y_pred)):
            y = y_pred[j].item()*(max_val-min_val)+min_val
            lb = label[j].item()*(max_val-min_val)+min_val
            pred.append(y)
            labels.append(lb)
    errs = np.array(pred)-np.array(labels)
    avg_err = abs(errs/np.array(labels)).sum()/len(errs)
    avg_acc = 1 - float(avg_err)

    logger.info('评估结果：Acc={:.3f}%'.format(avg_acc*100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM')
    parser.add_argument('--dataset', default='complete_data.csv', type=str,help='dataset path')
    parser.add_argument('--input_size', default=1, type=int,help='')
    parser.add_argument('--output_size', default=1, type=int,help='')
    parser.add_argument('--batch_size', default=512, type=int,help='train batch_size')
    parser.add_argument('--hidden_size', default=64, type=int,help='')
    parser.add_argument('--num_layers', default=3, type=int,help='')
    parser.add_argument('--seq_len', default=24, type=int,help='')
    parser.add_argument('--model_path', default='./checkpoints/lstm_best.pt', type=str,help='')
    
    args = parser.parse_args()
    
    para_dict = vars(args)
    
    device="cuda"
    torch.cuda.set_device(0)

    data,L = read_data(para_dict['dataset'])
    # data = torch.Tensor(data).to(device)
    # 归一�?
    min_val = min(data.iloc[:,1])
    max_val = max(data.iloc[:,1])
    data.iloc[:, 1] = (data.iloc[:,1]-min_val)/(max_val-min_val)
 
    train_pro = 0.9
    val_pro = 0.95
    test_pro = 1
    train = data.loc[:len(data)*train_pro,:]
    val = data.loc[len(data)*train_pro+1:len(data)*val_pro,:]
    test = data.loc[len(data)*val_pro+1:len(data)*test_pro,:]
 
    batch_size = para_dict["batch_size"]
    N = para_dict["seq_len"]

    test_data_set,test_data = data_loader(test, N,batch_size,False)
    logger.info('测试数据导入完毕')

    logger.info("开始评估")
    test_proc(para_dict, test_data,min_val,max_val)
    logger.info('评估完毕！！！！')