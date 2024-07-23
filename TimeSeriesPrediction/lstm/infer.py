import torch
import torch_mlu
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
    if args.infer:
        X = torch.FloatTensor([data.iloc[:, 1]]).view(-1,1)
        XY.append((X))
        print(XY,11111111)
    else:
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
    # logger.info("文件中有nan行共{}".format(data.isnull().sum(axis=1)))
    return data,L

def infer_proc(para_dict,infer_data,min_val,max_val):
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
    for curdata in infer_data:
        seq = curdata
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
        for j in range(len(y_pred)):
            y = y_pred[j].item()*(max_val-min_val)+min_val
            pred.append(y)
    print("预测结果为：",pred)


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
        print(seq.item()*(max_val-min_val)+min_val, label.item()*(max_val-min_val)+min_val,type(seq),type(label))
        seq = seq.to(device)
        label = label.to(device)
        with torch.no_grad():
            y_pred = model(seq)
        for j in range(len(y_pred)):
            y = y_pred[j].item()*(max_val-min_val)+min_val
            lb = label[j].item()*(max_val-min_val)+min_val
            pred.append(y)
            labels.append(lb)
    print(pred,labels)
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
    parser.add_argument('--infer', default=True,  action='store_true',help='')
    parser.add_argument('--infer_data', default='"2021-08-25 03:45:00",200981.7805',type=str,help='input a data,such as "2021-08-25 03:45:00",200981.7805')
    
    args = parser.parse_args()
    
    para_dict = vars(args)
    
    device="mlu"
    torch.mlu.set_device(device)
    data,L = read_data(para_dict['dataset'])
    train_pro = 0.9
    val_pro = 0.95
    test_pro = 1
    train = data.loc[:len(data)*train_pro,:]
    val = data.loc[len(data)*train_pro+1:len(data)*val_pro,:]
    test = data.loc[len(data)*val_pro+1:len(data)*test_pro,:]
    min_val = min(data.iloc[:,1])
    max_val = max(data.iloc[:,1])
    data.iloc[:, 1] = (data.iloc[:,1]-min_val)/(max_val-min_val)
    batch_size = para_dict["batch_size"]
    N = para_dict["seq_len"]
    if args.infer:
        time,power=args.infer_data.split(",")
        # print(time,power)
        # data_pre = pd.DataFrame([[time,float(power)]])
        data_pre = pd.DataFrame([["2021-08-25 03:45:00",258164.1828]])
        data_pre.iloc[:, 1] = (data_pre.iloc[:,1]-min_val)/(max_val-min_val)
        infer=data_pre.loc[:]
        # infer_data_set,infer_data = data_loader(infer, 1,1,False)
        X=process_data(data_pre,1)
        seq_set = MyDataset(X)
        seq = DataLoader(dataset=seq_set,batch_size=1,shuffle=False,drop_last=True)
        logger.info("开始预测")
        para_dict["batch_size"] = 1
        infer_proc(para_dict, seq,min_val,max_val)
        logger.info('预测完毕！！！！')
    else:
        test_data_set,test_data = data_loader(test, N,batch_size,False)
        logger.info('测试数据导入完毕')
        logger.info("开始评估")
        test_proc(para_dict, test_data,min_val,max_val)
        logger.info('评估完毕！！！！')