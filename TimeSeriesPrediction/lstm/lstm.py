import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import argparse
import os


def fetch_data(dataset: str, root, scaler):
    if dataset == "jenaclimate":
        data_path = os.path.join(root, "jena_climate_2009_2016.csv.zip")
        df = pd.read_csv(data_path, compression='zip')
        # 选择需要的列并处理时间列
        df = df[['Date Time', 'T (degC)']]
        df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S') 
        df.set_index('Date Time', inplace=True)
        # 填充缺失值
        df = df.fillna(method='ffill')
        # 数据标准化
        df['T (degC)'] = scaler.fit_transform(df[['T (degC)']]) 
    elif dataset == "electricity":
        data_path = os.path.join(root, "household_power_consumption.zip")
        df = pd.read_csv(data_path, sep=';', parse_dates={'datetime': ['Date', 'Time']}, infer_datetime_format=True, low_memory=False, na_values=['nan','?'], index_col='datetime')
        df = df[['Global_active_power']].resample('H').mean().fillna(method='ffill')
        df['Global_active_power'] = scaler.fit_transform(df[['Global_active_power']])
    elif dataset == "airpassengers":
        data_path = os.path.join(root, "airline-passengers.csv")
        df = pd.read_csv(data_path, parse_dates=['Month'], index_col='Month')
        df = df.fillna(method='ffill')
        df['Passengers'] = scaler.fit_transform(df[['Passengers']])
    else:
        print(f"please choose dataset in [jenaclimate, electricity, airpassengers] .")
    
    return df
        

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
        self.sequences = self.create_sequences(data, seq_length)

    def create_sequences(self, data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            sequence = data[i:i + seq_length]
            label = data[i + seq_length]
            sequences.append((sequence, label))
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    
    
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        
        self.register_buffer('h0', torch.zeros(num_layers, 1, hidden_layer_size))
        self.register_buffer('c0', torch.zeros(num_layers, 1, hidden_layer_size))

    def forward(self, input_seq):
        # h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).requires_grad_()
        # c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).requires_grad_()
        batch_size = input_seq.shape[0]
        
        # 扩展 h0 和 c0 以匹配当前 batch size
        h0 = self.h0.expand(-1, batch_size, -1).contiguous()
        c0 = self.c0.expand(-1, batch_size, -1).contiguous()
        
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        predictions = self.linear(lstm_out[:, -1, :])
        
        return predictions   


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_length", default=24, type=int,
                        help="Sequence length.")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batchsize")
    parser.add_argument("--train_eval_split", default=0.8, type=float,
                        help="Train eval split percentage")
    parser.add_argument("--epochs", type=int, default=4, 
                        help="Training epochs")
    parser.add_argument("--saved_dir", type=str, default="./model")
    parser.add_argument("--dataset", type=str, default="jenaclimate")
    parser.add_argument("--mode", type=str, default="both", choices=["train", "infer", "both"] )
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), type=str,
                        choices=['cuda', 'cpu'], help='Device to use for training and inference.')
    parser.add_argument('--weights', type=str, default="",  help='Weight path for inference.')
    parser.add_argument('--data', type=str, default="../data", help="Dataset path")
    args = parser.parse_args()
    
    print(vars(args))
    
    device = torch.device(args.device)
    
    scaler = MinMaxScaler()
    df = fetch_data(args.dataset, args.data, scaler)    
    dataset = TimeSeriesDataset(df.values, args.seq_length)
    train_size = int(len(dataset) * args.train_eval_split)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = LSTMModel()
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    if args.mode in ["train", "both"]:
        model.train()
        for epoch in range(args.epochs):
            for seq, labels in train_loader:
                seq = seq.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                y_pred = model(seq)
                loss = loss_fn(y_pred, labels)
                loss.backward()
                optimizer.step()
            
            # end of epoch
            # if epoch % 10 == 0:
            print(f"[INFO] Epoch {epoch} / {args.epochs} loss: {loss.item()}")
        
        torch.save(model.state_dict(), os.path.join(args.saved_dir, f'lstm_{args.dataset}.pth'))
    
    if args.mode in ["both", "infer"]:
        if args.weights:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        model.eval()
        test_predictions = []
        test_targets = []
        with torch.no_grad():
            for seq, labels in test_loader:
                seq = seq.to(device)
                labels = labels.to(device)
                y_pred = model(seq)
                test_predictions.extend(y_pred.cpu().numpy())
                test_targets.extend(labels.cpu().numpy())
        
        # 反标准化预测值
        test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
        test_targets = scaler.inverse_transform(np.array(test_targets).reshape(-1, 1))
        
        # 计算预测结果与标签的差距
        mse = np.mean((test_predictions - test_targets) ** 2)
        rmse = np.sqrt(mse)
        
        print(f'Root Mean Squared Error: {rmse}')
        
        
if __name__ == "__main__":
    main()
