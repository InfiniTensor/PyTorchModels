#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dcgan训练日志解析器
用法：
  python parse_logs.py --log train.log --output metrics.csv
  python parse_logs.py -l train.log -o metrics.csv
"""
import re
import argparse
import pandas as pd

def parse_log_file(log_path):
    """解析日志文件提取关键指标"""
    pattern = re.compile(
        r'\[(\d+)/(\d+)\]\[(\d+)/(\d+)\]\s+'
        r'Loss_D:\s+([\d.]+)\s+'
        r'Loss_G:\s+([\d.]+)\s+'
        r'D\(x\):\s+([\d.]+)\s+'
        r'D\(G\(z\)\):\s+([\d.]+)\s+/\s+([\d.]+)'
    )
    
    data = []
    
    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                total_epochs = int(match.group(2))
                iteration = int(match.group(3))
                total_iterations = int(match.group(4))
                loss_d = float(match.group(5))
                loss_g = float(match.group(6))
                d_x = float(match.group(7))
                d_g_z = float(match.group(8))
                d_g_z_ratio = float(match.group(9))
                
                data.append({
                    'epoch': epoch,
                    'total_epochs': total_epochs,
                    'iteration': iteration,
                    'total_iterations': total_iterations,
                    'loss_d': loss_d,
                    'loss_g': loss_g,
                    'd_x': d_x,
                    'd_g_z': d_g_z,
                    'd_g_z_ratio': d_g_z_ratio
                })
    
    return pd.DataFrame(data)

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Parse DCGAN training logs and extract metrics.')
    parser.add_argument('--log', '-l', required=True, help='输入日志文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出CSV文件路径（如 metrics.csv）')
    
    args = parser.parse_args()
    
    print(f"Parsing log file: {args.log}")
    df = parse_log_file(args.log)
    
    print(f"Saving metrics to: {args.output}")
    df.to_csv(args.output, index=False)
    
    print(f"Done! Extracted {len(df)} records.")

if __name__ == '__main__':
    main()
