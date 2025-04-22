#!/usr/bin/env python3
"""
DCGAN训练日志解析器
用法：
  python parse_logs.py --log train.log --output metrics.csv
  python parse_logs.py -l train.log -o metrics.csv
"""
import argparse
import re
import pandas as pd

def parse_log_file(log_path: str) -> pd.DataFrame:
    """解析DCGAN训练日志"""
    pattern = re.compile(
        r'\[(\d+)/(\d+)\]\[(\d+)/(\d+)\].*?'
        r'Loss_D: ([\d.]+).*?'
        r'Loss_G: ([\d.]+).*?'
        r'D\(x\): ([\d.]+).*?'
        r'D\(G\(z\)\): ([\d.]+)'
    )
    
    data = []
    with open(log_path, 'r') as f:
        for line in f:
            if match := pattern.search(line):
                data.append({
                    'epoch': int(match.group(1)),
                    'total_epochs': int(match.group(2)),
                    'iteration': int(match.group(3)),
                    'loss_d': float(match.group(5)),
                    'loss_g': float(match.group(6)),
                    'd_x': float(match.group(7)),
                    'd_g_z': float(match.group(8))
                })
    
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(
        description='DCGAN训练日志解析工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-l', '--log', required=True, help='输入日志文件路径')
    parser.add_argument('-o', '--output', required=True, help='输出CSV文件路径')
    
    args = parser.parse_args()
    
    df = parse_log_file(args.log)
    df.to_csv(args.output, index=False)
    print(f"成功保存 {len(df)} 条记录到 {args.output}")

if __name__ == '__main__':
    main()
