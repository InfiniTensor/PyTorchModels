#!/usr/bin/env python3
"""
DQN训练日志解析器
用法：
  python parse_logs.py --log train.log --output metrics.csv
  python parse_logs.py -l train.log -o metrics.csv
"""
import re
import argparse
import pandas as pd
from pathlib import Path

def parse_log_file(log_path: str) -> pd.DataFrame:
    """解析DQN训练日志，提取关键指标"""
    pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Episode (\d+), Loss: ([\d\.]+)'
    )
    
    data = []
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if match := pattern.search(line):
                data.append({
                    'timestamp': match.group(1),
                    'episode': int(match.group(2)),
                    'loss': float(match.group(3))
                })
    
    df = pd.DataFrame(data)
    df['loss_smooth'] = df['loss'].rolling(window=5, min_periods=1).mean()
    return df

def main():
    parser = argparse.ArgumentParser(
        description='DQN训练日志解析工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-l', '--log', 
                       required=True,
                       help='输入日志文件路径')
    parser.add_argument('-o', '--output',
                       required=True,
                       help='输出CSV文件路径')
    
    args = parser.parse_args()

    if not Path(args.log).exists():
        raise FileNotFoundError(f"日志文件不存在: {args.log}")

    df = parse_log_file(args.log)    
    df.to_csv(args.output, index=False, float_format='%.6f')
    print(f"成功保存 {len(df)} 条训练记录到 {args.output}")

if __name__ == "__main__":
    main()
