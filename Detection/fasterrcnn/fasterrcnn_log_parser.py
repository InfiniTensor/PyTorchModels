#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FasterRCNN训练日志解析器
用法：
  python parse_logs.py --log train.log --output metrics.csv
  python parse_logs.py -l train.log -o metrics.csv
"""

import re
import pandas as pd
from pathlib import Path
import argparse

def parse_log(log_path: str) -> pd.DataFrame:
    """解析日志文件提取关键指标"""
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    pattern = re.compile(
        r'lr:(?P<lr>[\d\.e-]+).*?'
        r'map:(?P<val_mAP>[\d\.]+).*?'
        r'loss:\{(?:.*?total_loss\':\s*(?P<total_loss>[\d\.]+)).*?'
        r'(\d+)it\s+\[[\d:]+\,\s*([\d\.]+)it/s\]',
        re.DOTALL
    )

    data = []
    for match in pattern.finditer(content):
        data.append({
            'epoch': len(data) + 1,
            'learning_rate': float(match.group('lr')),
            'val_mAP': float(match.group('val_mAP')),
            'total_loss': float(match.group('total_loss')),
            'throughput(it/s)': float(match.group(5))
        })

    return pd.DataFrame(data)

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(
        description='解析FasterRCNN训练日志并提取关键指标',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-l', '--log', 
                       default='train.log',
                       help='输入日志文件路径')
    parser.add_argument('-o', '--output', 
                       required=True,
                       help='输出CSV文件路径（如 metrics.csv）')
    
    args = parser.parse_args()

    # 检查输入文件是否存在
    if not Path(args.log).exists():
        raise FileNotFoundError(f"日志文件不存在: {args.log}")

    # 解析并保存结果
    df = parse_log(args.log)
    df.to_csv(args.output, index=False, float_format='%.4f')
    
    print(f"✅ 解析完成！结果已保存到 {args.output}")
    print(f"共处理 {len(df)} 个epoch，最终mAP: {df['val_mAP'].iloc[-1]:.4f}")

if __name__ == "__main__":
    main()
