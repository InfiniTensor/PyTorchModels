#!/usr/bin/env python3
"""
Mask R-CNN训练日志解析器
用法：
  python parse_logs.py --log train.log --output metrics.csv
  python parse_logs.py -l train.log -o metrics.csv
"""
import argparse
import re
import csv
from typing import List, Dict

def parse_log_file(log_file: str) -> List[Dict]:
    """解析Mask R-CNN训练日志"""
    results = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if train_match := re.search(r'iter: (\d+)\s+loss: ([\d.]+) \(([\d.]+)\)', line):
                results.append({
                    'type': 'train',
                    'iter': int(train_match.group(1)),
                    'loss': float(train_match.group(2)),
                    'loss_avg': float(train_match.group(3))
                })
            elif map_match := re.search(r'Average Precision.*= ([\d.]+)', line):
                results.append({
                    'type': 'eval',
                    'mAP': float(map_match.group(1))
                })
            elif time_match := re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line):
                if results:
                    results[-1]['timestamp'] = time_match.group(1)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Mask R-CNN训练日志解析工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-l', '--log', required=True, help='输入日志文件路径')
    parser.add_argument('-o', '--output', required=True, help='输出CSV文件路径')
    
    args = parser.parse_args()
    
    metrics = parse_log_file(args.log)
    if not metrics:
        print("错误: 未找到任何评估指标")
        return
    
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'type', 'iter', 'loss', 'loss_avg', 'mAP'])
        writer.writeheader()
        writer.writerows(metrics)
    
    print(f"成功保存 {len(metrics)} 条记录到 {args.output}")

if __name__ == '__main__':
    main()
