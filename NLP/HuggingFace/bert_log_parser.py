#!/usr/bin/env python3
"""
BERT训练日志解析器
用法：
  python parse_logs.py --log train.log --output metrics.csv
  python parse_logs.py -l train.log -o metrics.csv
"""
import argparse
import re
import json
import csv
from typing import List, Dict

def extract_metrics(line: str) -> Dict:
    """从日志行中提取JSON格式的评估指标"""
    json_str = line[line.find('{'):line.rfind('}')+1]
    try:
        metrics = json.loads(json_str.replace("'", '"'))
        return {k.replace('eval_', ''): v for k, v in metrics.items()}
    except json.JSONDecodeError:
        return None

def parse_log_file(log_file: str) -> List[Dict]:
    """解析BERT训练日志文件"""
    results = []
    current_epoch = None
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            if epoch_match := re.search(r'Epoch (\d+)', line, re.IGNORECASE):
                current_epoch = epoch_match.group(1)
                
            if ('exact' in line and 'f1' in line) and (metrics := extract_metrics(line)):
                metrics['epoch'] = current_epoch
                results.append(metrics)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='BERT训练日志解析工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-l', '--log', required=True, help='输入日志文件路径')
    parser.add_argument('-o', '--output', required=True, help='输出CSV文件路径')
    
    args = parser.parse_args()
    
    metrics = parse_log_file(args.log)
    if not metrics:
        print("错误: 未找到任何评估指标")
        return
    
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    
    print(f"成功保存 {len(metrics)} 条记录到 {args.output}")

if __name__ == '__main__':
    main()
