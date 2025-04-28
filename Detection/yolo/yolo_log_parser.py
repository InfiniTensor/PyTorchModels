#!/usr/bin/env python3
import re
import pandas as pd
from pathlib import Path
import argparse

def parse_yolov5_log(log_path: str) -> pd.DataFrame:
    """
    解析YOLOv5训练日志，提取每个epoch的关键指标
    返回包含以下字段的DataFrame:
    - epoch
    - gpu_mem (GB)
    - box_loss
    - obj_loss
    - cls_loss
    - instances (平均每batch的实例数)
    - lr (学习率，需从日志其他位置提取)
    """
    # 初始化数据存储
    data = []
    current_epoch = None
    epoch_data = {
        'epoch': 0,
        'gpu_mem': 0,
        'box_loss': 0,
        'obj_loss': 0,
        'cls_loss': 0,
        'instances': 0,
        'batch_count': 0
    }

    # 正则表达式模式 - 更新以匹配实际日志格式
    epoch_pattern = re.compile(r'^\s*(\d+)/\d+\s+(\d+\.\d+)G\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s+640')
    lr_pattern = re.compile(r'lr:\s*([\d\.e+-]+)')

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 匹配epoch数据行 (如 "0/24    6.4G     0.1095    0.06172      0.104        169        640")
            if epoch_match := epoch_pattern.search(line):
                epoch = int(epoch_match.group(1))
                if epoch != current_epoch:
                    if current_epoch is not None and epoch_data['batch_count'] > 0:
                        # 计算当前epoch的平均值并保存
                        data.append({
                            'epoch': current_epoch,
                            'gpu_mem': epoch_data['gpu_mem'] / epoch_data['batch_count'],
                            'box_loss': epoch_data['box_loss'] / epoch_data['batch_count'],
                            'obj_loss': epoch_data['obj_loss'] / epoch_data['batch_count'],
                            'cls_loss': epoch_data['cls_loss'] / epoch_data['batch_count'],
                            'instances': epoch_data['instances'] / epoch_data['batch_count']
                        })
                    # 重置为新epoch
                    current_epoch = epoch
                    epoch_data = {
                        'epoch': epoch,
                        'gpu_mem': 0,
                        'box_loss': 0,
                        'obj_loss': 0,
                        'cls_loss': 0,
                        'instances': 0,
                        'batch_count': 0
                    }

                # 累加当前batch数据
                epoch_data['gpu_mem'] += float(epoch_match.group(2))
                epoch_data['box_loss'] += float(epoch_match.group(3))
                epoch_data['obj_loss'] += float(epoch_match.group(4))
                epoch_data['cls_loss'] += float(epoch_match.group(5))
                epoch_data['instances'] += int(epoch_match.group(6))
                epoch_data['batch_count'] += 1

            # 匹配学习率
            elif lr_match := lr_pattern.search(line):
                current_lr = float(lr_match.group(1))
                if data and 'lr' not in data[-1]:
                    data[-1]['lr'] = current_lr

    # 处理最后一个epoch
    if current_epoch is not None and epoch_data['batch_count'] > 0:
        data.append({
            'epoch': current_epoch,
            'gpu_mem': epoch_data['gpu_mem'] / epoch_data['batch_count'],
            'box_loss': epoch_data['box_loss'] / epoch_data['batch_count'],
            'obj_loss': epoch_data['obj_loss'] / epoch_data['batch_count'],
            'cls_loss': epoch_data['cls_loss'] / epoch_data['batch_count'],
            'instances': epoch_data['instances'] / epoch_data['batch_count']
        })

    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description='YOLOv5训练日志解析工具')
    parser.add_argument('-l', '--log', 
                       default='train.log',
                       help='输入日志文件路径（默认：train.log）')
    parser.add_argument('-o', '--output',
                       required=True,
                       help='输出CSV文件路径（如 yolov5_metrics.csv）')
    
    args = parser.parse_args()

    if not Path(args.log).exists():
        raise FileNotFoundError(f"日志文件不存在: {args.log}")

    df = parse_yolov5_log(args.log)
    df.to_csv(args.output, index=False, float_format='%.6f')
    print(f"✅ 解析完成！结果已保存到 {args.output}")
    print(f"共处理 {len(df)} 个epoch")

if __name__ == "__main__":
    main()
