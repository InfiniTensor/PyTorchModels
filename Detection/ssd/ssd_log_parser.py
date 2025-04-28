import re
import argparse
import pandas as pd

def parse_ssd_log(log_file, output_csv):
    # 初始化数据结构
    data = {
        'epoch': [],
        'batch': [],
        'batch_time': [],
        'data_time': [],
        'loss': []
    }
    
    epoch_pattern = r'Epoch: \[(\d+)\]\[(\d+)/(\d+)\]'
    metrics_pattern = r'Batch Time ([\d.]+) \(([\d.]+)\)\s+Data Time ([\d.]+) \(([\d.]+)\)\s+Loss ([\d.]+) \(([\d.]+)\)'
    
    with open(log_file, 'r') as f:
        for line in f:
            epoch_match = re.search(epoch_pattern, line)
            if not epoch_match:
                continue
                
            # 匹配指标数据
            metrics_match = re.search(metrics_pattern, line)
            if not metrics_match:
                continue
                
            # 提取数据
            epoch = int(epoch_match.group(1))
            batch = int(epoch_match.group(2))
            total_batches = int(epoch_match.group(3))
            
            current_batch_time = float(metrics_match.group(1))
            avg_batch_time = float(metrics_match.group(2))
            current_data_time = float(metrics_match.group(3))
            avg_data_time = float(metrics_match.group(4))
            current_loss = float(metrics_match.group(5))
            avg_loss = float(metrics_match.group(6))
            
            # 存储数据
            data['epoch'].append(epoch)
            data['batch'].append(f"{batch}/{total_batches}")
            data['batch_time'].append(current_batch_time)
            data['data_time'].append(current_data_time)
            data['loss'].append(current_loss)
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"解析完成，结果已保存到 {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SSD训练日志解析工具')
    parser.add_argument('-l', '--log', 
                       default='ssd_train.log',
                       help='输入日志文件路径（默认：ssd_train.log）')
    parser.add_argument('-o', '--output',
                       required=True,
                       help='输出CSV文件路径（如 ssd_metrics.csv）')
    
    args = parser.parse_args()
    parse_ssd_log(args.log, args.output)
