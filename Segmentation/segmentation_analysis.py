#!/usr/bin/env python3
import re
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def parse_logs(log_paths, plot=False):
    # 正则表达式
    batch_re = re.compile(r'(?P<speed>[\d\.]+)it/s(?:, batch_loss=(?P<loss>[\d\.]+))?')
    epoch_info_re = re.compile(r'\[INFO\] Train \[(?P<epoch>\d+)/\d+\] Loss: (?P<loss>[\d\.]+)')
    epoch_sum_re = re.compile(r'Epoch\s*(?P<epoch>\d+)/\d+\s*\|\s*Avg Loss:\s*(?P<loss>[\d\.]+)\s*\|\s*Avg it/s:\s*(?P<speed>[\d\.]+)')

    all_data = {}

    for path in log_paths:
        model_name = os.path.splitext(os.path.basename(path))[0]
        batch_records = []
        epoch_records = []
        current_epoch = None
        batch_speeds = []

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                # epoch 切换
                m_epoch_start = re.search(r'(?:Training epoch|Epoch)\s*(\d+)/\d+', line)
                if m_epoch_start:
                    current_epoch = int(m_epoch_start.group(1))
                    batch_speeds.clear()

                # batch-level
                m_batch = batch_re.search(line)
                if m_batch and current_epoch is not None:
                    speed = float(m_batch.group('speed'))
                    loss = m_batch.group('loss')
                    loss = float(loss) if loss is not None else pd.NA
                    batch_records.append({
                        'epoch': current_epoch,
                        'global_batch_idx': len(batch_records),
                        'loss': loss,
                        'it/s': speed
                    })
                    batch_speeds.append(speed)
                    continue

                # epoch summary 优先匹配 Avg Loss 行
                m_sum = epoch_sum_re.search(line)
                if m_sum:
                    epoch = int(m_sum.group('epoch'))
                    loss = float(m_sum.group('loss'))
                    speed = float(m_sum.group('speed'))
                    epoch_records.append({'epoch': epoch, 'loss': loss, 'it/s': speed})
                    continue

                # 兼容 Deeplab 的 INFO 行
                m_info = epoch_info_re.search(line)
                if m_info:
                    epoch = int(m_info.group('epoch'))
                    loss = float(m_info.group('loss'))
                    speed = (sum(batch_speeds)/len(batch_speeds)) if batch_speeds else pd.NA
                    epoch_records.append({'epoch': epoch, 'loss': loss, 'it/s': speed})

        df_batch = pd.DataFrame(batch_records)
        df_epoch = pd.DataFrame(epoch_records)

        # 写入 CSV
        df_batch.to_csv(f"{model_name}_batch_metrics.csv", index=False)
        df_epoch.to_csv(f"{model_name}_epoch_metrics.csv", index=False)

        all_data[model_name] = (df_batch, df_epoch)

    if plot:
        for model, (df_batch, df_epoch) in all_data.items():
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Metrics for {model}", fontsize=16)

            # Batch Loss
            if not df_batch['loss'].dropna().empty:
                df_batch['loss'].plot(ax=axs[0,0], title="Batch Loss")
                axs[0,0].set_xlabel("Global Batch Index")
                axs[0,0].set_ylabel("Loss")
            else:
                axs[0,0].text(0.5, 0.5, "No batch loss data", ha='center', va='center')
                axs[0,0].set_title("Batch Loss"); axs[0,0].axis('off')

            # Batch it/s
            if not df_batch['it/s'].empty:
                df_batch['it/s'].plot(ax=axs[0,1], title="Batch it/s")
                axs[0,1].set_xlabel("Global Batch Index")
                axs[0,1].set_ylabel("it/s")
            else:
                axs[0,1].text(0.5, 0.5, "No batch speed data", ha='center', va='center')
                axs[0,1].set_title("Batch it/s"); axs[0,1].axis('off')

            # Epoch Loss
            if not df_epoch['loss'].empty:
                df_epoch.plot(x='epoch', y='loss', marker='o', ax=axs[1,0], title="Epoch Loss")
                axs[1,0].set_xlabel("Epoch"); axs[1,0].set_ylabel("Loss")
            else:
                axs[1,0].text(0.5, 0.5, "No epoch loss data", ha='center', va='center')
                axs[1,0].set_title("Epoch Loss"); axs[1,0].axis('off')

            # Epoch it/s
            if not df_epoch['it/s'].empty:
                df_epoch.plot(x='epoch', y='it/s', marker='o', ax=axs[1,1], title="Epoch it/s")
                axs[1,1].set_xlabel("Epoch"); axs[1,1].set_ylabel("it/s")
            else:
                axs[1,1].text(0.5, 0.5, "No epoch speed data", ha='center', va='center')
                axs[1,1].set_title("Epoch it/s"); axs[1,1].axis('off')

            plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Parse training logs and extract metrics")
    parser.add_argument("logs", nargs="+", help="路径到训练日志文件")
    parser.add_argument("--plot", action="store_true", help="是否绘制图表")
    args = parser.parse_args()

    parse_logs(args.logs, plot=args.plot)

if __name__ == "__main__":
    main()

