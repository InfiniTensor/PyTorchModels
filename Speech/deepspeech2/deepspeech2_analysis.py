#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_ds2_metrics.py — DeepSpeech2 日志解析 + 可选绘图
自动把输出文件名与 <log>.log 关联：
    deepspeech_train.log
      ├─ deepspeech_train_batch_metrics.csv
      ├─ deepspeech_train_epoch_metrics.csv
      ├─ deepspeech_train_batch_loss.png     (可选)
      ├─ deepspeech_train_batch_its.png      (可选)
      ├─ deepspeech_train_epoch_avg_loss.png (可选)
      └─ deepspeech_train_epoch_cer.png      (可选)
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt   # 只有 --plot 时才真正用

# ────────────────── 正则 ──────────────────
BATCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/\d+.*?\[\S+\s+(?P<its>[\d.]+)it/s.*?batch_loss=(?P<loss>[\d.]+)"
)
AVG_RE   = re.compile(
    r"\[Epoch\s+(?P<epoch>\d+)/\d+\]\s+Avg loss:\s+(?P<avg_loss>[\d.]+),\s+Avg it/s:\s+(?P<avg_its>[\d.]+)"
)
CER_RE   = re.compile(
    r"Test epoch:\s+(?P<epoch>\d+).*?loss:\s+(?P<loss>[\d.]+),\s+cer:\s+(?P<cer>[\d.]+)"
)

# ────────────────── 解析日志 ──────────────────
def parse_log(path: Path):
    batch_rows, epoch_rows = [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, 1):
            if (m := BATCH_RE.search(line)):
                batch_rows.append([ln, int(m["epoch"]), float(m["loss"]), float(m["its"])])
                continue
            if (m := AVG_RE.search(line)):
                epoch_rows.append([ln, int(m["epoch"]), float(m["avg_loss"]),
                                   float(m["avg_its"]), "", ""])  # final_loss, cer
                continue
            if (m := CER_RE.search(line)):
                for row in epoch_rows[::-1]:
                    if row[1] == int(m["epoch"]) and row[4] == "":
                        row[4] = float(m["loss"]); row[5] = float(m["cer"])
                        break
    return batch_rows, epoch_rows

# ────────────────── 写 CSV ──────────────────
def write_csv(rows, header, path: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header, *rows])

# ────────────────── 画图 ──────────────────
def line_plot(x, y, xlabel, ylabel, title, out_png):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def plot_all(batch_rows, epoch_rows, stem):
    idx        = list(range(len(batch_rows)))
    batch_loss = [r[2] for r in batch_rows]
    batch_its  = [r[3] for r in batch_rows]
    epochs     = [r[1] for r in epoch_rows]
    avg_loss   = [r[2] for r in epoch_rows]
    cer_pair   = [(r[1], float(r[5])) for r in epoch_rows if r[5] != ""]

    line_plot(idx, batch_loss, "Batch idx", "Loss",
              "Batch Loss", f"{stem}_batch_loss.png")
    line_plot(idx, batch_its,  "Batch idx", "it/s",
              "Batch it/s",  f"{stem}_batch_its.png")
    line_plot(epochs, avg_loss, "Epoch", "Avg loss",
              "Epoch Avg Loss", f"{stem}_epoch_avg_loss.png")
    if cer_pair:
        ep, ce = zip(*cer_pair)
        line_plot(ep, ce, "Epoch", "CER",
                  "Epoch CER", f"{stem}_epoch_cer.png")

# ────────────────── CLI ──────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("log_file", type=Path, help="DeepSpeech2 训练日志 *.log")
    p.add_argument("--plot", action="store_true", help="同时绘制 PNG 图")
    p.add_argument("--batch_csv", type=Path,
                   help="批次指标 CSV（默认自动命名）")
    p.add_argument("--epoch_csv", type=Path,
                   help="Epoch 指标 CSV（默认自动命名）")
    args = p.parse_args()

    stem = args.log_file.stem  # e.g., deepspeech_train
    batch_csv = args.batch_csv or Path(f"{stem}_batch_metrics.csv")
    epoch_csv = args.epoch_csv or Path(f"{stem}_epoch_metrics.csv")

    batch, epoch = parse_log(args.log_file)
    write_csv(batch, ["lineno","epoch","batch_loss","batch_it/s"], batch_csv)
    write_csv(epoch, ["lineno","epoch","avg_loss","avg_it/s","final_loss","cer"],
              epoch_csv)
    print(f"[✓] {batch_csv}")
    print(f"[✓] {epoch_csv}")

    if args.plot:
        plot_all(batch, epoch, stem)
        print(f"[✓] PNG 图以 '{stem}_*.png' 命名保存完成")

if __name__ == "__main__":
    main()

