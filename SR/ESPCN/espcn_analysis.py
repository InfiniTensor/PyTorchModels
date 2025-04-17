#!/usr/bin/env python3
"""ESPCN Performance Extractor — Epoch‑level Summary

* 提取训练/验证迭代速率 (it/s)、Loss、PSNR
* **按 Epoch 汇总**：每个 Epoch 输出一次平均 it/s（train / val 分开），以及该
  Epoch 末尾的 Loss、PSNR
* 可视化 Epoch 级曲线；支持 `--no-plot`
* 保存单一 CSV（epoch_summary.csv）便于横向比较

Usage
-----
    python espcn_analysis.py espcn_train.log  --plot      # 带图表
    python espcn_analysis.py espcn_train.log              # 仅生成 CSV
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
EPOCH_RE = re.compile(r"Epoch\s+(?P<epoch>\d+)/(?:\d+)", flags=re.IGNORECASE)
TRAIN_RE = re.compile(r"Training:.*?\s(?P<cur>\d+)/(?:\d+).*?,\s*(?P<speed>[\d.]+)it/s\]")
VAL_RE   = re.compile(r"Validating:.*?\s(?P<cur>\d+)/(?:\d+).*?,\s*(?P<speed>[\d.]+)it/s\]")
METRIC_RE = re.compile(r"\[(?P<split>Train|Val)]\s+Loss:\s*(?P<loss>[\d.]+),\s*PSNR:\s*(?P<psnr>[\d.]+)", flags=re.IGNORECASE)

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_log(file_path: Path) -> pd.DataFrame:
    """Return tidy DF with columns: epoch, phase, speed, loss, psnr"""

    rows: List[Dict[str, Any]] = []
    current_epoch = 0

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if m := EPOCH_RE.search(line):
                current_epoch = int(m["epoch"])
                continue
            if m := TRAIN_RE.search(line):
                rows.append({
                    "epoch": current_epoch,
                    "phase": "train_iter",
                    "speed": float(m["speed"]),
                })
                continue
            if m := VAL_RE.search(line):
                rows.append({
                    "epoch": current_epoch,
                    "phase": "val_iter",
                    "speed": float(m["speed"]),
                })
                continue
            if m := METRIC_RE.search(line):
                rows.append({
                    "epoch": current_epoch,
                    "phase": f"{m['split'].lower()}_metric",
                    "loss": float(m["loss"]),
                    "psnr": float(m["psnr"]),
                })
    return pd.DataFrame(rows)


def make_epoch_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per‑epoch averages / last metrics."""
    epochs = sorted(df["epoch"].unique())
    records: List[Dict[str, Any]] = []
    for ep in epochs:
        rec: Dict[str, Any] = {"epoch": ep}
        # Avg it/s
        for phase, key in [("train_iter", "train_it_s"), ("val_iter", "val_it_s")]:
            sub = df[(df["epoch"] == ep) & (df["phase"] == phase)]
            if not sub.empty:
                rec[key] = sub["speed"].mean()
        # Last loss / PSNR of epoch
        for phase, prefix in [("train_metric", "train"), ("val_metric", "val")]:
            sub = df[(df["epoch"] == ep) & (df["phase"] == phase)]
            if not sub.empty:
                rec[f"{prefix}_loss"] = sub["loss"].iloc[-1]
                rec[f"{prefix}_psnr"] = sub["psnr"].iloc[-1]
        records.append(rec)
    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_epoch_curves(df_sum: pd.DataFrame) -> None:
    fig, (ax_speed, ax_quality) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Speed
    ax_speed.plot(df_sum["epoch"], df_sum["train_it_s"], label="train it/s", marker="o")
    if "val_it_s" in df_sum:
        ax_speed.plot(df_sum["epoch"], df_sum["val_it_s"], label="val it/s", marker="s")
    ax_speed.set_ylabel("Avg it/s")
    ax_speed.grid(True, alpha=0.3)
    ax_speed.legend()

    # Quality
    ax_quality.plot(df_sum["epoch"], df_sum["train_loss"], label="train loss", marker="o", color="tab:red")
    if "val_loss" in df_sum:
        ax_quality.plot(df_sum["epoch"], df_sum["val_loss"], label="val loss", marker="s", color="tab:orange")
    ax_psnr = ax_quality.twinx()
    ax_psnr.plot(df_sum["epoch"], df_sum["train_psnr"], label="train PSNR", marker="o", ls="--", color="tab:green")
    if "val_psnr" in df_sum:
        ax_psnr.plot(df_sum["epoch"], df_sum["val_psnr"], label="val PSNR", marker="s", ls="--", color="tab:olive")

    ax_quality.set_ylabel("Loss", color="tab:red")
    ax_psnr.set_ylabel("PSNR (dB)", color="tab:green")
    ax_quality.set_xlabel("Epoch")
    ax_quality.grid(True, alpha=0.3)

    # Combine legends
    lines, labels = [], []
    for ax in [ax_speed, ax_quality, ax_psnr]:
        ln, lb = ax.get_legend_handles_labels()
        lines += ln
        labels += lb
    fig.legend(lines, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("ESPCN Training Performance per Epoch")
    plt.show()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Extract ESPCN epoch‑level metrics from log")
    p.add_argument("log_file", type=Path, help="Path to espcn_train.log")
    p.add_argument("--output_csv", type=Path, default=Path("espcn_analysis.csv"), help="CSV summary output")
    p.add_argument("--plot", dest="plot", action="store_true", help="Enable charts")
    args = p.parse_args()

    raw_df = parse_log(args.log_file)
    if raw_df.empty:
        sys.exit("[ERROR] No metrics parsed – aborting.")

    summary_df = make_epoch_summary(raw_df)
    summary_df.to_csv(args.output_csv, index=False)
    print(f"[INFO] Epoch summary saved to {args.output_csv}")
    print(summary_df)

    if args.plot:
        plot_epoch_curves(summary_df)


if __name__ == "__main__":
    main()

