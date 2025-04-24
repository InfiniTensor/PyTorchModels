#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse training log(s) → CSVs + plots – **per‑log filenames**

2025‑04‑23 UPDATE #3 (bug‑fix)
=============================
* **plot_epoch() / plot_batches() 重新内嵌**，避免 NameError。
* 若 Accuracy 全缺省，则 epoch 图内仅绘 loss/val‑loss/it/s。
* 批次图保持单张合并；若只传单日志且 --plot‑batch，用该日志批次曲线。 
"""

import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

# ───────────────────────── regex ─────────────────────────
BATCH_RE = re.compile(r"Epoch\s+(?P<epoch>\d+).*?(?P<its>\d+\.\d+)it/s.*?Batch_loss=(?P<loss>\d+\.\d+)")
EPOCH_RE = re.compile(r"Epoch\s+(?P<epoch>\d+)(?:/\d+)?\s+\|\s+Avg Loss:\s+(?P<avg_loss>\d+\.\d+)\s+\|\s+Avg it/s:\s+(?P<avg_its>\d+\.\d+)")
TEST_RE  = re.compile(r"Test set:\s+Average loss:\s+(?P<test_loss>\d+\.\d+),\s+Accuracy:\s+(?P<correct>\d+)/(?:\d+).*?\((?P<acc>\d+(?:\.\d+)?)%\)")
VAL_RE   = re.compile(
    r"Val:\s*epoch\s*=\s*(?P<epoch>\d+)(?:/\d+)?[^:]*?:.*?val_loss\s*=\s*(?P<val_loss>\d+\.\d+)",
    re.IGNORECASE,
)

# ──────────────────────── parsing ────────────────────────

def parse_file(path: Path):
    batches, epoch_summary, test_summary, val_summary = [], {}, {}, {}
    step_counter = defaultdict(int)
    ep_context = None
    with path.open(encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if (m := BATCH_RE.search(ln)):
                ep = int(m["epoch"])
                step_counter[ep] += 1
                batches.append((ep, step_counter[ep], float(m["loss"]), float(m["its"])) )
                ep_context = ep
                continue
            if (m := EPOCH_RE.search(ln)):
                ep = int(m["epoch"])
                epoch_summary[ep] = (float(m["avg_loss"]), float(m["avg_its"]))
                ep_context = ep
                continue
            if (m := TEST_RE.search(ln)) and ep_context is not None:
                test_summary[ep_context] = (float(m["test_loss"]), float(m["acc"]))
                continue
            if (m := VAL_RE.search(ln)):
                ep = int(m["epoch"])
                val_summary[ep] = float(m["val_loss"])
    return batches, epoch_summary, test_summary, val_summary

# ─────────────────────── CSV writers ─────────────────────

def write_batches_csv(prefix, rows):
    out = Path(f"{prefix}_batches.csv")
    with out.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["file", "epoch", "step", "loss", "it/s"])
        for ep, step, loss, its in rows:
            w.writerow([prefix, ep, step, loss, its])
    print(f"[OK] {out}")
    return out


def write_summary_csv(prefix, data):
    epoch_summ, test_summ, val_summ = data
    epochs = sorted(set(epoch_summ) | set(val_summ) | set(test_summ))
    out = Path(f"{prefix}_summary.csv")
    with out.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["epoch", "avg_loss", "val_loss", "avg_it/s", "test_loss", "accuracy"])
        for ep in epochs:
            al, it = epoch_summ.get(ep, (None, None))
            vl      = val_summ.get(ep)         
            tl, ac  = test_summ.get(ep, (None, None))
            w.writerow([ep, al, vl, it, tl, ac])
    print(f"[OK] {out}")
    return out

# ──────────────────────── plotting ───────────────────────

def plot_epoch(summary_csv: Path):
    """Draw epoch‑level curves.
    Figure 1 (<stem>_loss_it.png): Train Loss, Val Loss, Avg it/s (no Accuracy).
    Figure 2 (<stem>_accuracy.png): Accuracy (%) – only if column present & non‑NaN.
    """
    df = pd.read_csv(summary_csv)
    stem = summary_csv.stem.replace("_summary", "")

    # ── Fig 1: loss / it/s ──────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(8,4))
    # loss curves
    ax1.plot(df["epoch"], df.get(f"{stem}_avg_loss", df["avg_loss"]), marker="o", label="Train Loss")
    val_col = f"{stem}_val_loss" if f"{stem}_val_loss" in df.columns else "val_loss"
    if val_col in df and not df[val_col].isna().all():
        ax1.plot(df["epoch"], df[val_col], linestyle="--", marker="x", label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")

    # it/s on right axis
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df.get(f"{stem}_avg_it/s", df["avg_it/s"]), linestyle=":", marker="s", label="Avg it/s")
    ax2.set_ylabel("it/s")

    # legend merge (only two axes)
    lines, labels = [], []
    for ax in [ax1, ax2]:
        lns, lbs = ax.get_legend_handles_labels(); lines+=lns; labels+=lbs
    ax1.legend(lines, labels, loc="best")

    plt.title(f"Metrics – {stem}")
    plt.tight_layout()
    out = summary_csv.with_name(f"{stem}_loss_it.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[OK] {out}")

    # ── Fig 2: accuracy (if available) ──────────────────────────────
    acc_col = f"{stem}_accuracy" if f"{stem}_accuracy" in df.columns else "accuracy"
    if acc_col in df and not df[acc_col].isna().all():
        plt.figure(figsize=(6,3))
        plt.plot(df["epoch"], df[acc_col], marker="^")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
        plt.title(f"Accuracy – {stem}")
        plt.tight_layout()
        out2 = summary_csv.with_name(f"{stem}_accuracy.png")
        plt.savefig(out2, dpi=300)
        plt.close()
        print(f"[OK] {out2}")


def plot_batches(batch_dict):
    """Draw all batch loss & it/s curves in one figure.
    If只处理单日志 → `<prefix>_batch_curves.png`；
    多日志 → `combined_batch_curves.png` (避免过长文件名)。"""
    plt.figure(figsize=(9, 4))
    ax1 = plt.gca()

    for p, rows in batch_dict.items():
        rows_sorted = sorted(rows, key=lambda r: (r[0], r[1]))
        steps = list(range(1, len(rows_sorted) + 1))
        loss  = [r[2] for r in rows_sorted]
        its   = [r[3] for r in rows_sorted]
        ax1.plot(steps, loss, alpha=0.4, label=f"{p} Loss")
        ax1.set_xlabel("Global Step")
        ax1.set_ylabel("Batch Loss")

    ax2 = ax1.twinx()
    for p, rows in batch_dict.items():
        rows_sorted = sorted(rows, key=lambda r: (r[0], r[1]))
        steps = list(range(1, len(rows_sorted) + 1))
        its   = [r[3] for r in rows_sorted]
        ax2.plot(steps, its, linestyle=":", label=f"{p} it/s")
    ax2.set_ylabel("it/s")

    lines, labels = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + l2, labels + lb2, loc="best")

    plt.title("Batch‑level Curves")
    plt.tight_layout()

    # choose filename
    if len(batch_dict) == 1:
        prefix = next(iter(batch_dict))
        out = Path(f"{prefix}_batch_curves.png")
    else:
        out = Path("combined_batch_curves.png")

    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[OK] {out}") 

# ─────────────────────────── main ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="log file")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-batch", action="store_true")
    args = ap.parse_args()

    combined_batches = {}
    log = args.log
    p = Path(log)
    prefix = p.stem
    batches, ep_s, t_s, v_s = parse_file(p)
    sum_csv = write_summary_csv(prefix, (ep_s, t_s, v_s))
    _ = write_batches_csv(prefix, batches)
    if args.plot:
        plot_epoch(sum_csv)
    combined_batches[prefix] = batches

    if args.plot_batch and combined_batches:
        plot_batches(combined_batches)

if __name__ == "__main__":
    main()

# python ./time_series_analysis.py ./lstm/lstm_train.log --plot --plot-batch
