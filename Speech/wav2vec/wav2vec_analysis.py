#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust wav2vec (🤗 Transformers) training‑log parser → CSV (+ optional plots).

Fixes
-----
* Handles `nan`, `inf` (or `-inf`) in metrics without crashing.
* Gracefully skips malformed lines instead of aborting the whole run.

Usage
-----
python wav2vec_analysis.py path/to/wav2vec_train.log [--plot]

Outputs a CSV named `<logname>_metrics.csv` with columns:
    step, epoch, loss, s_per_it

If `--plot` is given, generates two PNGs in the same folder:
    <logname>_loss.png   – loss vs step
    <logname>_speed.png  – seconds/iter vs step
"""

from __future__ import annotations
import re
import argparse
from pathlib import Path
from math import inf, nan
import json
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # plotting optional
    plt = None

# ───────────────────────── regex ─────────────────────────
# Progress‑bar line example: "  1/315 [00:04<23:11,  4.43s/it]"
PB_RE = re.compile(r"""\s*\d+/\d+\s+\[.*?,\s+(?P<spit>[\d.]+)s/it\]""")

# Dict line printed by 🤗 Trainer logger – single quotes + python literals
DICT_RE = re.compile(r"""^\{.*'loss'\s*:\s*[^,]+.*\}$""")

# Fallback regex if literal‑eval fails
FALLBACK_RE = re.compile(
    r"'loss'\s*:\s*(?P<loss>[^,}]+).*'epoch'\s*:\s*(?P<epoch>[^,}]+)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(s: str) -> float:
    """Convert str → float, accepting 'nan', 'inf', '-inf'."""
    try:
        return float(s)
    except ValueError:
        s_low = s.strip().lower()
        if s_low == 'nan':
            return nan
        if s_low == 'inf':
            return inf
        if s_low == '-inf':
            return -inf
        raise


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_log(path: Path) -> pd.DataFrame:
    data: list[dict] = []
    current_spit: float | None = None  # last seconds/iter parsed
    step = 0

    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 1) Progress‑bar speed (available on preceding line)
            m_pb = PB_RE.search(line)
            if m_pb:
                current_spit = float(m_pb.group('spit'))
                continue

            # 2) Metric dict emitted by 🤗 Trainer (next line)
            if DICT_RE.match(line.strip()):
                line_stripped = line.strip()

                # Replace nan/inf with quoted strings to allow JSON parsing
                json_like = (line_stripped
                              .replace("nan", '"nan"')
                              .replace("inf", '"inf"')
                              .replace("-inf", '"-inf"')
                              .replace("'", '"'))
                record = {}
                try:
                    record = json.loads(json_like)
                except json.JSONDecodeError:
                    # Fallback to regex extraction
                    m_fb = FALLBACK_RE.search(line_stripped)
                    if not m_fb:
                        continue  # skip malformed line
                    record['loss'] = m_fb.group('loss')
                    record['epoch'] = m_fb.group('epoch')

                loss = _safe_float(str(record.get('loss', 'nan')))
                epoch = _safe_float(str(record.get('epoch', 'nan')))

                step += 1
                data.append({
                    'step': step,
                    'epoch': epoch,
                    'loss': loss,
                    's_per_it': current_spit
                })
    return pd.DataFrame.from_records(data)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metrics(df: pd.DataFrame, stem: str) -> None:
    if plt is None:
        print("matplotlib not installed – skipping plots.")
        return

    # Loss curve
    plt.figure()
    plt.plot(df['step'], df['loss'], marker='o', linewidth=1)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training loss')
    plt.tight_layout()
    plt.savefig(f"{stem}_loss.png", dpi=150)
    plt.close()

    # Speed curve (skip if all NaNs)
    if df['s_per_it'].notna().any():
        plt.figure()
        plt.plot(df['step'], df['s_per_it'], marker='o', linewidth=1)
        plt.xlabel('Step')
        plt.ylabel('Seconds / iteration')
        plt.title('Training speed')
        plt.tight_layout()
        plt.savefig(f"{stem}_speed.png", dpi=150)
        plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Extract loss + seconds/iter from wav2vec training log.')
    parser.add_argument('log', type=Path, help='training‑log file (.log)')
    parser.add_argument('--plot', action='store_true', help='generate PNG plots')
    args = parser.parse_args()

    df = parse_log(args.log)
    if df.empty:
        print('No metrics found – nothing to do.')
        return

    stem = args.log.with_suffix('').name
    csv_path = args.log.with_suffix('').with_name(f"{stem}_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}   ({len(df)} rows)")

    if args.plot:
        plot_metrics(df, stem)
        print('Plots saved.')


if __name__ == '__main__':
    main()

