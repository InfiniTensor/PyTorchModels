#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse distributed NeuMF training log, extract per‑epoch throughput / loss /
time, and (optionally) plot them.

Example
-------
python recommend_analysis.py recommend_train.log \
       --out_csv recommend_analysis.csv --plot

If your epoch is very sjort, adjust --time_reset_sec (default 600 s).
"""

import re, csv, argparse
from datetime import timedelta

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    pd = plt = None

PAT = re.compile(
    r'(?P<cur>\d+)/(?P<total>\d+)'
    r'.*?\[(?P<elapsed>[0-9:]+)'
    r'.*?(?P<speed>[0-9.]+)\s*it/s'
    r'.*?loss=(?P<loss>[0-9.]+)'
)

def hms(txt):
    p = list(map(int, txt.split(':')))
    if len(p) == 2: p = [0] + p
    return timedelta(hours=p[0], minutes=p[1], seconds=p[2]).total_seconds()

def parse(log_path, reset_sec):
    epochs, ep = [], {'it_s': [], 'loss': [], 'time': 0}
    last_elapsed = 0

    with open(log_path, encoding='utf-8', errors='ignore') as f:
        for ln in f:
            m = PAT.search(ln);  # 跳过无关行
            if not m: continue

            el = hms(m['elapsed'])
            # —— 新 epoch 判定：耗时突然掉到很小 —— #
            if el + reset_sec < last_elapsed and ep['it_s']:
                epochs.append(ep)
                ep = {'it_s': [], 'loss': [], 'time': 0}

            last_elapsed = el
        
            ep['it_s'].append(float(m['speed']))
            ep['loss'].append(float(m['loss']))
            ep['time'] = max(ep['time'], el)

    if ep['it_s']:  # push 最后一个
        epochs.append(ep)

    return [{
        'epoch': i + 1,
        'avg_it_s': sum(e['it_s']) / len(e['it_s']),
        'avg_loss': sum(e['loss']) / len(e['loss']),
        'epoch_time_sec': e['time']
    } for i, e in enumerate(epochs)]

def to_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader(); w.writerows(rows)

def plot(csv_path):
    if pd is None or plt is None:
        raise RuntimeError("matplotlib / pandas 未安装")
    df = pd.read_csv(csv_path)

    plt.figure()
    plt.plot(df['epoch'], df['avg_it_s'], marker='o')
    plt.title('Throughput (it/s)'); plt.xlabel('Epoch'); plt.ylabel('avg_it_s')
    plt.tight_layout()

    plt.figure()
    plt.plot(df['epoch'], df['avg_loss'], marker='o')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('avg_loss')
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    # python recommend_analysis.py recommend_train.log --plot
    ap = argparse.ArgumentParser()
    ap.add_argument('log_path')
    ap.add_argument('--out_csv', default='recommend_analysis.csv')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--time_reset_sec', type=int, default=600,
                    help='elapsed‑time drop threshold to mark new epoch')
    args = ap.parse_args()

    data = parse(args.log_path, args.time_reset_sec)
    if not data:
        raise SystemExit("No progress lines matched pattern.")
    to_csv(data, args.out_csv)
    print(f"✔  Parsed {len(data)} epoch(s) → {args.out_csv}")

    if args.plot:
        plot(args.out_csv)

