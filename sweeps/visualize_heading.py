"""
Visualize heading profiles per path to understand discrimination potential.

Produces results/heading_analysis.png with:
  Row 1: Mean heading histogram per path (polar plots, shows "which compass
         directions does this path spend time pointing toward")
  Row 2: Heading time series (5 example recordings per path, overlaid;
         normalized time so paths of different durations align)
  Row 3: Heading change-rate (|dheading/dt|) per path — where do turns happen?
"""

import os
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TRAIN = os.path.join(ROOT, 'data', 'train')
RESULTS    = os.path.join(ROOT, 'results')

PATH_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
PATH_LABELS = {
    0: 'P0: Central→ETH, no stairs',
    1: 'P1: +Clausiusstr, stairs early',
    2: 'P2: Walchebrücke, stairs mid',
    3: 'P3: reverse of P2 (downhill)',
    4: 'P4: downhill, stairs mid',
}


def heading_series(raw):
    """Return 1-D heading [deg] sampled ~uniformly over recording duration."""
    pmx = np.array(raw['data']['phone_mx']['values'], dtype=float)
    pmy = np.array(raw['data']['phone_my']['values'], dtype=float)
    n   = min(len(pmx), len(pmy))
    if n < 10:
        return None
    return np.degrees(np.arctan2(pmy[:n], pmx[:n])) % 360


def circ_diff(a, b):
    """Signed smallest angular difference (a - b) in [-180, 180]."""
    return (a - b + 180) % 360 - 180


def load_samples():
    """Load all recordings, indexed by path_idx."""
    by_path = {k: [] for k in range(5)}
    for fname in sorted(os.listdir(DATA_TRAIN)):
        if not fname.endswith('.pkl'):
            continue
        with open(os.path.join(DATA_TRAIN, fname), 'rb') as f:
            raw = pickle.load(f)
        h = heading_series(raw)
        if h is None:
            continue
        p = int(raw['labels']['path_idx'])
        by_path[p].append({'filename': fname, 'heading': h})
    return by_path


def resample_to(x, n_target):
    """Linear-interpolation resample 1-D array to n_target points."""
    idx = np.linspace(0, len(x) - 1, n_target)
    return np.interp(idx, np.arange(len(x)), x)


def main():
    by_path = load_samples()
    for p in range(5):
        print(f'Path {p}: {len(by_path[p])} recordings with heading')

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 5, hspace=0.35, wspace=0.25)

    # Row 1: polar heading histograms per path (averaged across recordings)
    for p in range(5):
        ax = fig.add_subplot(gs[0, p], polar=True)
        # Average the 36-bin histogram across all recordings of this path
        bins = np.linspace(0, 360, 37)
        mean_hist = np.zeros(36)
        for rec in by_path[p]:
            hist, _ = np.histogram(rec['heading'], bins=bins)
            mean_hist += hist / (hist.sum() + 1e-9)
        mean_hist /= max(1, len(by_path[p]))

        theta = np.deg2rad(bins[:-1] + 5)   # bin centers
        ax.bar(theta, mean_hist, width=np.deg2rad(10),
               color=PATH_COLORS[p], alpha=0.8, edgecolor='white', lw=0.5)
        ax.set_theta_direction(-1)          # clockwise (compass-like)
        ax.set_theta_offset(np.pi / 2)      # 0° at top (North)
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(['N', 'E', 'S', 'W'], fontsize=9)
        ax.set_yticklabels([])
        ax.set_title(PATH_LABELS[p], fontsize=10, pad=14)

    # Row 2: heading time series (10 examples per path, overlaid; normalized time)
    TARGET_LEN = 300
    for p in range(5):
        ax = fig.add_subplot(gs[1, p])
        samples = by_path[p][:10]
        for rec in samples:
            h = rec['heading']
            if len(h) < 20:
                continue
            h_r = resample_to(h, TARGET_LEN)
            ax.plot(np.linspace(0, 1, TARGET_LEN), h_r,
                    color=PATH_COLORS[p], alpha=0.35, lw=0.8)
        # Overlay mean heading (via circular mean on resampled series)
        stacked = []
        for rec in samples:
            if len(rec['heading']) >= 20:
                stacked.append(resample_to(rec['heading'], TARGET_LEN))
        if stacked:
            s = np.array(stacked)
            # Circular mean per column (time point)
            mean_h = np.degrees(np.arctan2(
                np.mean(np.sin(np.deg2rad(s)), axis=0),
                np.mean(np.cos(np.deg2rad(s)), axis=0))) % 360
            ax.plot(np.linspace(0, 1, TARGET_LEN), mean_h,
                    color='black', lw=2.0, alpha=0.9, label='circular mean')
        ax.set_title(f'Path {p} headings over time', fontsize=10)
        ax.set_xlabel('Normalized time')
        ax.set_ylabel('Heading [deg]')
        ax.set_ylim(0, 360)
        ax.set_yticks([0, 90, 180, 270, 360])
        ax.grid(alpha=0.3)

    # Row 3: turn intensity |dh/dt| (resampled) per path
    for p in range(5):
        ax = fig.add_subplot(gs[2, p])
        stacked = []
        for rec in by_path[p]:
            h = rec['heading']
            if len(h) < 30:
                continue
            h_r = resample_to(h, TARGET_LEN)
            dh  = np.abs(circ_diff(h_r[1:], h_r[:-1]))
            stacked.append(dh)
        if stacked:
            s = np.array(stacked)
            mean_dh = s.mean(axis=0)
            # Smooth for visibility
            k = 7
            mean_dh = np.convolve(mean_dh, np.ones(k) / k, mode='same')
            t = np.linspace(0, 1, TARGET_LEN - 1)
            ax.plot(t, mean_dh, color=PATH_COLORS[p], lw=2)
            ax.fill_between(t, 0, mean_dh, color=PATH_COLORS[p], alpha=0.3)
        ax.set_title(f'Path {p} turn intensity', fontsize=10)
        ax.set_xlabel('Normalized time')
        ax.set_ylabel('|Δheading| [deg/sample]')
        ax.set_ylim(0, 30)
        ax.grid(alpha=0.3)

    fig.suptitle('Heading profiles per path  '
                 '(Row 1: compass distribution | Row 2: heading over time | '
                 'Row 3: where turns happen)',
                 fontsize=13, fontweight='bold', y=0.995)
    out = os.path.join(RESULTS, 'heading_analysis.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

    # Print textual summary: mean heading histogram per path, top-3 bins
    print('\nDominant heading directions per path (top 3 bins of 10°):')
    for p in range(5):
        bins = np.linspace(0, 360, 37)
        agg = np.zeros(36)
        for rec in by_path[p]:
            hist, _ = np.histogram(rec['heading'], bins=bins)
            agg += hist / (hist.sum() + 1e-9)
        agg /= max(1, len(by_path[p]))
        top = np.argsort(agg)[::-1][:3]
        top_str = ', '.join(f'{int(bins[i]):3d}°({agg[i]*100:.1f}%)' for i in top)
        print(f'  Path {p}: {top_str}')


if __name__ == '__main__':
    main()
