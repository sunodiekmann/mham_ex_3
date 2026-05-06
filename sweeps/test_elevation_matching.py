"""
Sanity check elevation profile matching: best-match accuracy + per-class.
"""

import os
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from elevation_matching import build_elevation_templates, match_recording_elevation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TRAIN = os.path.join(ROOT, 'data', 'train')


def main():
    print('Building elevation templates ...')
    templates = build_elevation_templates()
    for p, t in templates.items():
        print(f'  P{p}: total_change={t["total_change"]:+.1f}m, '
              f'abs_total={t["abs_total"]:.1f}m')

    print('\nProcessing 396 recordings ...')
    rows = []
    for i, fname in enumerate(sorted(os.listdir(DATA_TRAIN))):
        if not fname.endswith('.pkl'):
            continue
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}')
        with open(os.path.join(DATA_TRAIN, fname), 'rb') as f:
            raw = pickle.load(f)
        feats = match_recording_elevation(raw, templates)
        feats['filename']  = fname
        feats['true_path'] = int(raw['labels']['path_idx'])
        rows.append(feats)

    df = pd.DataFrame(rows)
    print(f'\nProcessed {len(df)} recordings.')

    # Best-match accuracy by L2 (magnitude-aware)
    print('\n=== Best-match by L2 (raw, magnitude-aware) ===')
    valid = df[df['elev_best_match_l2'] >= 0]
    acc = (valid['elev_best_match_l2'] == valid['true_path']).mean()
    print(f'  Overall: {acc:.4f} ({len(valid)} valid)')
    for p in range(5):
        sub = valid[valid['true_path'] == p]
        if len(sub) > 0:
            corr_count = (sub['elev_best_match_l2'] == p).sum()
            print(f'  Path {p}: {corr_count}/{len(sub)} = {corr_count/len(sub):.3f}')

    print('\nConfusion matrix (rows=true, cols=l2_match):')
    print(pd.crosstab(valid['true_path'], valid['elev_best_match_l2'], dropna=False))

    # Best-match by correlation (shape-only)
    print('\n=== Best-match by correlation (shape-only) ===')
    valid2 = df[df['elev_best_match_corr'] >= 0]
    acc2 = (valid2['elev_best_match_corr'] == valid2['true_path']).mean()
    print(f'  Overall: {acc2:.4f}')
    for p in range(5):
        sub = valid2[valid2['true_path'] == p]
        if len(sub) > 0:
            corr_count = (sub['elev_best_match_corr'] == p).sum()
            print(f'  Path {p}: {corr_count}/{len(sub)} = {corr_count/len(sub):.3f}')

    # Per-path mean distances (diagonal = lower better for L2/DTW, higher for corr)
    print('\n=== Mean elev_l2_raw to each template, per true class ===')
    for p in range(5):
        sub = df[df['true_path'] == p]
        if len(sub) == 0:
            continue
        means = [sub[f'elev_l2_raw_p{i}'].mean() for i in range(5)]
        diag = means[p]
        argmin_p = int(np.argmin(means))
        marker = '✓' if argmin_p == p else f'(→P{argmin_p})'
        print(f'  True P{p}: dists = {[round(m, 1) for m in means]}  diag={diag:.1f} {marker}')

    print('\n=== Mean correlation to each template, per true class ===')
    for p in range(5):
        sub = df[df['true_path'] == p]
        if len(sub) == 0:
            continue
        means = [sub[f'elev_corr_p{i}'].mean() for i in range(5)]
        argmax_p = int(np.argmax(means))
        marker = '✓' if argmax_p == p else f'(→P{argmax_p})'
        print(f'  True P{p}: corrs = {[round(m, 3) for m in means]}  {marker}')


if __name__ == '__main__':
    main()
