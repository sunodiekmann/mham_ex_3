"""
Sanity check: reconstruct each train recording's path from sensors,
match against templates, and see how often best-match agrees with truth.

This is a baseline check before integrating into the classifier.
"""

import os
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from path_reconstruction import (build_path_templates, match_recording_to_templates)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TRAIN = os.path.join(ROOT, 'data', 'train')


def main():
    print('Building templates from GPX ...')
    templates = build_path_templates()
    for p, t in templates.items():
        print(f'  Path {p}: {t["n_bins"]} bins, sign={t["sign"]:+d}')

    files = sorted(f for f in os.listdir(DATA_TRAIN) if f.endswith('.pkl'))
    rows = []
    for i, fname in enumerate(files):
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{len(files)}')
        with open(os.path.join(DATA_TRAIN, fname), 'rb') as f:
            raw = pickle.load(f)
        feats = match_recording_to_templates(raw, templates)
        feats['filename'] = fname
        feats['true_path'] = int(raw['labels']['path_idx'])
        rows.append(feats)

    df = pd.DataFrame(rows)
    print(f'\nProcessed {len(df)} recordings.')
    print(f'  rec_n_bins: median={int(df["rec_n_bins"].median())}, '
          f'p25={int(df["rec_n_bins"].quantile(0.25))}, '
          f'p75={int(df["rec_n_bins"].quantile(0.75))}')
    print(f'  rec_sign distribution: {df["rec_sign"].value_counts().to_dict()}')
    print(f'  recordings with <5 bins: {(df["rec_n_bins"] < 5).sum()}')

    # Confusion: best-match vs truth
    valid = df[df['rec_best_match'] >= 0].copy()
    print(f'\nBest-match accuracy on valid recordings ({len(valid)}/{len(df)}):')
    acc = (valid['rec_best_match'] == valid['true_path']).mean()
    print(f'  Overall: {acc:.4f}')
    for p in range(5):
        sub = valid[valid['true_path'] == p]
        if len(sub) > 0:
            corr = (sub['rec_best_match'] == p).mean()
            print(f'  Path {p}: {corr:.4f}  (n={len(sub)})')

    # Confusion matrix
    print('\nConfusion matrix (rows = true, cols = best_match):')
    cm = pd.crosstab(valid['true_path'], valid['rec_best_match'],
                       margins=False, dropna=False)
    print(cm)

    # Distance distributions
    print('\nMean distance to each template, per true class:')
    for p in range(5):
        sub = valid[valid['true_path'] == p]
        if len(sub) == 0:
            continue
        means = [sub[f'rec_dist_p{i}'].mean() for i in range(5)]
        diag = means[p]
        print(f'  True P{p}: dists = {means}  → diag={diag:.2f}, '
              f'argmin={int(np.argmin(means))}')


if __name__ == '__main__':
    main()
