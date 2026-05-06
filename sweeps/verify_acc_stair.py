"""
Verify the acc-based stair detector produces sensible per-path distributions.

Expected:
  Path 0 (no stairs):      low acc_indep_stair_frac (ideally < 0.05)
  Path 1 (stairs early):   moderate
  Path 2 (Walchetor):      moderate-high (this is the one pressure misses)
  Path 3 (stairs early):   moderate
  Path 4 (stairs middle):  moderate-high

Also: check how often pressure and acc AGREE vs disagree per path.
Higher agreement score → higher confidence detection.
"""

import os
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_path import extract_path_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TRAIN = os.path.join(ROOT, 'data', 'train')


def main():
    rows = []
    for fname in sorted(os.listdir(DATA_TRAIN)):
        if not fname.endswith('.pkl'):
            continue
        with open(os.path.join(DATA_TRAIN, fname), 'rb') as f:
            raw = pickle.load(f)
        feats = extract_path_features(raw)
        feats['path_idx'] = int(raw['labels']['path_idx'])
        feats['filename'] = fname
        rows.append(feats)
    df = pd.DataFrame(rows)

    print(f'Processed {len(df)} recordings\n')

    show = ['acc_indep_stair_frac', 'acc_indep_stair_burst_count',
            'stair_frac', 'stair_both_frac',
            'stair_prs_only_frac', 'stair_acc_only_frac',
            'stair_agreement_score']
    print('=== Mean by path ===')
    print(df.groupby('path_idx')[show].mean().round(3))

    # Detection rates
    print('\n=== Detection rates per path ===')
    for p in sorted(df['path_idx'].unique()):
        sub = df[df['path_idx'] == p]
        prs_detect = (sub['stair_frac'] > 0).mean()
        acc_detect = (sub['acc_indep_stair_frac'] > 0.02).mean()
        both       = ((sub['stair_frac'] > 0) & (sub['acc_indep_stair_frac'] > 0.02)).mean()
        neither    = ((sub['stair_frac'] == 0) & (sub['acc_indep_stair_frac'] <= 0.02)).mean()
        print(f'  Path {p}: n={len(sub):3d}  '
              f'prs={prs_detect:.3f}  acc={acc_detect:.3f}  '
              f'both={both:.3f}  neither={neither:.3f}')

    # Expected truth labels:
    truth = {0: 'no stairs', 1: 'stairs (Clausiusstr)', 2: 'stairs (Walchetor)',
             3: 'stairs (rev Walchetor)', 4: 'stairs (middle desc)'}
    print('\n=== vs ground truth ===')
    print('Path | Truth          | Prs-detect | Acc-detect | Both  | Agreement')
    for p in sorted(df['path_idx'].unique()):
        sub = df[df['path_idx'] == p]
        prs_detect = (sub['stair_frac'] > 0).mean()
        acc_detect = (sub['acc_indep_stair_frac'] > 0.02).mean()
        both       = ((sub['stair_frac'] > 0) & (sub['acc_indep_stair_frac'] > 0.02)).mean()
        agr        = sub['stair_agreement_score'].mean()
        print(f'  {p}  | {truth[p]:<25} | {prs_detect:.3f}   | {acc_detect:.3f}   | '
              f'{both:.3f} | {agr:.3f}')

    # Per-watch-location view: show detector works on ankle, muted on wrist
    print('\n=== Per watch_loc × path: acc_indep_stair_frac ===')
    piv = df.pivot_table(values='acc_indep_stair_frac',
                          index='path_idx', columns='watch_loc',
                          aggfunc='mean').round(3)
    print(piv)


if __name__ == '__main__':
    main()
