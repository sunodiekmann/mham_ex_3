"""
Test per-activity feature subsetting. Some categories add noise for some
activities. Try dropping noisy categories per activity:
  running:  context, phone IMU stats (audit suggested both help to drop)
  cycling:  phone IMU stats (audit: +0.015 if dropped)

Uses the ACTUAL production GBM (per-activity tuned hyperparameters).
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import train_activities as tact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_TRAIN = os.path.join(ROOT, 'results', 'features_train.csv')

cv = StratifiedKFold(n_splits=tact.N_FOLDS, shuffle=True, random_state=tact.RANDOM_STATE)


def is_phone_imu(f):
    return f.startswith('phone_')


def is_context(f):
    return f in ('duration', 'altitude_range', 'altitude_net_gain',
                  'temp_slope', 'watch_loc')


def is_watch_acc(f):
    return (f.startswith('acc_') and not f.startswith('acc_corr')) \
        or f == 'acc_high_std_frac' or f == 'acc_anisotropy'


df = pd.read_csv(FEATURES_TRAIN)

# All features available
all_feats = tact.FEATURE_COLS

# Test dropping subsets
print(f'{"Activity":<10} {"Subset":<35} {"Bal Acc":>10}')
print('-' * 60)

candidates = {
    'all features':              lambda f: True,
    'drop phone IMU stats':      lambda f: not is_phone_imu(f),
    'drop context':              lambda f: not is_context(f),
    'drop phone + context':      lambda f: not (is_phone_imu(f) or is_context(f)),
    'drop watch acc stats':      lambda f: not is_watch_acc(f),
    'drop phone + watch_acc':    lambda f: not (is_phone_imu(f) or is_watch_acc(f)),
}

best_per_activity = {}
for activity in tact.ACTIVITIES:
    y = df[activity].values.astype(int)
    pipe_factory = lambda: tact.make_pipe(activity)

    activity_results = {}
    for subset_name, keep_fn in candidates.items():
        feats = [f for f in all_feats if keep_fn(f)]
        if len(feats) < 5:
            continue
        X = df[feats].values.astype(float)
        scores = cross_val_score(pipe_factory(), X, y, cv=cv,
                                  scoring='balanced_accuracy', n_jobs=-1)
        activity_results[subset_name] = (scores.mean(), scores.std(), len(feats))
        print(f'{activity:<10} {subset_name:<35} {scores.mean():>7.4f} ± '
              f'{scores.std():.4f}  ({len(feats)} feats)')

    best = max(activity_results.items(), key=lambda x: x[1][0])
    best_per_activity[activity] = best
    print(f'  → BEST: {best[0]} → {best[1][0]:.4f}')
    print()

print('\n=== Recommended per-activity feature subsets ===')
for act, (name, (score, std, n)) in best_per_activity.items():
    print(f'  {act}: {name}  → {score:.4f} (was current vs all-features)')
