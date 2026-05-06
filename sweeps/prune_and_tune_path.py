"""
Combined step:
  1. Test different N_ALT_SEGMENTS (10 vs 15 vs 20 for altitude-indexed heading).
  2. For each, run importance-based pruning: find the K that maximizes CV.
  3. Report final best configuration.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import train_path as tp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_TRAIN = os.path.join(ROOT, 'results', 'features_train.csv')

RANDOM_STATE = 42
N_FOLDS = 5
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def make_pipe():
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('clf', GradientBoostingClassifier(
            n_estimators=153, max_depth=3, learning_rate=0.114,
            min_samples_leaf=7, subsample=0.819, random_state=RANDOM_STATE)),
    ])


def load_for_n_alt_segs(n_alt):
    """Reload path features with altered N_ALT_SEGMENTS."""
    tp.N_ALT_SEGMENTS = n_alt
    tp.ALT_HEADING_COLS = (
        [f'alt_heading_{i}_sin' for i in range(n_alt)] +
        [f'alt_heading_{i}_cos' for i in range(n_alt)]
    )
    df, y = tp.load_data(tp.DATA_TRAIN, has_labels=True)
    df, feat_cols = tp._merge_csv_features(df, FEATURES_TRAIN, tp.HEADING_COLS)
    X = df[feat_cols].values
    return X, y, feat_cols


print('=== Altitude-segment count sweep + importance pruning ===\n')
results = []
for n_alt in [10, 15, 20]:
    print(f'\n--- N_ALT_SEGMENTS = {n_alt} ---')
    X, y, feat_cols = load_for_n_alt_segs(n_alt)
    print(f'Total features: {len(feat_cols)}')

    # Baseline CV with all features
    pipe = make_pipe()
    all_score = cross_val_score(pipe, X, y, cv=cv, scoring='balanced_accuracy')
    print(f'All features:  {all_score.mean():.4f} ± {all_score.std():.4f}')

    # Get GBM importances (fit on full data, then use for ranking)
    pipe.fit(X, y)
    importances = pipe['clf'].feature_importances_
    rank = np.argsort(importances)[::-1]

    # Try different top-K
    best_k = len(feat_cols)
    best_score = all_score.mean()
    best_std = all_score.std()
    for k in [40, 60, 80, 100, 120, 150]:
        if k > len(feat_cols):
            continue
        keep = rank[:k]
        X_sub = X[:, keep]
        s = cross_val_score(make_pipe(), X_sub, y, cv=cv, scoring='balanced_accuracy')
        marker = '★' if s.mean() > best_score else ' '
        print(f'  top-{k:3d}: {s.mean():.4f} ± {s.std():.4f} {marker}')
        if s.mean() > best_score:
            best_score = s.mean()
            best_std = s.std()
            best_k = k

    results.append({'n_alt': n_alt, 'best_k': best_k, 'best_score': best_score,
                    'best_std': best_std, 'all_features': len(feat_cols),
                    'all_score': all_score.mean()})

print('\n=== Summary ===')
print(f'{"n_alt":<7} {"all feats":<10} {"all score":<12} {"best K":<8} {"best score":<12}')
for r in results:
    print(f"{r['n_alt']:<7} {r['all_features']:<10} {r['all_score']:<12.4f} "
          f"{r['best_k']:<8} {r['best_score']:<12.4f}")

winner = max(results, key=lambda r: r['best_score'])
print(f"\nWinner: N_ALT_SEGMENTS={winner['n_alt']}, top-{winner['best_k']} features "
      f"→ {winner['best_score']:.4f} ± {winner['best_std']:.4f}")
