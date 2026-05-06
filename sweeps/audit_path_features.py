"""
Path classifier feature audit.

Goals:
  1. Show which features carry signal vs noise (GBM importance + permutation).
  2. Test dropping low-importance features — does CV improve, hold, or hurt?
  3. Categorize by feature family so we can see if whole groups are dead weight.
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import train_path as tp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_TRAIN = os.path.join(ROOT, 'results', 'features_train.csv')

RANDOM_STATE = 42
N_FOLDS = 5
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def categorize(feat):
    if feat.startswith('turn_') and '_' in feat[5:]:
        return 'turn-sequence (new)'
    if feat in ('num_significant_turns', 'total_signed_rotation',
                'total_abs_rotation', 'turns_positive', 'turns_negative'):
        return 'turn-sequence (new)'
    if feat.startswith('heading_seq_') or feat.startswith('heading_turnrate_') \
            or feat.startswith('heading_seq_'):
        return 'heading trajectory (new)'
    if feat.startswith('heading_hist_'):
        return 'heading histogram'
    if feat.startswith('heading_'):
        return 'heading aggregate'
    if feat.startswith('stair_frac_t') or feat.startswith('stair_burst') \
            or feat.startswith('stair_longest') or feat.startswith('stair_total') \
            or feat.startswith('stair_mass') or feat.startswith('max_deriv_sustained'):
        return 'stair multi-threshold (new)'
    if feat.startswith('stair_'):
        return 'stair original'
    if feat.startswith('pressure_') or feat.startswith('prs_'):
        return 'pressure'
    if feat.startswith('alt') or feat.startswith('altitude'):
        return 'altitude'
    if feat.startswith('acc_'):
        return 'accelerometer'
    return 'other'


# Load with current train_path feature layout
print('Loading ...')
df, y = tp.load_data(tp.DATA_TRAIN, has_labels=True)
df, feat_cols = tp._merge_csv_features(df, FEATURES_TRAIN, tp.HEADING_COLS)
X = df[feat_cols].values
print(f'Features: {len(feat_cols)}  Samples: {len(y)}')

pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('clf', GradientBoostingClassifier(
        n_estimators=153, max_depth=3, learning_rate=0.114,
        min_samples_leaf=7, subsample=0.819, random_state=RANDOM_STATE)),
])

# Baseline CV with all features
baseline_cv = cross_val_score(pipe, X, y, cv=cv, scoring='balanced_accuracy')
print(f'\nBaseline (all features): {baseline_cv.mean():.4f} ± {baseline_cv.std():.4f}')

# Fit once to get GBM importances
pipe.fit(X, y)
importances = pipe['clf'].feature_importances_
rank = np.argsort(importances)[::-1]
imp_df = pd.DataFrame({'feature': [feat_cols[i] for i in rank],
                        'importance': importances[rank],
                        'category': [categorize(feat_cols[i]) for i in rank]})

print('\nTop-20 features:')
for _, row in imp_df.head(20).iterrows():
    print(f'  {row["feature"]:<42s}  {row["importance"]:.4f}   [{row["category"]}]')

print('\nImportance by category (sum + count):')
cat_stats = imp_df.groupby('category').agg(
    total=('importance', 'sum'),
    count=('importance', 'count'),
    avg=('importance', 'mean'),
    nonzero=('importance', lambda x: (x > 0.001).sum()),
).sort_values('total', ascending=False)
print(cat_stats.round(4))

# How many features carry meaningful importance?
nonzero = (importances > 0).sum()
meaningful = (importances > 0.005).sum()
print(f'\n  non-zero importance: {nonzero}/{len(feat_cols)}')
print(f'  importance > 0.005:  {meaningful}/{len(feat_cols)}')

# --- Ablation: progressively drop the bottom features ---
print('\n=== Ablation: keep top-K features ===')
for k in [30, 50, 80, 120, len(feat_cols)]:
    if k > len(feat_cols):
        k = len(feat_cols)
    keep = rank[:k]
    X_sub = X[:, keep]
    scores = cross_val_score(pipe, X_sub, y, cv=cv, scoring='balanced_accuracy')
    print(f'  top-{k:3d} features: {scores.mean():.4f} ± {scores.std():.4f}')

# --- Category ablation: drop one whole category at a time ---
print('\n=== Ablation: drop one category at a time ===')
cats = imp_df['category'].unique()
for c in cats:
    mask = np.array([categorize(f) != c for f in feat_cols])
    X_sub = X[:, mask]
    scores = cross_val_score(pipe, X_sub, y, cv=cv, scoring='balanced_accuracy')
    delta = scores.mean() - baseline_cv.mean()
    marker = '↓' if delta < -0.003 else ('↑' if delta > 0.003 else '·')
    print(f'  drop {c:<30s}  → {scores.mean():.4f} ({delta:+.4f}) {marker}'
          f'  [{sum(~mask):3d} feats removed]')
