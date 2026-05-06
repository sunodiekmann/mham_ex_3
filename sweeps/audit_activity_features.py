"""
Activity classifier feature audit + per-activity tuning.

Goals:
  1. Per activity, see which features are most important (GBM importance).
  2. Try retuning GBM per activity (each activity may benefit from different
     regularization given its imbalance).
  3. Ablation: drop low-importance feature categories.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import randint, uniform, loguniform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_activities import FEATURE_COLS, ACTIVITIES, RANDOM_STATE, N_FOLDS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_TRAIN = os.path.join(ROOT, 'results', 'features_train.csv')

df = pd.read_csv(FEATURES_TRAIN)
X = df[FEATURE_COLS].values.astype(float)
print(f'Features: {len(FEATURE_COLS)}  Samples: {len(df)}')

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def categorize(feat):
    if feat.startswith('seg_watch_'):  return 'segment-watch (new)'
    if feat.startswith('seg_phone_'):  return 'segment-phone (new)'
    if feat.startswith('aerial_'):     return 'aerial-phase (new)'
    if feat.startswith('phone_'):      return 'phone IMU stats'
    if feat.startswith('acc_corr_'):   return 'acc cross-axis corr'
    if feat.startswith('acc_'):        return 'watch acc stats'
    if feat.startswith('gyro_'):       return 'watch gyro stats'
    if feat in ('duration', 'altitude_range', 'altitude_net_gain',
                'temp_slope', 'watch_loc'):
        return 'context'
    return 'other'


# ---------------------------------------------------------------------------
# Per-activity importance + drop-category ablation
# ---------------------------------------------------------------------------
print()
for activity in ACTIVITIES:
    y = df[activity].values.astype(int)
    pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, min_samples_leaf=3, random_state=RANDOM_STATE)),
    ])

    # Baseline CV
    base_score = cross_val_score(pipe, X, y, cv=cv, scoring='balanced_accuracy').mean()
    pipe.fit(X, y)
    importances = pipe['clf'].feature_importances_
    rank = np.argsort(importances)[::-1]

    print(f'\n=== {activity.upper()}  baseline GBM CV={base_score:.4f} ===')
    print('Top 8 features:')
    for i in rank[:8]:
        print(f'  {FEATURE_COLS[i]:<40s}  {importances[i]:.4f}  [{categorize(FEATURE_COLS[i])}]')

    # Category-drop ablation
    cats = sorted(set(categorize(f) for f in FEATURE_COLS))
    print(' Drop category → CV change:')
    for c in cats:
        keep_idx = [i for i, f in enumerate(FEATURE_COLS) if categorize(f) != c]
        if not keep_idx:
            continue
        X_sub = X[:, keep_idx]
        s = cross_val_score(pipe, X_sub, y, cv=cv, scoring='balanced_accuracy').mean()
        delta = s - base_score
        marker = '↑' if delta > 0.003 else ('↓' if delta < -0.003 else '·')
        print(f'   drop {c:<25s}: {s:.4f} ({delta:+.4f}) {marker}'
              f'   [{len(FEATURE_COLS) - len(keep_idx)} feats]')


# ---------------------------------------------------------------------------
# Per-activity hyperparameter tuning
# ---------------------------------------------------------------------------
print('\n=== Per-activity hyperparameter retuning (RandomizedSearchCV 30 iter) ===')
search_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE + 1)

for activity in ACTIVITIES:
    y = df[activity].values.astype(int)
    pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('clf', GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])
    s = RandomizedSearchCV(
        pipe,
        {'clf__n_estimators': randint(100, 500),
         'clf__learning_rate': loguniform(0.02, 0.2),
         'clf__max_depth': randint(2, 6),
         'clf__subsample': uniform(0.6, 0.4),
         'clf__min_samples_leaf': randint(2, 10)},
        n_iter=30, cv=search_cv, scoring='balanced_accuracy',
        n_jobs=-1, random_state=RANDOM_STATE, refit=True,
    )
    s.fit(X, y)
    outer = cross_val_score(s.best_estimator_, X, y, cv=cv,
                             scoring='balanced_accuracy', n_jobs=-1)
    print(f'  {activity:<10s}: best={s.best_score_:.4f}  outer={outer.mean():.4f} '
          f'± {outer.std():.4f}  params={s.best_params_}')
