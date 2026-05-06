"""
Try HistGradientBoostingClassifier (sklearn's LightGBM-like) on path.
Also test ExtraTrees and a deeper RandomForest as alternatives.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform

from sklearn.base import clone
from sklearn.ensemble import (HistGradientBoostingClassifier, ExtraTreesClassifier,
                                RandomForestClassifier, GradientBoostingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import train_path as tp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'results')
FEATURES_TRAIN = os.path.join(RESULTS_DIR, 'features_train.csv')
OOF_CSV = os.path.join(RESULTS_DIR, 'oof_predictions.csv')

RANDOM_STATE = 42
N_FOLDS = 5
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
search_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE+1)

print('Loading data with OOF labels ...')
df, y = tp.load_data(tp.DATA_TRAIN, has_labels=True, oof_csv=OOF_CSV)
df, feat_cols = tp._merge_csv_features(df, FEATURES_TRAIN, tp.HEADING_COLS)
X = df[feat_cols].values.astype(float)
print(f'Features: {X.shape[1]}  Samples: {X.shape[0]}\n')


# Baseline: current production GBM
print('Baseline GBM (production): ...', end=' ', flush=True)
base_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('clf', GradientBoostingClassifier(
        n_estimators=153, max_depth=3, learning_rate=0.114,
        min_samples_leaf=7, subsample=0.819, random_state=RANDOM_STATE)),
])
base = cross_val_score(base_pipe, X, y, cv=cv, scoring='balanced_accuracy')
print(f'{base.mean():.4f} ± {base.std():.4f}')


# 1. HistGradientBoostingClassifier with default params
print('\nHistGBM defaults: ...', end=' ', flush=True)
hist_pipe = Pipeline([
    ('clf', HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
])
s = cross_val_score(hist_pipe, X, y, cv=cv, scoring='balanced_accuracy')
print(f'{s.mean():.4f} ± {s.std():.4f}')


# 2. HistGBM tuned
print('\nHistGBM tuned (RandomizedSearch 50 iter): ...')
hist_search = RandomizedSearchCV(
    Pipeline([('clf', HistGradientBoostingClassifier(random_state=RANDOM_STATE))]),
    {'clf__max_iter': randint(100, 500),
     'clf__learning_rate': loguniform(0.02, 0.25),
     'clf__max_depth': [None, 3, 4, 5, 6, 8],
     'clf__min_samples_leaf': randint(5, 30),
     'clf__l2_regularization': loguniform(0.001, 1.0)},
    n_iter=50, cv=search_cv, scoring='balanced_accuracy',
    n_jobs=-1, random_state=RANDOM_STATE, refit=True,
)
hist_search.fit(X, y)
print(f'  best inner CV : {hist_search.best_score_:.4f}')
print(f'  params        : {hist_search.best_params_}')
outer = cross_val_score(hist_search.best_estimator_, X, y, cv=cv,
                         scoring='balanced_accuracy')
print(f'  outer CV      : {outer.mean():.4f} ± {outer.std():.4f}')


# 3. ExtraTrees
print('\nExtraTrees (n_est=500): ...', end=' ', flush=True)
et_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('clf', ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3,
                                   class_weight='balanced',
                                   random_state=RANDOM_STATE, n_jobs=-1)),
])
s = cross_val_score(et_pipe, X, y, cv=cv, scoring='balanced_accuracy')
print(f'{s.mean():.4f} ± {s.std():.4f}')


# 4. Deeper RandomForest tuned
print('\nRandomForest tuned: ...')
rf_search = RandomizedSearchCV(
    Pipeline([('imp', SimpleImputer(strategy='median')),
                ('clf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))]),
    {'clf__n_estimators': randint(200, 800),
     'clf__max_depth': [None, 8, 12, 16, 20, 30],
     'clf__min_samples_leaf': randint(1, 8),
     'clf__max_features': ['sqrt', 'log2', 0.3, 0.5],
     'clf__class_weight': ['balanced', None]},
    n_iter=30, cv=search_cv, scoring='balanced_accuracy',
    n_jobs=-1, random_state=RANDOM_STATE, refit=True,
)
rf_search.fit(X, y)
outer_rf = cross_val_score(rf_search.best_estimator_, X, y, cv=cv,
                            scoring='balanced_accuracy')
print(f'  best inner CV : {rf_search.best_score_:.4f}')
print(f'  outer CV      : {outer_rf.mean():.4f} ± {outer_rf.std():.4f}')


# 5. HistGBM + GBM voting ensemble
from sklearn.ensemble import VotingClassifier
print('\nHistGBM + GBM voting (soft): ...', end=' ', flush=True)
vote = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('clf', VotingClassifier([
        ('gbm', GradientBoostingClassifier(
            n_estimators=153, max_depth=3, learning_rate=0.114,
            min_samples_leaf=7, subsample=0.819, random_state=RANDOM_STATE)),
        ('hgb', HistGradientBoostingClassifier(
            **{k.replace('clf__', ''): v for k, v in hist_search.best_params_.items()},
            random_state=RANDOM_STATE)),
    ], voting='soft', n_jobs=1)),
])
s = cross_val_score(vote, X, y, cv=cv, scoring='balanced_accuracy')
print(f'{s.mean():.4f} ± {s.std():.4f}')


print('\n' + '=' * 50)
print('Summary')
print('=' * 50)
print(f'  Baseline GBM           : {base.mean():.4f} ± {base.std():.4f}')
print(f'  HistGBM (tuned)        : {outer.mean():.4f} ± {outer.std():.4f}')
print(f'  RandomForest (tuned)   : {outer_rf.mean():.4f} ± {outer_rf.std():.4f}')
print(f'  HistGBM+GBM voting     : {s.mean():.4f} ± {s.std():.4f}')
