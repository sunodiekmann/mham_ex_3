"""
Final retune of GBM hyperparameters on the pipeline with SelectFromModel(80).
Also compares keeping the selector vs dropping it.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import train_path as tp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_TRAIN = os.path.join(ROOT, 'results', 'features_train.csv')

RANDOM_STATE = 42
N_FOLDS = 5
N_ITER  = 60

print('Loading data ...')
df, y = tp.load_data(tp.DATA_TRAIN, has_labels=True)
df, feat_cols = tp._merge_csv_features(df, FEATURES_TRAIN, tp.HEADING_COLS)
X = df[feat_cols].values
print(f'Features: {X.shape[1]}  Samples: {X.shape[0]}')

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
search_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE + 1)


def make_pipeline(with_selector=True, max_features=80, gbm_kwargs=None):
    gbm_kwargs = gbm_kwargs or {}
    steps = [
        ('imp', SimpleImputer(strategy='median')),
    ]
    if with_selector:
        sel = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=5, random_state=RANDOM_STATE)
        steps.append(('fs', SelectFromModel(sel, max_features=max_features,
                                             threshold=-np.inf)))
    steps.append(('clf', GradientBoostingClassifier(
        random_state=RANDOM_STATE, **gbm_kwargs)))
    return Pipeline(steps)


# 1. Retune with feature selection
print('\n=== Retune GBM (with top-80 selection) ===')
search = RandomizedSearchCV(
    make_pipeline(with_selector=True),
    {'clf__n_estimators':     randint(100, 500),
     'clf__learning_rate':    loguniform(0.02, 0.25),
     'clf__max_depth':        randint(2, 6),
     'clf__subsample':        uniform(0.6, 0.4),
     'clf__min_samples_leaf': randint(2, 12)},
    n_iter=N_ITER, cv=search_cv, scoring='balanced_accuracy',
    n_jobs=-1, random_state=RANDOM_STATE, refit=True,
)
search.fit(X, y)
print(f'  best search score = {search.best_score_:.4f}')
print(f'  params            = {search.best_params_}')
outer = cross_val_score(search.best_estimator_, X, y, cv=cv,
                         scoring='balanced_accuracy', n_jobs=-1)
print(f'  outer CV          = {outer.mean():.4f} ± {outer.std():.4f}')

# 2. Retune without feature selection (baseline for comparison)
print('\n=== Retune GBM (no feature selection) ===')
search_no = RandomizedSearchCV(
    make_pipeline(with_selector=False),
    {'clf__n_estimators':     randint(100, 500),
     'clf__learning_rate':    loguniform(0.02, 0.25),
     'clf__max_depth':        randint(2, 6),
     'clf__subsample':        uniform(0.6, 0.4),
     'clf__min_samples_leaf': randint(2, 12)},
    n_iter=N_ITER, cv=search_cv, scoring='balanced_accuracy',
    n_jobs=-1, random_state=RANDOM_STATE, refit=True,
)
search_no.fit(X, y)
print(f'  best search score = {search_no.best_score_:.4f}')
print(f'  params            = {search_no.best_params_}')
outer_no = cross_val_score(search_no.best_estimator_, X, y, cv=cv,
                            scoring='balanced_accuracy', n_jobs=-1)
print(f'  outer CV          = {outer_no.mean():.4f} ± {outer_no.std():.4f}')

print('\n=== Summary ===')
print(f'  With    selector: {outer.mean():.4f} ± {outer.std():.4f}')
print(f'  Without selector: {outer_no.mean():.4f} ± {outer_no.std():.4f}')
