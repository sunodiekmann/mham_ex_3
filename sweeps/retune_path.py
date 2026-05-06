"""
Retune path GBM on the expanded feature set (stair + heading/turn features).
Also compares candidate architectures: single GBM vs two-stage (uphill/downhill).
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import train_path as tp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_TRAIN = os.path.join(ROOT, 'results', 'features_train.csv')

RANDOM_STATE = 42
N_FOLDS = 5
N_ITER  = 60

# Load data through train_path's loader (includes pressure-based feature extraction)
print('Loading data ...')
df, y = tp.load_data(tp.DATA_TRAIN, has_labels=True)
df, feat_cols = tp._merge_csv_features(df, FEATURES_TRAIN, tp.HEADING_COLS)
X = df[feat_cols].values
print(f'Features: {X.shape[1]}  Samples: {X.shape[0]}')

cv     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
search = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE + 1)


def pipe(clf):
    return Pipeline([('imp', SimpleImputer(strategy='median')), ('clf', clf)])


# ---------------------------------------------------------------------------
# 1. Retune single GBM
# ---------------------------------------------------------------------------
print('\n=== Retuning single GBM ===')
gb_search = RandomizedSearchCV(
    pipe(GradientBoostingClassifier(random_state=RANDOM_STATE)),
    {'clf__n_estimators':     randint(100, 600),
     'clf__learning_rate':    loguniform(0.01, 0.25),
     'clf__max_depth':        randint(2, 6),
     'clf__subsample':        uniform(0.6, 0.4),
     'clf__min_samples_leaf': randint(2, 12)},
    n_iter=N_ITER, cv=search, scoring='balanced_accuracy',
    n_jobs=-1, random_state=RANDOM_STATE, refit=True,
)
gb_search.fit(X, y)
print(f'  best balanced_acc = {gb_search.best_score_:.4f}')
print(f'  params            = {gb_search.best_params_}')

outer = cross_val_score(gb_search.best_estimator_, X, y, cv=cv,
                         scoring='balanced_accuracy', n_jobs=-1)
print(f'  outer CV          = {outer.mean():.4f} ± {outer.std():.4f}')


# ---------------------------------------------------------------------------
# 2. Two-stage: uphill (0,1,2) vs downhill (3,4) → subtype
# ---------------------------------------------------------------------------
class TwoStagePath(BaseEstimator, ClassifierMixin):
    """Stage 1: uphill vs downhill (binary). Stage 2: subtype within each group."""
    def __init__(self, stage1=None, stage_up=None, stage_down=None):
        self.stage1 = stage1
        self.stage_up = stage_up
        self.stage_down = stage_down

    def fit(self, X, y):
        self.classes_ = np.array([0, 1, 2, 3, 4])
        y_s1 = (y >= 3).astype(int)    # 0 = uphill (0,1,2), 1 = downhill (3,4)
        self.stage1_ = clone(self.stage1).fit(X, y_s1)

        up_mask = y < 3
        self.stage_up_ = clone(self.stage_up).fit(X[up_mask], y[up_mask])

        down_mask = y >= 3
        self.stage_down_ = clone(self.stage_down).fit(X[down_mask], y[down_mask])
        return self

    def predict(self, X):
        p_down = self.stage1_.predict(X)
        out = np.empty(len(X), dtype=int)
        if (p_down == 0).any():
            out[p_down == 0] = self.stage_up_.predict(X[p_down == 0])
        if (p_down == 1).any():
            out[p_down == 1] = self.stage_down_.predict(X[p_down == 1])
        return out


def make_two_stage():
    # Stage 1 is easy (uphill/downhill ≈ sign of altitude gain) → simple GBM
    s1 = pipe(GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=3,
        min_samples_leaf=4, subsample=0.9, random_state=RANDOM_STATE))
    # Stage 2 uphill: 3-class (0/1/2), hard — needs full regularized GBM
    s_up = pipe(GradientBoostingClassifier(
        **{k.replace('clf__', ''): v for k, v in gb_search.best_params_.items()},
        random_state=RANDOM_STATE))
    # Stage 2 downhill: 2-class (3/4), was already near-perfect — keep simpler
    s_down = pipe(GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=3,
        min_samples_leaf=4, subsample=0.9, random_state=RANDOM_STATE))
    return TwoStagePath(stage1=s1, stage_up=s_up, stage_down=s_down)


# ---------------------------------------------------------------------------
# 3. Evaluate candidates on identical CV folds
# ---------------------------------------------------------------------------
print('\n=== Candidates on identical CV folds ===')

candidates = {
    'Single GBM (retuned)': gb_search.best_estimator_,
    'Two-stage':            make_two_stage(),
}

print(f'\n{"Model":<26} {"Bal Acc":>10} {"Std":>8}   '
      f'{"P0":>5} {"P1":>5} {"P2":>5} {"P3":>5} {"P4":>5}')
print('-' * 78)

for name, model in candidates.items():
    fold_scores = []
    per_class = {k: [] for k in [0, 1, 2, 3, 4]}
    for tr, va in cv.split(X, y):
        m = clone(model)
        m.fit(X[tr], y[tr])
        preds = m.predict(X[va])
        fold_scores.append(balanced_accuracy_score(y[va], preds))
        cm = confusion_matrix(y[va], preds, labels=[0, 1, 2, 3, 4])
        for k in [0, 1, 2, 3, 4]:
            if cm[k].sum() > 0:
                per_class[k].append(cm[k, k] / cm[k].sum())
    mean = np.mean(fold_scores)
    std  = np.std(fold_scores)
    pc   = {k: np.mean(per_class[k]) for k in [0, 1, 2, 3, 4]}
    print(f'{name:<26} {mean:>10.4f} {std:>8.4f}   '
          f'{pc[0]:>5.3f} {pc[1]:>5.3f} {pc[2]:>5.3f} {pc[3]:>5.3f} {pc[4]:>5.3f}')
