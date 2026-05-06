"""
Try ensemble strategies for the path classifier:
  1. Multi-seed GBM (average proba over different random seeds)
  2. GBM + SVM soft voting
  3. GBM + SVM + RF stacking
  4. Two-stage: uphill (0/1/2) vs downhill (3/4) → subtype
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
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

print('Loading data with OOF labels for honest CV ...')
df, y = tp.load_data(tp.DATA_TRAIN, has_labels=True, oof_csv=OOF_CSV)
df, feat_cols = tp._merge_csv_features(df, FEATURES_TRAIN, tp.HEADING_COLS)
X = df[feat_cols].values.astype(float)
print(f'Features: {X.shape[1]}  Samples: {X.shape[0]}')


def make_gbm(seed=RANDOM_STATE):
    return GradientBoostingClassifier(
        n_estimators=153, max_depth=3, learning_rate=0.114,
        min_samples_leaf=7, subsample=0.819, random_state=seed)


def base_gbm_pipe():
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('clf', make_gbm()),
    ])


# ---------------------------------------------------------------------------
# Multi-seed GBM ensemble
# ---------------------------------------------------------------------------
class MultiSeedGBM(BaseEstimator, ClassifierMixin):
    def __init__(self, n_seeds=5, **gbm_kw):
        self.n_seeds = n_seeds
        self.gbm_kw = gbm_kw

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = []
        for s in range(self.n_seeds):
            m = Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('clf', GradientBoostingClassifier(
                    random_state=RANDOM_STATE + s, **self.gbm_kw)),
            ])
            m.fit(X, y)
            self.models_.append(m)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# ---------------------------------------------------------------------------
# Two-stage: uphill (0/1/2) vs downhill (3/4) → subtype
# ---------------------------------------------------------------------------
class TwoStagePath(BaseEstimator, ClassifierMixin):
    def __init__(self, stage1=None, stage_up=None, stage_down=None):
        self.stage1 = stage1
        self.stage_up = stage_up
        self.stage_down = stage_down

    def fit(self, X, y):
        self.classes_ = np.array([0, 1, 2, 3, 4])
        self.s1 = clone(self.stage1).fit(X, (y >= 3).astype(int))
        up_mask = y < 3
        self.s_up = clone(self.stage_up).fit(X[up_mask], y[up_mask])
        self.s_down = clone(self.stage_down).fit(X[y >= 3], y[y >= 3])
        return self

    def predict(self, X):
        p_down = self.s1.predict(X)
        out = np.empty(len(X), dtype=int)
        if (p_down == 0).any():
            out[p_down == 0] = self.s_up.predict(X[p_down == 0])
        if (p_down == 1).any():
            out[p_down == 1] = self.s_down.predict(X[p_down == 1])
        return out


def make_two_stage():
    s1 = base_gbm_pipe()
    s_up = base_gbm_pipe()
    s_down = base_gbm_pipe()
    return TwoStagePath(stage1=s1, stage_up=s_up, stage_down=s_down)


# ---------------------------------------------------------------------------
# Stacking: GBM + SVM + RF → LogReg meta
# ---------------------------------------------------------------------------
def make_stacking():
    base = [
        ('gbm', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('clf', make_gbm()),
        ])),
        ('svm', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc',  StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, gamma='scale',
                         class_weight='balanced', probability=True,
                         random_state=RANDOM_STATE)),
        ])),
        ('rf', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(
                n_estimators=400, min_samples_leaf=3,
                class_weight='balanced', random_state=RANDOM_STATE,
                n_jobs=-1)),
        ])),
    ]
    return StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=5,
    )


# ---------------------------------------------------------------------------
# Soft-voting GBM + SVM
# ---------------------------------------------------------------------------
from sklearn.ensemble import VotingClassifier


def make_voting():
    gbm = Pipeline([('imp', SimpleImputer(strategy='median')), ('clf', make_gbm())])
    svm = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('clf', SVC(kernel='rbf', C=1.0, gamma='scale',
                     class_weight='balanced', probability=True,
                     random_state=RANDOM_STATE)),
    ])
    return VotingClassifier([('gbm', gbm), ('svm', svm)],
                             voting='soft', weights=[2, 1], n_jobs=1)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
candidates = {
    'GBM (current)':       base_gbm_pipe(),
    'Multi-seed GBM (5)':  MultiSeedGBM(n_seeds=5,
                                          n_estimators=153, max_depth=3,
                                          learning_rate=0.114,
                                          min_samples_leaf=7, subsample=0.819),
    'GBM+SVM voting':      make_voting(),
    'Stacking GBM+SVM+RF': make_stacking(),
    'Two-stage (uphill/downhill)': make_two_stage(),
}

print(f'\n{"Model":<32} {"Bal Acc":>10} {"Std":>8}   '
      f'{"P0":>5} {"P1":>5} {"P2":>5} {"P3":>5} {"P4":>5}')
print('-' * 84)
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
    pc = {k: np.mean(per_class[k]) for k in [0, 1, 2, 3, 4]}
    print(f'{name:<32} {np.mean(fold_scores):>10.4f} {np.std(fold_scores):>8.4f}   '
          f'{pc[0]:>5.3f} {pc[1]:>5.3f} {pc[2]:>5.3f} {pc[3]:>5.3f} {pc[4]:>5.3f}')
