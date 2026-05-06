"""
Compare candidate improvements to the watch-location model on identical CV splits.

Baselines  :
  - GBM  (current production model, 0.957 CV)
  - SVM  (retuned on 57 features)

Candidates :
  - Two-stage  (ankle-vs-rest → wrist-vs-belt)
  - Ensemble   (soft-voting GBM + SVM)
  - GBM-reg    (more-regularized GBM, lower learning rate)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin, clone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_watch_loc import FEATURE_COLS, engineer_features, LOC_NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_TRAIN = os.path.join(ROOT, 'results', 'features_train.csv')

RANDOM_STATE = 42
N_FOLDS = 5

df = pd.read_csv(FEATURES_TRAIN)
df = engineer_features(df)
X  = df[FEATURE_COLS].values.astype(float)
y  = df['watch_loc'].values.astype(int)
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def make_pipe(clf):
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('clf', clf),
    ])


# ---------------------------------------------------------------------------
# Two-stage classifier
# Stage 1: GBM for ankle (2) vs rest (0 or 1), binary
# Stage 2: GBM for wrist (0) vs belt (1), binary, trained on non-ankle samples
# Prediction:
#   P(ankle) = stage1 proba
#   if P(ankle) > 0.5 → predict ankle
#   else → stage2 decides wrist vs belt
# ---------------------------------------------------------------------------
class TwoStageWatchLoc(BaseEstimator, ClassifierMixin):
    def __init__(self, stage1=None, stage2=None, ankle_threshold=0.5):
        self.stage1 = stage1
        self.stage2 = stage2
        self.ankle_threshold = ankle_threshold

    def fit(self, X, y):
        self.stage1_ = clone(self.stage1)
        self.stage2_ = clone(self.stage2)

        # Stage 1: ankle (1) vs rest (0)
        y_s1 = (y == 2).astype(int)
        self.stage1_.fit(X, y_s1)

        # Stage 2: wrist (0) vs belt (1), trained only on non-ankle samples
        mask = y != 2
        self.stage2_.fit(X[mask], y[mask])
        return self

    def predict_proba(self, X):
        p_ankle  = self.stage1_.predict_proba(X)[:, 1]
        p_wb     = self.stage2_.predict_proba(X)          # columns: [wrist, belt]
        p_wrist  = (1 - p_ankle) * p_wb[:, 0]
        p_belt   = (1 - p_ankle) * p_wb[:, 1]
        return np.column_stack([p_wrist, p_belt, p_ankle])

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def two_stage_gbm_gbm():
    stage1 = make_pipe(GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=3,
        min_samples_leaf=4, subsample=0.9, random_state=RANDOM_STATE))
    stage2 = make_pipe(GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=3,
        min_samples_leaf=4, subsample=0.9, random_state=RANDOM_STATE))
    return TwoStageWatchLoc(stage1=stage1, stage2=stage2)


def two_stage_svm_gbm():
    # SVM is stronger at ankle (0.978), GBM stronger at wrist/belt
    stage1 = make_pipe(SVC(kernel='rbf', C=1.03, gamma=0.019,
                            class_weight='balanced', probability=True,
                            random_state=RANDOM_STATE))
    stage2 = make_pipe(GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=3,
        min_samples_leaf=4, subsample=0.9, random_state=RANDOM_STATE))
    return TwoStageWatchLoc(stage1=stage1, stage2=stage2)


class MultiSeedGBM(BaseEstimator, ClassifierMixin):
    """Average predicted probabilities over N GBMs with different seeds."""
    def __init__(self, n_seeds=5, **kw):
        self.n_seeds = n_seeds
        self.kw = kw

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = []
        for seed in range(self.n_seeds):
            pipe = make_pipe(GradientBoostingClassifier(
                random_state=RANDOM_STATE + seed, **self.kw))
            pipe.fit(X, y)
            self.models_.append(pipe)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def multi_seed_gbm():
    return MultiSeedGBM(n_seeds=5, n_estimators=385, learning_rate=0.086,
                         max_depth=3, min_samples_leaf=4, subsample=0.95)


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
def gbm_production():
    return make_pipe(GradientBoostingClassifier(
        n_estimators=385, learning_rate=0.086, max_depth=3,
        min_samples_leaf=4, subsample=0.95, random_state=RANDOM_STATE))


def gbm_regularized():
    # Lower LR, more trees, higher leaf min — reduce overfitting
    return make_pipe(GradientBoostingClassifier(
        n_estimators=600, learning_rate=0.04, max_depth=3,
        min_samples_leaf=6, subsample=0.85, random_state=RANDOM_STATE))


def svm_retuned():
    return make_pipe(SVC(kernel='rbf', C=1.03, gamma=0.019,
                          class_weight='balanced', probability=True,
                          random_state=RANDOM_STATE))


def ensemble_gbm_svm():
    gbm = GradientBoostingClassifier(
        n_estimators=385, learning_rate=0.086, max_depth=3,
        min_samples_leaf=4, subsample=0.95, random_state=RANDOM_STATE)
    svm = SVC(kernel='rbf', C=1.03, gamma=0.019,
              class_weight='balanced', probability=True,
              random_state=RANDOM_STATE)
    return Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('clf', VotingClassifier([('gbm', gbm), ('svm', svm)],
                                  voting='soft', weights=[2, 1], n_jobs=-1)),
    ])


candidates = {
    'GBM (current)':       gbm_production,
    'SVM (retuned)':       svm_retuned,
    'Ensemble GBM+SVM':    ensemble_gbm_svm,
    'Multi-seed GBM':      multi_seed_gbm,
    'Two-stage GBM+GBM':   two_stage_gbm_gbm,
    'Two-stage SVM+GBM':   two_stage_svm_gbm,
}

# ---------------------------------------------------------------------------
# Evaluate on the same CV splits
# ---------------------------------------------------------------------------
print(f'Feature count: {X.shape[1]}  Samples: {X.shape[0]}\n')
print(f'{"Model":<22} {"Bal Acc":>12} {"Std":>8}   {"Wrist":>7} {"Belt":>7} {"Ankle":>7}')
print('-' * 70)

results = {}
for name, factory in candidates.items():
    fold_scores = []
    per_class   = {0: [], 1: [], 2: []}
    for tr, va in cv.split(X, y):
        model = factory()
        model.fit(X[tr], y[tr])
        preds = model.predict(X[va])
        fold_scores.append(balanced_accuracy_score(y[va], preds))
        cm = confusion_matrix(y[va], preds, labels=[0, 1, 2])
        for k in [0, 1, 2]:
            if cm[k].sum() > 0:
                per_class[k].append(cm[k, k] / cm[k].sum())
    mean  = np.mean(fold_scores)
    std   = np.std(fold_scores)
    pc    = {k: np.mean(per_class[k]) for k in [0, 1, 2]}
    print(f'{name:<22} {mean:>12.4f} {std:>8.4f}   '
          f'{pc[0]:>7.3f} {pc[1]:>7.3f} {pc[2]:>7.3f}')
    results[name] = mean

best = max(results, key=results.get)
print(f'\nBest: {best}  ({results[best]:.4f})')
