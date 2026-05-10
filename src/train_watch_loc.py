"""
Watch-location classifier (wrist / belt / ankle).

Inspired by Kunze et al. "Where am I: Recognizing On-body Positions of
Wearable Sensors" (LoCA 2005).

Key idea: orientation-invariant accelerometer magnitude features extracted
during walking distinguish body-worn sensor positions:
  - Ankle : sharp heel-strike impulse  → high acc std, high gyro std
  - Wrist : arm-swing rotation         → moderate acc std, moderate gyro std
  - Belt  : hip sway only              → low acc std, low gyro std
"""

import os
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

# When run as __main__, also register as 'train_watch_loc' so custom classes
# defined below pickle/unpickle cleanly when loaded elsewhere (e.g. predict.py).
if __name__ == '__main__':
    sys.modules.setdefault('train_watch_loc', sys.modules[__name__])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(ROOT, 'models')
RESULTS_DIR = os.path.join(ROOT, 'results')
FEATURES_TRAIN = os.path.join(RESULTS_DIR, 'features_train.csv')
MODEL_OUT      = os.path.join(MODELS_DIR, 'watch_loc.pkl')

RANDOM_STATE = 42
N_FOLDS      = 5
WATCH_TOP_K_FEATURES = 50

LABEL     = 'watch_loc'
LOC_NAMES = {0: 'Wrist', 1: 'Belt', 2: 'Ankle'}

WATCH_ACC_FEATS = [
    'acc_mean_p25', 'acc_mean_p50', 'acc_mean_p75',
    'acc_std_p25',  'acc_std_p50',  'acc_std_p75',
    'acc_energy_p25', 'acc_energy_p50', 'acc_energy_p75',
    'acc_domfreq_p25', 'acc_domfreq_p50', 'acc_domfreq_p75',
    'acc_high_std_frac',
]

WATCH_GYRO_FEATS = [
    'gyro_std_p25',  'gyro_std_p50',  'gyro_std_p75',
    'gyro_energy_p25', 'gyro_energy_p50', 'gyro_energy_p75',
    'gyro_domfreq_p25', 'gyro_domfreq_p50', 'gyro_domfreq_p75',
]

DERIVED_FEATS = [
    'gyro_domfreq_iqr',
    'watch_phone_gyro_ratio',
    'watch_phone_acc_ratio',
    'gyro_acc_std_ratio_p25', 'gyro_acc_std_ratio_p50', 'gyro_acc_std_ratio_p75',
    'gyro_std_iqr',
    'log_gyro_energy_p50', 'log_acc_energy_p50',
    'acc_anisotropy',
]

# Spec-derived orientation features (from features.py):
#   - Wrist/Ankle have fixed watch orientation → one axis carries gravity.
#   - Belt rotation is unspecified → gravity spreads across axes.
ORIENTATION_FEATS = [
    'grav_abs_x', 'grav_abs_y', 'grav_abs_z',
    'grav_frac_x', 'grav_frac_y', 'grav_frac_z',
    'grav_secondary_ratio', 'grav_spread',
    'grav_dom_axis_x', 'grav_dom_axis_y', 'grav_dom_axis_z',
    'grav_axis_consistency', 'grav_tilt',
]

PER_AXIS_DYNAMIC_FEATS = [
    'axis_std_x', 'axis_std_y', 'axis_std_z',
    'axis_walkband_energy_x', 'axis_walkband_energy_y', 'axis_walkband_energy_z',
    'axis_walkband_frac_x', 'axis_walkband_frac_y', 'axis_walkband_frac_z',
    'axis_walkband_dom_x', 'axis_walkband_dom_y', 'axis_walkband_dom_z',
]

PER_AXIS_GYRO_FEATS = [
    'gyro_axis_std_x', 'gyro_axis_std_y', 'gyro_axis_std_z',
    'gyro_axis_walkband_energy_x', 'gyro_axis_walkband_energy_y', 'gyro_axis_walkband_energy_z',
    'gyro_axis_walkband_frac_x', 'gyro_axis_walkband_frac_y', 'gyro_axis_walkband_frac_z',
    'gyro_axis_walkband_dom_x', 'gyro_axis_walkband_dom_y', 'gyro_axis_walkband_dom_z',
]

FEATURE_COLS = (WATCH_ACC_FEATS + WATCH_GYRO_FEATS + DERIVED_FEATS
                + ORIENTATION_FEATS + PER_AXIS_DYNAMIC_FEATS + PER_AXIS_GYRO_FEATS)


def select_feature_cols(df, y, feature_cols=FEATURE_COLS, top_k=WATCH_TOP_K_FEATURES):
    """Rank features with a simple GBM and keep the strongest subset."""
    feature_cols = list(feature_cols)
    if top_k is None or top_k >= len(feature_cols):
        return feature_cols

    rank_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', GradientBoostingClassifier(
            n_estimators=385, learning_rate=0.086, max_depth=3,
            min_samples_leaf=4, subsample=0.95, random_state=RANDOM_STATE)),
    ])
    X_rank = df[feature_cols].values.astype(float)
    rank_pipe.fit(X_rank, y)
    rank = np.argsort(rank_pipe['clf'].feature_importances_)[::-1]
    return [feature_cols[i] for i in rank[:top_k]]


class TwoStageWatchLoc(BaseEstimator, ClassifierMixin):
    """
    Two-stage watch-location classifier.

    Stage 1 (SVM-RBF): binary ankle-vs-rest. SVM is stronger on ankle recall
                       (0.978 vs 0.964) and more stable (std 0.008).
    Stage 2 (GBM):     binary wrist-vs-belt, trained only on non-ankle samples.
                       GBM wins on this harder pair.

    Predicted probabilities:
      P(ankle) = stage1 proba
      P(wrist) = (1 - P(ankle)) * P(wrist | not ankle)
      P(belt ) = (1 - P(ankle)) * P(belt  | not ankle)

    Outer-CV balanced accuracy: 0.964 (vs 0.951 single GBM, 0.933 baseline).
    """
    def __init__(self, stage1=None, stage2=None):
        self.stage1 = stage1
        self.stage2 = stage2

    def fit(self, X, y):
        self.classes_ = np.array([0, 1, 2])
        self.stage1_ = clone(self.stage1)
        self.stage2_ = clone(self.stage2)

        # Stage 1: ankle (label 1) vs rest (label 0)
        y_s1 = (y == 2).astype(int)
        self.stage1_.fit(X, y_s1)

        # Stage 2: wrist (0) vs belt (1), trained only on non-ankle samples
        mask = y != 2
        self.stage2_.fit(X[mask], y[mask])
        return self

    def predict_proba(self, X):
        p_ankle = self.stage1_.predict_proba(X)[:, 1]
        p_wb    = self.stage2_.predict_proba(X)   # [:, 0]=wrist, [:, 1]=belt
        p_wrist = (1 - p_ankle) * p_wb[:, 0]
        p_belt  = (1 - p_ankle) * p_wb[:, 1]
        return np.column_stack([p_wrist, p_belt, p_ankle])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# Pickle writes the class path as <__module__>.<__qualname__>. Force __module__
# to 'train_watch_loc' so predict.py (which imports this module normally) can
# unpickle, regardless of whether we were running as __main__ when we saved.
# The sys.modules registration at the top of the file ensures pickle's identity
# check (`train_watch_loc.TwoStageWatchLoc is TwoStageWatchLoc`) succeeds.
TwoStageWatchLoc.__module__ = 'train_watch_loc'


def engineer_features(df):
    """Compute derived features expected by the watch location model."""
    df = df.copy()
    df['gyro_domfreq_iqr']       = df['gyro_domfreq_p75']  - df['gyro_domfreq_p25']
    df['watch_phone_gyro_ratio'] = df['gyro_std_p50']  / (df['phone_gyro_std_p50'] + 1e-9)
    df['watch_phone_acc_ratio']  = df['acc_std_p50']   / (df['phone_acc_std_p50']  + 1e-9)
    df['gyro_acc_std_ratio_p25'] = df['gyro_std_p25']  / (df['acc_std_p25'] + 1e-9)
    df['gyro_acc_std_ratio_p50'] = df['gyro_std_p50']  / (df['acc_std_p50'] + 1e-9)
    df['gyro_acc_std_ratio_p75'] = df['gyro_std_p75']  / (df['acc_std_p75'] + 1e-9)
    df['gyro_std_iqr']           = df['gyro_std_p75']  - df['gyro_std_p25']
    df['log_gyro_energy_p50']    = np.log1p(df['gyro_energy_p50'])
    df['log_acc_energy_p50']     = np.log1p(df['acc_energy_p50'])
    return df


if __name__ == '__main__':
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(FEATURES_TRAIN)
    df = engineer_features(df)
    y = df[LABEL].values.astype(int)
    selected_feature_cols = select_feature_cols(df, y)
    print(f'Loaded {len(df)} training recordings, {len(selected_feature_cols)} features selected.')
    if len(selected_feature_cols) < len(FEATURE_COLS):
        print(f'Feature pruning: top-{len(selected_feature_cols)} of {len(FEATURE_COLS)} watch features')

    X = df[selected_feature_cols].values.astype(float)
    print(f'Class distribution: { {LOC_NAMES[k]: int((y==k).sum()) for k in [0,1,2]} }')

    # Two-stage (SVM ankle-vs-rest → GBM wrist-vs-belt) outperforms every
    # single-model variant on the 69-feature set. See sweeps/compare_watch_loc.py.
    def _pipe(clf):
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  StandardScaler()),
            ('clf',     clf),
        ])

    stage1 = _pipe(SVC(kernel='rbf', C=1.03, gamma=0.019,
                        class_weight='balanced', probability=True,
                        random_state=RANDOM_STATE))
    stage2 = _pipe(GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=3,
        min_samples_leaf=4, subsample=0.9, random_state=RANDOM_STATE))
    pipe = TwoStageWatchLoc(stage1=stage1, stage2=stage2)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        pipe, X, y, cv=cv,
        scoring=['accuracy', 'balanced_accuracy'],
        return_train_score=True,
        return_estimator=True,
    )

    print(f'\n=== {N_FOLDS}-fold Stratified Cross-Validation ===')
    print(f'  Train accuracy   : {cv_results["train_accuracy"].mean():.3f} ± {cv_results["train_accuracy"].std():.3f}')
    print(f'  Val accuracy     : {cv_results["test_accuracy"].mean():.3f} ± {cv_results["test_accuracy"].std():.3f}')
    print(f'  Val balanced acc : {cv_results["test_balanced_accuracy"].mean():.3f} ± {cv_results["test_balanced_accuracy"].std():.3f}')

    all_true, all_pred = [], []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        estimator = cv_results['estimator'][fold_idx]
        all_true.extend(y[val_idx])
        all_pred.extend(estimator.predict(X[val_idx]))

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    print(f'\n=== Aggregated Classification Report ===')
    print(classification_report(
        all_true, all_pred,
        target_names=[LOC_NAMES[k] for k in [0, 1, 2]],
        digits=3,
    ))
    print('Per-class accuracy:')
    for k in [0, 1, 2]:
        mask  = all_true == k
        acc_k = (all_pred[mask] == all_true[mask]).mean()
        print(f'  {LOC_NAMES[k]:6s}: {acc_k:.3f}  (n={mask.sum()})')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cm = confusion_matrix(all_true, all_pred, normalize='true')
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[LOC_NAMES[k] for k in [0, 1, 2]],
    ).plot(ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title(
        f'Confusion Matrix (normalised)\n'
        f'CV balanced acc = {cv_results["test_balanced_accuracy"].mean():.3f}'
    )

    X_df = pd.DataFrame(X, columns=selected_feature_cols)
    sep_scores = {}
    for feat in selected_feature_cols:
        vals = [X_df.loc[y == k, feat].values for k in [0, 1, 2]]
        means = [v.mean() for v in vals]
        stds  = [v.std()  for v in vals]
        pooled_std = np.mean(stds) + 1e-9
        sep_scores[feat] = np.std(means) / pooled_std

    top = sorted(sep_scores, key=sep_scores.get, reverse=True)[:15]
    axes[1].barh(top[::-1], [sep_scores[f] for f in top[::-1]], color='steelblue')
    axes[1].set_title('Top-15 Features by Between/Within-class Separation')
    axes[1].set_xlabel('Between-class std / pooled within-class std')
    axes[1].grid(axis='x', alpha=0.3)

    plt.suptitle('Watch Location Classifier', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'watch_loc_results.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print('Saved: watch_loc_results.png')

    pipe.fit(X, y)
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump({
            'model': pipe,
            'feature_cols': selected_feature_cols,
            'all_feature_cols': FEATURE_COLS,
            'label': LABEL,
            'top_k_features': WATCH_TOP_K_FEATURES,
        }, f)
    print(f'Final model saved: {MODEL_OUT}')

    train_preds = pipe.predict(X)
    print(f'Train accuracy (full fit): {(train_preds == y).mean():.3f}')
