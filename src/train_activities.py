"""
Multi-label activity classifier (standing / walking / running / cycling).

Each label is treated as an independent binary classification problem.
Phone sensor features are used as the primary signal because they are
location-independent (unlike the watch, which moves with watch_loc).
Watch features are included alongside watch_loc so the model can
condition on placement.

Labels are recording-level: True means the activity occurred for >60s
at some point during the recording.
"""

import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(ROOT, 'models')
RESULTS_DIR = os.path.join(ROOT, 'results')
FEATURES_TRAIN = os.path.join(RESULTS_DIR, 'features_train.csv')
MODEL_OUT      = os.path.join(MODELS_DIR, 'activities.pkl')

RANDOM_STATE = 42
N_FOLDS      = 5

ACTIVITIES = ['standing', 'walking', 'running', 'cycling']

PHONE_FEATS = [
    'phone_acc_std_p25',  'phone_acc_std_p50',  'phone_acc_std_p75',  'phone_acc_std_p90',
    'phone_acc_energy_p25','phone_acc_energy_p50','phone_acc_energy_p75',
    'phone_acc_domfreq_p25','phone_acc_domfreq_p50','phone_acc_domfreq_p75',
    'phone_gyro_std_p25', 'phone_gyro_std_p50', 'phone_gyro_std_p75',
    'phone_gyro_energy_p25','phone_gyro_energy_p50','phone_gyro_energy_p75',
    'phone_gyro_domfreq_p25','phone_gyro_domfreq_p50','phone_gyro_domfreq_p75',
    'phone_lacc_std_p25', 'phone_lacc_std_p50', 'phone_lacc_std_p75', 'phone_lacc_std_p90',
    'phone_lacc_domfreq_p25','phone_lacc_domfreq_p50','phone_lacc_domfreq_p75',
    'phone_acc_corr_xy_p25','phone_acc_corr_xy_p50','phone_acc_corr_xy_p75',
    'phone_acc_corr_xz_p25','phone_acc_corr_xz_p50','phone_acc_corr_xz_p75',
    'phone_acc_corr_yz_p25','phone_acc_corr_yz_p50','phone_acc_corr_yz_p75',
]

WATCH_FEATS = [
    'acc_std_p25', 'acc_std_p50', 'acc_std_p75', 'acc_std_p90',
    'acc_energy_p25', 'acc_energy_p50', 'acc_energy_p75',
    'acc_domfreq_p25', 'acc_domfreq_p50', 'acc_domfreq_p75',
    'acc_high_std_frac', 'acc_anisotropy',
    'gyro_std_p25', 'gyro_std_p50', 'gyro_std_p75',
    'gyro_energy_p25', 'gyro_energy_p50', 'gyro_energy_p75',
    'acc_corr_xy_p25','acc_corr_xy_p50','acc_corr_xy_p75',
    'acc_corr_xz_p25','acc_corr_xz_p50','acc_corr_xz_p75',
    'acc_corr_yz_p25','acc_corr_yz_p50','acc_corr_yz_p75',
]

CONTEXT_FEATS = [
    'duration',
    'altitude_range', 'altitude_net_gain',
    'acc_walk_frac',
    'acc_still_frac',
    'temp_slope',
    'watch_loc',
]

# Segment-based activity features: directly match the spec's ">=60 s
# uninterrupted" label definition. Computed from watch and phone IMU.
SEGMENT_FEATS = [
    f'seg_{sensor}_{act}_{suffix}'
    for sensor in ('watch', 'phone')
    for act in ('running', 'walking', 'standing', 'cycling')
    for suffix in ('longest_s', 'any_60s', 'frac')
]

# Aerial-phase features: running-specific physics. Acc drops to ~0 during
# the moment all feet are off the ground. Walking lacks this entirely.
AERIAL_FEATS = [
    f'aerial_{sensor}_{suffix}'
    for sensor in ('watch', 'phone')
    for suffix in ('min_p05', 'min_p25', 'below_thresh_frac', 'min_per_window_count')
]

FEATURE_COLS = PHONE_FEATS + WATCH_FEATS + CONTEXT_FEATS + SEGMENT_FEATS + AERIAL_FEATS

# Per-activity feature subsets, tuned via sweeps/per_activity_features.py.
# Different activities benefit from different feature dropouts:
#   - standing/walking/running: drop watch acc magnitude stats
#       (segment + aerial features already capture the signal more cleanly)
#   - cycling: drop ALL phone IMU stats (top features by importance, but
#       net-negative on CV — the GBM relies on spurious signals when phone
#       IMU is present; segment + watch features generalise better)
def _drop_watch_acc(f):
    return not (
        (f.startswith('acc_') and not f.startswith('acc_corr')) or
        f == 'acc_high_std_frac' or f == 'acc_anisotropy'
    )

def _drop_phone_imu(f):
    return not f.startswith('phone_')

ACTIVITY_FEATURE_FILTERS = {
    'standing': _drop_watch_acc,
    'walking':  _drop_watch_acc,
    'running':  _drop_watch_acc,
    'cycling':  _drop_phone_imu,
}


def feature_cols_for(activity):
    """Per-activity feature subset (drops features that are net-negative)."""
    keep_fn = ACTIVITY_FEATURE_FILTERS.get(activity, lambda f: True)
    return [f for f in FEATURE_COLS if keep_fn(f)]


# Per-activity GBM hyperparameters (tuned via RandomizedSearchCV in
# sweeps/audit_activity_features.py). Each activity gets a different config:
#   - standing/cycling: deeper trees, faster learning rate
#   - walking/running: shallow trees, more regularization
ACTIVITY_GBM_PARAMS = {
    'standing': dict(n_estimators=261, learning_rate=0.186, max_depth=3,
                      min_samples_leaf=5, subsample=0.758),
    'walking':  dict(n_estimators=234, learning_rate=0.059, max_depth=2,
                      min_samples_leaf=4, subsample=0.668),
    'running':  dict(n_estimators=158, learning_rate=0.054, max_depth=2,
                      min_samples_leaf=4, subsample=0.760),
    'cycling':  dict(n_estimators=251, learning_rate=0.147, max_depth=5,
                      min_samples_leaf=9, subsample=0.860),
}


def make_pipe(activity):
    """Per-activity tuned GBM (replaces SVM+GBM+RF voting ensemble)."""
    params = ACTIVITY_GBM_PARAMS[activity]
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('clf', GradientBoostingClassifier(random_state=RANDOM_STATE, **params)),
    ])


if __name__ == '__main__':
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(FEATURES_TRAIN)
    print(f'Loaded {len(df)} recordings, {len(FEATURE_COLS)} features.')
    print('Activity distribution:')
    for a in ACTIVITIES:
        print(f'  {a:<10}: {df[a].sum():3d} ({df[a].mean()*100:.1f}%)')

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f'\n=== {N_FOLDS}-fold Stratified CV (per activity) ===')
    results   = {}
    all_preds = {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for idx, activity in enumerate(ACTIVITIES):
        y    = df[activity].values.astype(int)
        feat_cols = feature_cols_for(activity)
        X = df[feat_cols].values.astype(float)
        pipe = make_pipe(activity)

        preds = cross_val_predict(pipe, X, y, cv=cv)
        bal   = balanced_accuracy_score(y, preds)
        f1    = f1_score(y, preds, zero_division=0)

        all_preds[activity] = preds
        results[activity]   = {'balanced_acc': bal, 'f1': f1}

        print(f'\n  [{activity.upper()}]  balanced_acc={bal:.3f}  f1={f1:.3f}')
        print(classification_report(y, preds,
                                     target_names=[f'no {activity}', activity],
                                     digits=3))

        ax = axes[idx // 2][idx % 2]
        tp = ((preds == 1) & (y == 1)).sum()
        fp = ((preds == 1) & (y == 0)).sum()
        fn = ((preds == 0) & (y == 1)).sum()
        tn = ((preds == 0) & (y == 0)).sum()
        cm = np.array([[tn, fp], [fn, tp]])
        im = ax.imshow(cm, cmap='Blues')
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha='center', va='center',
                        color='white' if cm[r, c] > cm.max()*0.5 else 'black', fontsize=13)
        ax.set_xticks([0, 1]); ax.set_xticklabels([f'no {activity}', activity])
        ax.set_yticks([0, 1]); ax.set_yticklabels([f'no {activity}', activity])
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'{activity.capitalize()}  bal_acc={bal:.3f}  F1={f1:.3f}')

    plt.suptitle('Activity Classifiers — CV Confusion Matrices', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'activity_results.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print('\nSaved: activity_results.png')

    print('\n=== Summary ===')
    print(f'  {"Activity":<12} {"Balanced Acc":>14} {"F1":>8}')
    for a in ACTIVITIES:
        r = results[a]
        print(f'  {a:<12} {r["balanced_acc"]:>14.3f} {r["f1"]:>8.3f}')

    print('\nFitting final models on all data...')
    models = {}
    feature_cols_per_activity = {}
    for activity in ACTIVITIES:
        y = df[activity].values.astype(int)
        feat_cols = feature_cols_for(activity)
        feature_cols_per_activity[activity] = feat_cols
        X_act = df[feat_cols].values.astype(float)
        pipe = make_pipe(activity)
        pipe.fit(X_act, y)
        models[activity] = pipe

    with open(MODEL_OUT, 'wb') as f:
        pickle.dump({'models': models,
                     'feature_cols': FEATURE_COLS,                # union (legacy)
                     'feature_cols_per_activity': feature_cols_per_activity,
                     'activities': ACTIVITIES}, f)
    print(f'Models saved: {MODEL_OUT}')

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for idx, activity in enumerate(ACTIVITIES):
        gb_clf = models[activity].named_steps['clf']
        imp    = gb_clf.feature_importances_
        feat_cols = feature_cols_per_activity[activity]
        top_idx = np.argsort(imp)[-15:]
        ax = axes[idx // 2][idx % 2]
        ax.barh([feat_cols[i] for i in top_idx], imp[top_idx], color='steelblue')
        ax.set_title(f'{activity.capitalize()} — Top-15 Feature Importances (GBM)')
        ax.set_xlabel('Importance')
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle('Activity Classifiers — Feature Importances', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'activity_importances.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print('Saved: activity_importances.png')
