"""
Multi-label activity classifier (standing / walking / running / cycling).

Each label is treated as an independent binary classification problem.
Phone sensor features are used as the primary signal because they are
location-independent (unlike the watch, which moves with watch_loc).
Watch features are also included alongside watch_loc so the model can
condition on placement.

Labels are recording-level: True means the activity occurred at some
point during the recording (not necessarily the whole time).
"""

import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                               VotingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score

FEATURES_TRAIN = 'features_train.csv'
MODEL_OUT      = 'model_activities.pkl'
RANDOM_STATE   = 42
N_FOLDS        = 5

ACTIVITIES = ['standing', 'walking', 'running', 'cycling']

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
PHONE_FEATS = [
    'phone_acc_std_p25',  'phone_acc_std_p50',  'phone_acc_std_p75',  'phone_acc_std_p90',
    'phone_acc_energy_p25','phone_acc_energy_p50','phone_acc_energy_p75',
    'phone_acc_domfreq_p25','phone_acc_domfreq_p50','phone_acc_domfreq_p75',
    'phone_gyro_std_p25', 'phone_gyro_std_p50', 'phone_gyro_std_p75',
    'phone_gyro_energy_p25','phone_gyro_energy_p50','phone_gyro_energy_p75',
    'phone_gyro_domfreq_p25','phone_gyro_domfreq_p50','phone_gyro_domfreq_p75',
    'phone_lacc_std_p25', 'phone_lacc_std_p50', 'phone_lacc_std_p75', 'phone_lacc_std_p90',
    'phone_lacc_domfreq_p25','phone_lacc_domfreq_p50','phone_lacc_domfreq_p75',
    # Cross-axis correlation (Ravi et al. 2005): walking/running translate in
    # one dimension → high |corr| on aligned axes; standing → near zero.
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
    # Watch cross-axis correlation
    'acc_corr_xy_p25','acc_corr_xy_p50','acc_corr_xy_p75',
    'acc_corr_xz_p25','acc_corr_xz_p50','acc_corr_xz_p75',
    'acc_corr_yz_p25','acc_corr_yz_p50','acc_corr_yz_p75',
]

CONTEXT_FEATS = [
    'duration',
    'altitude_range', 'altitude_net_gain',
    'acc_walk_frac',
    'acc_still_frac',   # fraction of still windows (acc_std<0.15g, mean≈1g) — standing proxy
    'temp_slope',
    'watch_loc',
]

FEATURE_COLS = PHONE_FEATS + WATCH_FEATS + CONTEXT_FEATS

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(FEATURES_TRAIN)
print(f'Loaded {len(df)} recordings, {len(FEATURE_COLS)} features.')
print('Activity distribution:')
for a in ACTIVITIES:
    print(f'  {a:<10}: {df[a].sum():3d} ({df[a].mean()*100:.1f}%)')

X = df[FEATURE_COLS].values.astype(float)
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ---------------------------------------------------------------------------
# Per-activity binary classifier
# GradientBoosting handles imbalance well and captures non-linear interactions.
# ---------------------------------------------------------------------------
def make_pipe(activity):
    """
    Plurality Voting (soft) over diverse base classifiers — Ravi et al. 2005.
    Each base estimator uses class_weight='balanced' to handle imbalance.
    Soft voting sums predicted probabilities; SVC needs probability=True.
    """
    # Three classifiers that all properly handle class imbalance via class_weight.
    # NB and kNN excluded — they ignore class_weight and hurt imbalanced labels.
    estimators = [
        ('svm', SVC(kernel='rbf', C=2.0, gamma='scale',
                    class_weight='balanced', probability=True,
                    random_state=RANDOM_STATE)),
        ('gb',  GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                    max_depth=3, subsample=0.8, min_samples_leaf=3,
                    random_state=RANDOM_STATE)),
        ('rf',  RandomForestClassifier(n_estimators=300, max_depth=15,
                    class_weight='balanced', random_state=RANDOM_STATE,
                    n_jobs=-1)),
    ]
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        # weights=[1,2,1]: double GBM weight recovers standing accuracy while
        # keeping running strong — found via grid search over weight combinations.
        ('clf', VotingClassifier(estimators=estimators, voting='soft',
                                 weights=[1, 2, 1], n_jobs=-1)),
    ])

# ---------------------------------------------------------------------------
# Cross-validation + evaluation
# ---------------------------------------------------------------------------
print(f'\n=== {N_FOLDS}-fold Stratified CV (per activity) ===')
results = {}
all_preds = {}

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for idx, activity in enumerate(ACTIVITIES):
    y = df[activity].values.astype(int)
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

    # Per-fold scores for error bars
    fold_f1 = cross_val_score(pipe, X, y, cv=cv, scoring='f1')

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
plt.savefig('activity_results.png', dpi=120, bbox_inches='tight')
plt.show()
print('\nSaved: activity_results.png')

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print('\n=== Summary ===')
print(f'  {"Activity":<12} {"Balanced Acc":>14} {"F1":>8}')
for a in ACTIVITIES:
    r = results[a]
    print(f'  {a:<12} {r["balanced_acc"]:>14.3f} {r["f1"]:>8.3f}')

# ---------------------------------------------------------------------------
# Feature importance (aggregate over activities)
# ---------------------------------------------------------------------------
print('\nFitting final models on all data...')
models = {}
for activity in ACTIVITIES:
    y = df[activity].values.astype(int)
    pipe = make_pipe(activity)
    pipe.fit(X, y)
    models[activity] = pipe

with open(MODEL_OUT, 'wb') as f:
    pickle.dump({'models': models, 'feature_cols': FEATURE_COLS,
                 'activities': ACTIVITIES}, f)
print(f'Models saved: {MODEL_OUT}')

# Feature importances from the GBM sub-estimator inside the voting ensemble
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for idx, activity in enumerate(ACTIVITIES):
    voter = models[activity].named_steps['clf']
    gb_clf = dict(voter.named_estimators_)['gb']
    imp = gb_clf.feature_importances_
    top_idx = np.argsort(imp)[-15:]
    ax = axes[idx // 2][idx % 2]
    ax.barh([FEATURE_COLS[i] for i in top_idx],
            imp[top_idx], color='steelblue')
    ax.set_title(f'{activity.capitalize()} — Top-15 Feature Importances (GBM)')
    ax.set_xlabel('Importance')
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('Activity Classifiers — Feature Importances', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('activity_importances.png', dpi=120, bbox_inches='tight')
plt.show()
print('Saved: activity_importances.png')
