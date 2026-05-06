"""
Hyperparameter sweep for the path classifier.

Two sweeps:
  1. Signal preprocessing constants (STAIR_THRESH, MIN_STAIR_DURATION, GPS_CORRUPTION_RANGE)
  2. GBM model hyperparameters (fixed on best preprocessing constants)

Outputs: results/sweep_path.png
"""

import warnings
warnings.filterwarnings('ignore')

import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import train_path as tp

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'results')
FEATURES_TRAIN_CSV = os.path.join(RESULTS_DIR, 'features_train.csv')

RANDOM_STATE = 42
N_FOLDS      = 5
N_ITER       = 40

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def load_features(stair_thresh, min_stair_dur, gps_range):
    """Temporarily patch train_path constants and re-extract features."""
    tp.STAIR_THRESH_HPA_S   = stair_thresh
    tp.MIN_STAIR_DURATION_S = min_stair_dur
    tp.GPS_CORRUPTION_RANGE = gps_range

    df, y = tp.load_data(tp.DATA_TRAIN, has_labels=True)
    df, feat_cols = tp._merge_csv_features(df, FEATURES_TRAIN_CSV, tp.HEADING_COLS)
    X = df[feat_cols]
    return X, y, feat_cols


def cv_score(X, y, model=None):
    if model is None:
        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            min_samples_leaf=4, subsample=0.8, random_state=RANDOM_STATE)
    pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('clf', model)])
    scores = cross_val_score(pipe, X, y, cv=skf, scoring='balanced_accuracy')
    return scores.mean(), scores.std()


# ---------------------------------------------------------------------------
# Sweep 1: Preprocessing constants
# ---------------------------------------------------------------------------

print('=' * 60)
print('SWEEP 1: Preprocessing constants')
print('=' * 60)

STAIR_THRESHOLDS = [0.018, 0.022, 0.025, 0.030, 0.040]
STAIR_DURATIONS  = [3.0, 5.0, 8.0, 12.0]
GPS_RANGES       = [10.0, 15.0, 20.0, 30.0]

print('\n--- Stair threshold (dur=8s, gps=20m) ---')
results_thresh = []
for thresh in STAIR_THRESHOLDS:
    X, y, _ = load_features(thresh, 8.0, 20.0)
    mean, std = cv_score(X, y)
    results_thresh.append((thresh, mean, std))
    print(f'  thresh={thresh:.3f}  balanced_acc={mean:.4f} ± {std:.4f}')

print('\n--- Stair min duration (thresh=0.025, gps=20m) ---')
results_dur = []
for dur in STAIR_DURATIONS:
    X, y, _ = load_features(0.025, dur, 20.0)
    mean, std = cv_score(X, y)
    results_dur.append((dur, mean, std))
    print(f'  min_dur={dur:.1f}s  balanced_acc={mean:.4f} ± {std:.4f}')

print('\n--- GPS corruption range (thresh=0.025, dur=8s) ---')
results_gps = []
for gps in GPS_RANGES:
    X, y, _ = load_features(0.025, 8.0, gps)
    mean, std = cv_score(X, y)
    results_gps.append((gps, mean, std))
    print(f'  gps_range={gps:.0f}m  balanced_acc={mean:.4f} ± {std:.4f}')

best_thresh = max(results_thresh, key=lambda x: x[1])[0]
best_dur    = max(results_dur,    key=lambda x: x[1])[0]
best_gps    = max(results_gps,    key=lambda x: x[1])[0]
print(f'\n  Best individual: thresh={best_thresh}  dur={best_dur}  gps={best_gps}')

thresh_cands = sorted(set([best_thresh,
    STAIR_THRESHOLDS[max(0, STAIR_THRESHOLDS.index(best_thresh)-1)],
    STAIR_THRESHOLDS[min(len(STAIR_THRESHOLDS)-1, STAIR_THRESHOLDS.index(best_thresh)+1)]]))
dur_cands = sorted(set([best_dur,
    STAIR_DURATIONS[max(0, STAIR_DURATIONS.index(best_dur)-1)],
    STAIR_DURATIONS[min(len(STAIR_DURATIONS)-1, STAIR_DURATIONS.index(best_dur)+1)]]))

grid_results = []
for thresh, dur in itertools.product(thresh_cands, dur_cands):
    X, y, _ = load_features(thresh, dur, best_gps)
    mean, std = cv_score(X, y)
    grid_results.append({'thresh': thresh, 'dur': dur, 'gps': best_gps,
                         'mean': mean, 'std': std})
    print(f'  thresh={thresh:.3f}  dur={dur:.1f}  → {mean:.4f} ± {std:.4f}')

grid_df = pd.DataFrame(grid_results)
best_row          = grid_df.loc[grid_df['mean'].idxmax()]
best_thresh_final = best_row['thresh']
best_dur_final    = best_row['dur']
best_gps_final    = best_row['gps']
print(f'\n  Best combo: thresh={best_thresh_final}  dur={best_dur_final}  gps={best_gps_final}'
      f'  → {best_row["mean"]:.4f}')


# ---------------------------------------------------------------------------
# Sweep 2: GBM hyperparameters
# ---------------------------------------------------------------------------

print('\n' + '=' * 60)
print('SWEEP 2: GBM hyperparameters')
print('=' * 60)

X, y, feat_cols = load_features(best_thresh_final, best_dur_final, best_gps_final)

param_dist = {
    'clf__n_estimators':     randint(100, 600),
    'clf__max_depth':        randint(2, 6),
    'clf__learning_rate':    uniform(0.02, 0.18),
    'clf__min_samples_leaf': randint(2, 10),
    'clf__subsample':        uniform(0.6, 0.4),
}

base_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('clf', GradientBoostingClassifier(random_state=RANDOM_STATE)),
])

search = RandomizedSearchCV(
    base_pipe, param_dist,
    n_iter=N_ITER, cv=skf,
    scoring='balanced_accuracy',
    n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
)
search.fit(X, y)

print(f'\n  Best CV balanced_acc : {search.best_score_:.4f}')
print(f'  Best params          : {search.best_params_}')

rf_mean, rf_std = cv_score(X, y, RandomForestClassifier(n_estimators=300,
                                                          min_samples_leaf=3,
                                                          random_state=RANDOM_STATE))
print(f'\n  RandomForest: balanced_acc={rf_mean:.4f} ± {rf_std:.4f}')
print(f'  GBM (best):   balanced_acc={search.best_score_:.4f}')

results_cv = pd.DataFrame(search.cv_results_)
results_cv = results_cv.sort_values('mean_test_score', ascending=False)
print('\n  Top 10 GBM configs:')
show_cols = ['param_clf__n_estimators', 'param_clf__max_depth',
             'param_clf__learning_rate', 'param_clf__min_samples_leaf',
             'param_clf__subsample', 'mean_test_score', 'std_test_score']
print(results_cv[show_cols].head(10).to_string(index=False))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('Path Classifier Hyperparameter Sweep', fontsize=14, fontweight='bold')

ax = axes[0, 0]
x  = [r[0] for r in results_thresh]
y_ = [r[1] for r in results_thresh]
e  = [r[2] for r in results_thresh]
ax.errorbar(x, y_, yerr=e, marker='o', capsize=4)
ax.axvline(best_thresh_final, color='red', linestyle='--', alpha=0.7, label=f'best={best_thresh_final}')
ax.set_xlabel('Stair threshold [hPa/s]')
ax.set_ylabel('CV balanced accuracy')
ax.set_title('Stair Detection Threshold')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
x  = [r[0] for r in results_dur]
y_ = [r[1] for r in results_dur]
e  = [r[2] for r in results_dur]
ax.errorbar(x, y_, yerr=e, marker='o', capsize=4)
ax.axvline(best_dur_final, color='red', linestyle='--', alpha=0.7, label=f'best={best_dur_final}s')
ax.set_xlabel('Min stair duration [s]')
ax.set_ylabel('CV balanced accuracy')
ax.set_title('Minimum Sustained Stair Duration')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
x  = [r[0] for r in results_gps]
y_ = [r[1] for r in results_gps]
e  = [r[2] for r in results_gps]
ax.errorbar(x, y_, yerr=e, marker='o', capsize=4)
ax.axvline(best_gps_final, color='red', linestyle='--', alpha=0.7, label=f'best={best_gps_final}m')
ax.set_xlabel('GPS corruption threshold [m]')
ax.set_ylabel('CV balanced accuracy')
ax.set_title('GPS Corruption Detection Range')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
for depth in sorted(results_cv['param_clf__max_depth'].unique()):
    sub = results_cv[results_cv['param_clf__max_depth'] == depth]
    ax.scatter(sub['param_clf__n_estimators'], sub['mean_test_score'],
               label=f'depth={depth}', alpha=0.6, s=30)
ax.axhline(search.best_score_, color='red', linestyle='--',
           alpha=0.7, label=f'best={search.best_score_:.4f}')
ax.set_xlabel('n_estimators')
ax.set_ylabel('CV balanced accuracy')
ax.set_title('GBM Search (coloured by max_depth)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'sweep_path.png'), dpi=120, bbox_inches='tight')
plt.close()
print('\nSaved: results/sweep_path.png')

print('\n' + '=' * 60)
print('SUMMARY — copy best params into src/train_path.py')
print('=' * 60)
print(f'  Best preprocessing: thresh={best_thresh_final}  '
      f'dur={best_dur_final}s  gps={best_gps_final}m')
print(f'  Best GBM params:    {search.best_params_}')
print(f'  Best CV score:      {search.best_score_:.4f}')
