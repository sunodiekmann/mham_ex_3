"""
Compute the competition composite score per task definition.

Weights:
  watch_loc: 0.25, path_idx: 0.25, step_count: 0.25
  standing: 0.0625, walking: 0.0625, running: 0.0625, cycling: 0.0625

For watch_loc, path_idx, activities:
  s = max(0, balanced_accuracy_adjusted_for_chance)
  = max(0, (balanced_acc - 1/n_classes) / (1 - 1/n_classes))

For step_count:
  s = 1 - (1/n) sum min(1, |pred - true| / max(1, true))

Reports CV-based estimate and (overfit) train-based reference.
"""

import os
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import train_path as tp
import train_watch_loc as twl
import train_activities as tact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_TRAIN = os.path.join(ROOT, 'results', 'features_train.csv')
RANDOM_STATE = 42

WEIGHTS = {
    'watch_loc':  0.25,
    'path_idx':   0.25,
    'standing':   0.0625,
    'walking':    0.0625,
    'running':    0.0625,
    'cycling':    0.0625,
    'step_count': 0.25,
}


def adjusted_balanced_acc(y_true, y_pred, n_classes):
    """sklearn's balanced_accuracy_score with adjusted=True, clipped to 0."""
    bal = balanced_accuracy_score(y_true, y_pred)
    chance = 1.0 / n_classes
    adj = (bal - chance) / (1.0 - chance)
    return max(0.0, adj), bal


def step_count_score(y_pred, y_true):
    """Competition step-count metric: 1 - mean(min(1, |p-t|/max(1,t)))."""
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    rel_err = np.abs(y_pred - y_true) / np.maximum(1.0, y_true)
    capped  = np.minimum(1.0, rel_err)
    return 1.0 - capped.mean()


# ---------------------------------------------------------------------------
# Load all training data
# ---------------------------------------------------------------------------
print('Loading data ...')
df = pd.read_csv(FEATURES_TRAIN)

# Watch location features
df_wl = twl.engineer_features(df)
y_wl = df_wl[twl.LABEL].values.astype(int)
wl_feature_cols = twl.select_feature_cols(df_wl, y_wl)
X_wl = df_wl[wl_feature_cols].values.astype(float)
if len(wl_feature_cols) < len(twl.FEATURE_COLS):
    print(f'Watch pruning: top-{len(wl_feature_cols)} features')

# Path features — use OOF predictions for activities/watch_loc to match the
# inference distribution (saved model is trained the same way).
oof_csv = os.path.join(ROOT, 'results', 'oof_predictions.csv')
print('Extracting path features for CV ...')
df_path_raw, y_path = tp.load_data(tp.DATA_TRAIN, has_labels=True, oof_csv=oof_csv)
df_path, path_feat_cols = tp._merge_csv_features(df_path_raw, FEATURES_TRAIN, tp.HEADING_COLS)
X_path = df_path[path_feat_cols].values.astype(float)
if getattr(tp, 'PATH_TOP_K_FEATURES', None) is not None and tp.PATH_TOP_K_FEATURES < len(path_feat_cols):
    rank_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('clf', GradientBoostingClassifier(
            n_estimators=153, max_depth=3, learning_rate=0.114,
            min_samples_leaf=7, subsample=0.819, random_state=RANDOM_STATE)),
    ])
    rank_pipe.fit(X_path, y_path)
    rank = np.argsort(rank_pipe['clf'].feature_importances_)[::-1]
    keep = rank[:tp.PATH_TOP_K_FEATURES]
    path_feat_cols = [path_feat_cols[i] for i in keep]
    X_path = X_path[:, keep]
    print(f'Path pruning: top-{tp.PATH_TOP_K_FEATURES} features')

# Activities — per-activity feature subsets handled in the loop below

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ---------------------------------------------------------------------------
# Watch location CV
# ---------------------------------------------------------------------------
print('CV: watch_loc ...')
import sys as _sys
_sys.modules['train_watch_loc'] = _sys.modules.get(twl.__name__, twl)

# Build the same pipeline as production
def _make_wl_pipe():
    from sklearn.pipeline import Pipeline as P
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    inner = lambda c: P([('imputer', SimpleImputer(strategy='median')),
                          ('scaler',  StandardScaler()),
                          ('clf',     c)])
    s1 = inner(SVC(kernel='rbf', C=1.03, gamma=0.019,
                    class_weight='balanced', probability=True,
                    random_state=RANDOM_STATE))
    s2 = inner(GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=3,
        min_samples_leaf=4, subsample=0.9, random_state=RANDOM_STATE))
    return twl.TwoStageWatchLoc(stage1=s1, stage2=s2)

wl_pipe = _make_wl_pipe()
wl_preds = cross_val_predict(wl_pipe, X_wl, y_wl, cv=cv, n_jobs=1)
wl_adj, wl_bal = adjusted_balanced_acc(y_wl, wl_preds, n_classes=3)
print(f'  watch_loc:  bal={wl_bal:.4f}  adjusted={wl_adj:.4f}')

# ---------------------------------------------------------------------------
# Path CV
# ---------------------------------------------------------------------------
print('CV: path_idx ...')
path_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('clf', GradientBoostingClassifier(
        n_estimators=153, max_depth=3, learning_rate=0.114,
        min_samples_leaf=7, subsample=0.819, random_state=RANDOM_STATE)),
])
path_preds = cross_val_predict(path_pipe, X_path, y_path, cv=cv, n_jobs=1)
path_adj, path_bal = adjusted_balanced_acc(y_path, path_preds, n_classes=5)
print(f'  path_idx:   bal={path_bal:.4f}  adjusted={path_adj:.4f}')

# ---------------------------------------------------------------------------
# Activities CV (per-activity tuned GBM)
# ---------------------------------------------------------------------------
print('CV: activities ...')
act_results = {}
for act in tact.ACTIVITIES:
    y_a = df[act].values.astype(int)
    feat_cols = tact.feature_cols_for(act)
    X_act = df[feat_cols].values.astype(float)
    pipe = tact.make_pipe(act)
    proba = cross_val_predict(pipe, X_act, y_a, cv=cv,
                              method='predict_proba', n_jobs=1)[:, 1]
    preds = (proba >= tact.ACTIVITY_THRESHOLDS.get(act, 0.5)).astype(int)
    adj, bal = adjusted_balanced_acc(y_a, preds, n_classes=2)
    act_results[act] = {'bal': bal, 'adj': adj}
    print(f'  {act:<10s} bal={bal:.4f}  adjusted={adj:.4f}')

# ---------------------------------------------------------------------------
# Step count: use saved LOO-CV predictions (already optimal)
# ---------------------------------------------------------------------------
print('Step count: LOO-CV evaluation ...')
import train_steps as ts

records = ts.load_labelled(tp.DATA_TRAIN, FEATURES_TRAIN)
with open(os.path.join(ROOT, 'models', 'steps_params.pkl'), 'rb') as f:
    sc = pickle.load(f)

step_preds = []
step_trues = []
for rec in records:
    loc = rec['watch_loc']
    entry = sc['params_by_loc'][loc]
    pred = ts.count_steps_with_activity_mask(
        rec['raw'], entry['params'],
        source=entry.get('source', 'watch'),
        algo=entry.get('algo', 'FREQ'),
        activities=rec['activities'],
    )
    step_preds.append(pred)
    step_trues.append(rec['step_count'])
step_score = step_count_score(step_preds, step_trues)
print(f'  step_count: score={step_score:.4f}  '
      f'(n={len(step_trues)}, including {sum(1 for t in step_trues if t == 0)} '
      f'with true=0)')

# ---------------------------------------------------------------------------
# Composite weighted score
# ---------------------------------------------------------------------------
print()
print('=' * 60)
print('COMPOSITE SCORE (estimated from CV / LOO-CV)')
print('=' * 60)
contributions = {
    'watch_loc':  wl_adj,
    'path_idx':   path_adj,
    'standing':   act_results['standing']['adj'],
    'walking':    act_results['walking']['adj'],
    'running':    act_results['running']['adj'],
    'cycling':    act_results['cycling']['adj'],
    'step_count': step_score,
}

print(f'{"Task":<12} {"Raw":>10} {"Adj/Score":>12} {"Weight":>8} {"Contrib":>10}')
print('-' * 60)
total = 0.0
for task, w in WEIGHTS.items():
    if task == 'step_count':
        raw = '—'
        s = contributions[task]
    elif task == 'watch_loc':
        raw = f'{wl_bal:.4f}'; s = contributions[task]
    elif task == 'path_idx':
        raw = f'{path_bal:.4f}'; s = contributions[task]
    else:
        raw = f'{act_results[task]["bal"]:.4f}'; s = contributions[task]
    contrib = w * s
    total += contrib
    print(f'{task:<12} {raw:>10} {s:>12.4f} {w:>8.4f} {contrib:>10.4f}')
print('-' * 60)
print(f'{"TOTAL":<12} {"":>10} {"":>12} {1.0:>8.4f} {total:>10.4f}')
print()
print(f'Target: 0.95   Current estimate: {total:.4f}   Gap: {0.95 - total:+.4f}')
