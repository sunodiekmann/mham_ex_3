"""
Compute out-of-fold (OOF) predictions for watch_loc and activities so the
path classifier can train against predicted-quality inputs (matching what
it sees at inference) instead of ground-truth labels.

Saves: results/oof_predictions.csv
Columns: filename, oof_watch_loc, oof_standing, oof_walking, oof_running, oof_cycling
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import train_watch_loc as twl
import train_activities as tact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_TRAIN = os.path.join(ROOT, 'results', 'features_train.csv')
OUT_CSV = os.path.join(ROOT, 'results', 'oof_predictions.csv')

RANDOM_STATE = 42
N_FOLDS = 5

print('Loading features ...')
df = pd.read_csv(FEATURES_TRAIN)
n = len(df)

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# OOF watch_loc (using same Two-stage SVM+GBM as production)
# ---------------------------------------------------------------------------
print('OOF watch_loc ...')
df_wl = twl.engineer_features(df)
X_wl = df_wl[twl.FEATURE_COLS].values.astype(float)
y_wl = df_wl['watch_loc'].values.astype(int)


def make_wl_pipe():
    inner = lambda c: Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('clf',     c),
    ])
    s1 = inner(SVC(kernel='rbf', C=1.03, gamma=0.019,
                    class_weight='balanced', probability=True,
                    random_state=RANDOM_STATE))
    s2 = inner(GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=3,
        min_samples_leaf=4, subsample=0.9, random_state=RANDOM_STATE))
    return twl.TwoStageWatchLoc(stage1=s1, stage2=s2)


oof_watch_loc = np.full(n, -1, dtype=int)
for fold_idx, (tr, va) in enumerate(cv.split(X_wl, y_wl)):
    print(f'  fold {fold_idx + 1}/{N_FOLDS} ...')
    pipe = make_wl_pipe()
    pipe.fit(X_wl[tr], y_wl[tr])
    oof_watch_loc[va] = pipe.predict(X_wl[va])

acc = (oof_watch_loc == y_wl).mean()
print(f'  OOF watch_loc accuracy: {acc:.4f}')


# ---------------------------------------------------------------------------
# OOF activities (per-activity GBM, per-activity feature subset)
# ---------------------------------------------------------------------------
print('OOF activities (per-activity feature subsets) ...')

oof_acts = {}
for activity in tact.ACTIVITIES:
    y_a = df[activity].values.astype(int)
    feat_cols = tact.feature_cols_for(activity)
    X_act = df[feat_cols].values.astype(float)
    oof = np.zeros(n, dtype=int)
    for fold_idx, (tr, va) in enumerate(cv.split(X_act, y_a)):
        pipe = tact.make_pipe(activity)
        pipe.fit(X_act[tr], y_a[tr])
        oof[va] = pipe.predict(X_act[va])
    acc = (oof == y_a).mean()
    print(f'  {activity:<10s} OOF accuracy: {acc:.4f}  ({len(feat_cols)} feats)')
    oof_acts[activity] = oof


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out = pd.DataFrame({
    'filename':       df['filename'].values,
    'oof_watch_loc':  oof_watch_loc.astype(int),
    'oof_standing':   oof_acts['standing'].astype(int),
    'oof_walking':    oof_acts['walking'].astype(int),
    'oof_running':    oof_acts['running'].astype(int),
    'oof_cycling':    oof_acts['cycling'].astype(int),
})
out.to_csv(OUT_CSV, index=False)
print(f'\nSaved OOF predictions: {OUT_CSV}')
print(f'  {len(out)} rows')
print(out.head())
