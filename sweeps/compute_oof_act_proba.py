"""
Compute out-of-fold ACTIVITY probabilities (per-recording, per-activity).

Existing `compute_oof_predictions.py` saves OOF labels (0/1). This complements
it by saving the underlying probability per activity, so the path classifier
audit can test whether feeding continuous P(walking), P(cycling) etc. as
features carries more signal than the hard 0/1 booleans.

Saves: results/oof_activity_proba.csv
Columns: filename, oof_proba_standing, oof_proba_walking, oof_proba_running, oof_proba_cycling
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import train_activities as tact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_TRAIN = os.path.join(ROOT, 'results', 'features_train.csv')
OUT_CSV = os.path.join(ROOT, 'results', 'oof_activity_proba.csv')

RANDOM_STATE = 42
N_FOLDS = 5

print('Loading features ...')
df = pd.read_csv(FEATURES_TRAIN)
n  = len(df)

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

cols = {'filename': df['filename'].values}
for activity in tact.ACTIVITIES:
    y_a       = df[activity].values.astype(int)
    feat_cols = tact.feature_cols_for(activity)
    X_act     = df[feat_cols].values.astype(float)
    proba     = np.zeros(n, dtype=float)
    for fold_idx, (tr, va) in enumerate(cv.split(X_act, y_a)):
        pipe = tact.make_pipe(activity)
        pipe.fit(X_act[tr], y_a[tr])
        proba[va] = pipe.predict_proba(X_act[va])[:, 1]
    print(f'  {activity:<10s} OOF proba range: [{proba.min():.3f}, {proba.max():.3f}]  '
          f'mean={proba.mean():.3f}')
    cols[f'oof_proba_{activity}'] = proba

out = pd.DataFrame(cols)
out.to_csv(OUT_CSV, index=False)
print(f'\nSaved OOF activity probabilities: {OUT_CSV}')
print(out.head())
