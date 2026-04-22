"""
Watch-location classifier (wrist / belt / ankle).

Inspired by Kunze et al. "Where am I: Recognizing On-body Positions of
Wearable Sensors" (LoCA 2005).

Key idea from the paper
-----------------------
Use **orientation-invariant** accelerometer magnitude features extracted
during walking to distinguish body-worn sensor positions.  Each location
has a characteristic motion signature during walking:
  - Ankle : sharp heel-strike impulse  → high acc std, high gyro std
  - Wrist : arm-swing rotation         → moderate acc std, moderate gyro std
  - Belt  : hip sway only              → low acc std, low gyro std

We extend the paper by also using gyroscope magnitude features, which our
exploration showed are the strongest discriminator (Section 13 of the
exploration notebook).

Pipeline
--------
1. Select orientation-invariant watch features (acc + gyro magnitude stats).
2. Impute the single NaN that can appear in heading features (not used here).
3. StandardScaler → RandomForestClassifier (100 trees).
4. Evaluate with 5-fold stratified cross-validation.
5. Train a final model on all data and pickle it for later use.
"""

import pickle

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, balanced_accuracy_score,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEATURES_TRAIN = 'features_train.csv'
MODEL_OUT      = 'model_watch_location.pkl'
RANDOM_STATE   = 42
N_FOLDS        = 5
N_TREES        = 200

LABEL = 'watch_loc'
LOC_NAMES = {0: 'Wrist', 1: 'Belt', 2: 'Ankle'}

# ---------------------------------------------------------------------------
# Feature selection
# Orientation-invariant watch features as in Kunze et al. (acc magnitude) +
# gyroscope magnitude which our exploration confirmed as the best separator.
# We deliberately exclude phone sensors — they don't change with watch location.
# ---------------------------------------------------------------------------
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

# Derived features computed below (not in CSV directly)
DERIVED_FEATS = [
    # IQR of gyro dominant frequency: belt is noisy/irregular (high IQR),
    # ankle is ultra-regular (near-zero IQR), wrist is moderate.
    'gyro_domfreq_iqr',
    # Watch gyro std / phone gyro std: belt watch rotation ≈ phone rotation
    # (both on torso), so ratio is lowest for belt.
    'watch_phone_gyro_ratio',
    'watch_phone_acc_ratio',
    # Classical ratio features
    'gyro_acc_std_ratio_p25', 'gyro_acc_std_ratio_p50', 'gyro_acc_std_ratio_p75',
    'gyro_std_iqr',
    'log_gyro_energy_p50', 'log_acc_energy_p50',
    # acc_anisotropy: max_axis_std / min_axis_std — arm swing at wrist
    # concentrates energy in one axis (ratio ~2.1); hip sway at belt spreads
    # evenly (~1.6). Strongest new wrist/belt separator.
    'acc_anisotropy',
]

FEATURE_COLS = WATCH_ACC_FEATS + WATCH_GYRO_FEATS + DERIVED_FEATS

# ---------------------------------------------------------------------------
# Load data + engineer derived features
# ---------------------------------------------------------------------------
df = pd.read_csv(FEATURES_TRAIN)

df['gyro_domfreq_iqr']       = df['gyro_domfreq_p75']  - df['gyro_domfreq_p25']
df['watch_phone_gyro_ratio'] = df['gyro_std_p50']  / (df['phone_gyro_std_p50'] + 1e-9)
df['watch_phone_acc_ratio']  = df['acc_std_p50']   / (df['phone_acc_std_p50']  + 1e-9)
df['gyro_acc_std_ratio_p25'] = df['gyro_std_p25']  / (df['acc_std_p25'] + 1e-9)
df['gyro_acc_std_ratio_p50'] = df['gyro_std_p50']  / (df['acc_std_p50'] + 1e-9)
df['gyro_acc_std_ratio_p75'] = df['gyro_std_p75']  / (df['acc_std_p75'] + 1e-9)
df['gyro_std_iqr']           = df['gyro_std_p75']  - df['gyro_std_p25']
df['log_gyro_energy_p50']    = np.log1p(df['gyro_energy_p50'])
df['log_acc_energy_p50']     = np.log1p(df['acc_energy_p50'])
print(f'Loaded {len(df)} training recordings, {len(FEATURE_COLS)} features selected.')

X = df[FEATURE_COLS].values.astype(float)
y = df[LABEL].values.astype(int)

print(f'Class distribution: { {LOC_NAMES[k]: int((y==k).sum()) for k in [0,1,2]} }')

# ---------------------------------------------------------------------------
# Pipeline: scaler + Random Forest
# Scaling is not strictly needed for RF but makes cross-validation fair when
# comparing to linear baselines later.
# ---------------------------------------------------------------------------
pipe = Pipeline([
    ('scaler', StandardScaler()),
    # SVM-RBF outperformed RF in sweep (balanced_acc 0.920 vs 0.913).
    # C=1.43, gamma=0.026 from RandomizedSearchCV over 50 iterations.
    ('clf', SVC(kernel='rbf', C=1.43, gamma=0.026,
                class_weight='balanced', random_state=RANDOM_STATE)),
])

# ---------------------------------------------------------------------------
# 5-fold stratified cross-validation
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

cv_results = cross_validate(
    pipe, X, y, cv=cv,
    scoring=['accuracy', 'balanced_accuracy'],
    return_train_score=True,
    return_estimator=True,
)

print(f'\n=== {N_FOLDS}-fold Stratified Cross-Validation ===')
print(f'  Train accuracy      : {cv_results["train_accuracy"].mean():.3f} ± {cv_results["train_accuracy"].std():.3f}')
print(f'  Val accuracy        : {cv_results["test_accuracy"].mean():.3f} ± {cv_results["test_accuracy"].std():.3f}')
print(f'  Val balanced acc    : {cv_results["test_balanced_accuracy"].mean():.3f} ± {cv_results["test_balanced_accuracy"].std():.3f}')

# ---------------------------------------------------------------------------
# Aggregate confusion matrix over all folds
# ---------------------------------------------------------------------------
all_true, all_pred = [], []
for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    estimator = cv_results['estimator'][fold_idx]
    preds = estimator.predict(X[val_idx])
    all_true.extend(y[val_idx])
    all_pred.extend(preds)

all_true = np.array(all_true)
all_pred = np.array(all_pred)

print(f'\n=== Aggregated Classification Report (all CV folds) ===')
print(classification_report(
    all_true, all_pred,
    target_names=[LOC_NAMES[k] for k in [0, 1, 2]],
    digits=3,
))

# ---------------------------------------------------------------------------
# Per-class accuracy
# ---------------------------------------------------------------------------
print('Per-class accuracy:')
for k in [0, 1, 2]:
    mask = all_true == k
    acc_k = (all_pred[mask] == all_true[mask]).mean()
    print(f'  {LOC_NAMES[k]:6s}: {acc_k:.3f}  (n={mask.sum()})')

# ---------------------------------------------------------------------------
# Visualise confusion matrix + feature importances
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cm = confusion_matrix(all_true, all_pred, normalize='true')
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[LOC_NAMES[k] for k in [0, 1, 2]],
)
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(
    f'Confusion Matrix (normalised)\n'
    f'CV balanced acc = {cv_results["test_balanced_accuracy"].mean():.3f}'
)

# Feature separability: mean |class_mean_diff| / pooled_std per feature
X_df = pd.DataFrame(X, columns=FEATURE_COLS)
sep_scores = {}
for feat in FEATURE_COLS:
    vals = [X_df.loc[y == k, feat].values for k in [0, 1, 2]]
    means = [v.mean() for v in vals]
    stds  = [v.std()  for v in vals]
    pooled_std = np.mean(stds) + 1e-9
    sep_scores[feat] = np.std(means) / pooled_std   # between-class spread / within-class spread

top = sorted(sep_scores, key=sep_scores.get, reverse=True)[:15]
axes[1].barh(top[::-1], [sep_scores[f] for f in top[::-1]], color='steelblue')
axes[1].set_title('Top-15 Features by Between/Within-class Separation')
axes[1].set_xlabel('Between-class std / pooled within-class std')
axes[1].grid(axis='x', alpha=0.3)

plt.suptitle('Watch Location Classifier — Random Forest', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('watch_location_results.png', dpi=120, bbox_inches='tight')
plt.show()
print('Saved: watch_location_results.png')

# ---------------------------------------------------------------------------
# Final model — retrain on all data
# ---------------------------------------------------------------------------
pipe.fit(X, y)
with open(MODEL_OUT, 'wb') as f:
    pickle.dump({'model': pipe, 'feature_cols': FEATURE_COLS, 'label': LABEL}, f)
print(f'Final model saved: {MODEL_OUT}')

# Quick sanity check on training set
train_preds = pipe.predict(X)
print(f'Train accuracy (full fit): {(train_preds == y).mean():.3f}')
