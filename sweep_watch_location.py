"""
Hyperparameter sweep + feature engineering for watch-location classifier.

Improvements over baseline:
  1. Derived ratio features  (gyro_std / acc_std) — strong belt separator
  2. IQR spread features     (p75 - p25) — ankle has tighter distribution
  3. Absolute gyro features  (gyro_std_p25 already there; add log-scaled)
  4. RandomizedSearchCV comparing RF, SVM(RBF), GradientBoosting
  5. Report comparison table and final confusion matrices
"""

import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, cross_val_score
)
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix,
    ConfusionMatrixDisplay, classification_report,
)
from scipy.stats import randint, uniform, loguniform

warnings.filterwarnings('ignore')

FEATURES_TRAIN = 'features_train.csv'
RANDOM_STATE   = 42
N_FOLDS        = 5
LOC_NAMES      = {0: 'Wrist', 1: 'Belt', 2: 'Ankle'}

# ---------------------------------------------------------------------------
# Load + engineer features
# ---------------------------------------------------------------------------
df = pd.read_csv(FEATURES_TRAIN)

# --- original watch features (same as baseline) ---
BASE_FEATS = [
    'acc_mean_p25', 'acc_mean_p50', 'acc_mean_p75',
    'acc_std_p25',  'acc_std_p50',  'acc_std_p75',
    'acc_energy_p25', 'acc_energy_p50', 'acc_energy_p75',
    'acc_domfreq_p25', 'acc_domfreq_p50', 'acc_domfreq_p75',
    'acc_high_std_frac',
    'gyro_std_p25',  'gyro_std_p50',  'gyro_std_p75',
    'gyro_energy_p25', 'gyro_energy_p50', 'gyro_energy_p75',
    'gyro_domfreq_p25', 'gyro_domfreq_p50', 'gyro_domfreq_p75',
]

# --- derived features ---
# 1. Gyro/Acc ratio: separates Belt (low) from Wrist+Ankle (high)
df['gyro_acc_std_ratio_p50']  = df['gyro_std_p50']  / (df['acc_std_p50']  + 1e-9)
df['gyro_acc_std_ratio_p25']  = df['gyro_std_p25']  / (df['acc_std_p25']  + 1e-9)
df['gyro_acc_std_ratio_p75']  = df['gyro_std_p75']  / (df['acc_std_p75']  + 1e-9)
df['gyro_acc_energy_ratio']   = df['gyro_energy_p50'] / (df['acc_energy_p50'] + 1e-9)

# 2. IQR (spread across windows) — ankle heel strikes create tighter, higher peaks
df['acc_std_iqr']   = df['acc_std_p75']   - df['acc_std_p25']
df['gyro_std_iqr']  = df['gyro_std_p75']  - df['gyro_std_p25']
df['acc_mean_iqr']  = df['acc_mean_p75']  - df['acc_mean_p25']

# 3. Log-scaled energy — compresses heavy tail, helps SVM
df['log_gyro_energy_p50'] = np.log1p(df['gyro_energy_p50'])
df['log_acc_energy_p50']  = np.log1p(df['acc_energy_p50'])

DERIVED_FEATS = [
    'gyro_acc_std_ratio_p25', 'gyro_acc_std_ratio_p50', 'gyro_acc_std_ratio_p75',
    'gyro_acc_energy_ratio',
    'acc_std_iqr', 'gyro_std_iqr', 'acc_mean_iqr',
    'log_gyro_energy_p50', 'log_acc_energy_p50',
]

ALL_FEATS = BASE_FEATS + DERIVED_FEATS

y = df['watch_loc'].values.astype(int)

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def cv_balanced_acc(pipe, X, label=''):
    scores = cross_val_score(pipe, X, y, cv=cv,
                             scoring='balanced_accuracy', n_jobs=-1)
    print(f'  {label:40s}: {scores.mean():.3f} ± {scores.std():.3f}')
    return scores.mean()

print('=== Ablation: effect of derived features ===')
X_base = df[BASE_FEATS].values.astype(float)
X_all  = df[ALL_FEATS].values.astype(float)

baseline_pipe = Pipeline([
    ('sc',  StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                   class_weight='balanced',
                                   random_state=RANDOM_STATE, n_jobs=-1)),
])
cv_balanced_acc(baseline_pipe, X_base, 'RF baseline (original features)')
cv_balanced_acc(baseline_pipe, X_all,  'RF baseline (+ derived features)')

# ---------------------------------------------------------------------------
# RandomizedSearchCV — three model families
# ---------------------------------------------------------------------------
print('\n=== RandomizedSearchCV (50 iterations each, 5-fold CV) ===')

search_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE+1)
N_ITER = 50

# ---- Random Forest ----
rf_params = {
    'clf__n_estimators':      randint(100, 600),
    'clf__max_depth':         [None, 10, 15, 20, 30],
    'clf__min_samples_leaf':  randint(1, 10),
    'clf__max_features':      ['sqrt', 'log2', 0.5, 0.7],
    'clf__class_weight':      ['balanced', 'balanced_subsample'],
}
rf_pipe = Pipeline([
    ('sc',  StandardScaler()),
    ('clf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
])
rf_search = RandomizedSearchCV(
    rf_pipe, rf_params, n_iter=N_ITER, cv=search_cv,
    scoring='balanced_accuracy', n_jobs=-1,
    random_state=RANDOM_STATE, refit=True,
)
rf_search.fit(X_all, y)
print(f'  RF  best balanced_acc: {rf_search.best_score_:.3f}  params: {rf_search.best_params_}')

# ---- SVM (RBF) ----
svm_params = {
    'clf__C':     loguniform(0.1, 100),
    'clf__gamma': ['scale', 'auto', *loguniform(1e-4, 1).rvs(10, random_state=0)],
    'clf__class_weight': ['balanced', None],
}
svm_pipe = Pipeline([
    ('sc',  StandardScaler()),
    ('clf', SVC(kernel='rbf', random_state=RANDOM_STATE)),
])
svm_search = RandomizedSearchCV(
    svm_pipe, svm_params, n_iter=N_ITER, cv=search_cv,
    scoring='balanced_accuracy', n_jobs=-1,
    random_state=RANDOM_STATE, refit=True,
)
svm_search.fit(X_all, y)
print(f'  SVM best balanced_acc: {svm_search.best_score_:.3f}  params: {svm_search.best_params_}')

# ---- Gradient Boosting ----
gb_params = {
    'clf__n_estimators':   randint(100, 500),
    'clf__learning_rate':  loguniform(0.01, 0.3),
    'clf__max_depth':      randint(2, 6),
    'clf__subsample':      uniform(0.6, 0.4),
    'clf__min_samples_leaf': randint(1, 10),
}
gb_pipe = Pipeline([
    ('sc',  StandardScaler()),
    ('clf', GradientBoostingClassifier(random_state=RANDOM_STATE)),
])
gb_search = RandomizedSearchCV(
    gb_pipe, gb_params, n_iter=N_ITER, cv=search_cv,
    scoring='balanced_accuracy', n_jobs=-1,
    random_state=RANDOM_STATE, refit=True,
)
gb_search.fit(X_all, y)
print(f'  GB  best balanced_acc: {gb_search.best_score_:.3f}  params: {gb_search.best_params_}')

# ---------------------------------------------------------------------------
# Final evaluation: best model on held-out CV folds (unbiased estimate)
# ---------------------------------------------------------------------------
best_name, best_search = max(
    [('RF', rf_search), ('SVM', svm_search), ('GB', gb_search)],
    key=lambda x: x[1].best_score_
)
best_pipe = best_search.best_estimator_
print(f'\nBest model: {best_name}')

# Re-evaluate best model with fresh CV to get unbiased score
from sklearn.model_selection import cross_validate
cv_res = cross_validate(best_pipe, X_all, y, cv=cv,
                        scoring=['accuracy', 'balanced_accuracy'],
                        return_estimator=True)
print(f'  Final CV accuracy       : {cv_res["test_accuracy"].mean():.3f} ± {cv_res["test_accuracy"].std():.3f}')
print(f'  Final CV balanced_acc   : {cv_res["test_balanced_accuracy"].mean():.3f} ± {cv_res["test_balanced_accuracy"].std():.3f}')

# Aggregate confusion matrix
all_true, all_pred = [], []
for fold_i, (tr, va) in enumerate(cv.split(X_all, y)):
    est = cv_res['estimator'][fold_i]
    all_true.extend(y[va])
    all_pred.extend(est.predict(X_all[va]))
all_true, all_pred = np.array(all_true), np.array(all_pred)

print(f'\n{classification_report(all_true, all_pred, target_names=[LOC_NAMES[k] for k in [0,1,2]], digits=3)}')

# ---------------------------------------------------------------------------
# Comparison summary plot
# ---------------------------------------------------------------------------
models = {
    'RF (baseline\noriginal feats)': cross_val_score(baseline_pipe, X_base, y, cv=cv,
                                                      scoring='balanced_accuracy').mean(),
    'RF (baseline\n+ derived feats)': cross_val_score(baseline_pipe, X_all,  y, cv=cv,
                                                       scoring='balanced_accuracy').mean(),
    f'RF (swept)':  rf_search.best_score_,
    f'SVM (swept)': svm_search.best_score_,
    f'GB (swept)':  gb_search.best_score_,
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#9E9E9E', '#9E9E9E', '#2196F3', '#FF9800', '#4CAF50']
bars = axes[0].bar(list(models.keys()), list(models.values()),
                   color=colors, edgecolor='white', width=0.6)
axes[0].set_ylim(0.8, 1.0)
axes[0].axhline(0.333, color='red', ls=':', alpha=0.5, label='Random baseline')
axes[0].set_ylabel('Balanced Accuracy (5-fold CV)')
axes[0].set_title('Model Comparison')
axes[0].tick_params(axis='x', labelsize=9)
for bar, val in zip(bars, models.values()):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

cm = confusion_matrix(all_true, all_pred, normalize='true')
ConfusionMatrixDisplay(cm, display_labels=[LOC_NAMES[k] for k in [0,1,2]]).plot(
    ax=axes[1], colorbar=False, cmap='Blues')
axes[1].set_title(f'Best model ({best_name}) — normalised confusion matrix')

plt.suptitle('Watch Location: Hyperparameter Sweep Results', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('watch_location_sweep.png', dpi=120, bbox_inches='tight')
plt.show()
print('Saved: watch_location_sweep.png')

# ---------------------------------------------------------------------------
# Save best model
# ---------------------------------------------------------------------------
best_pipe.fit(X_all, y)   # refit on all data
with open('model_watch_location.pkl', 'wb') as f:
    pickle.dump({'model': best_pipe, 'feature_cols': ALL_FEATS,
                 'derived': DERIVED_FEATS, 'base': BASE_FEATS,
                 'label': 'watch_loc'}, f)
print('Updated model saved: model_watch_location.pkl')
