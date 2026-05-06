"""
Visualize the path-classifier progress:
  - Confusion matrix comparison: baseline (0.901) vs current (0.912)
  - Per-class accuracy bars
  - Feature importance of heading vs pressure vs stair features
  - Mean heading trajectory per path with the decile-indexed points overlaid
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import train_path as tp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
FEATURES_TRAIN = os.path.join(RESULTS, 'features_train.csv')

RANDOM_STATE = 42
N_FOLDS = 5
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

PATH_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
PATH_LABELS = {
    0: 'P0\nno stairs',
    1: 'P1\nstairs early\n(Clausiusstr)',
    2: 'P2\nWalchebrücke\nstairs mid',
    3: 'P3\nreverse P2\ndownhill',
    4: 'P4\ndownhill\nstairs mid',
}

BASELINE_PER_CLASS = [0.817, 0.895, 0.878, 0.958, 0.957]
BASELINE_CV        = 0.901

# Load full feature set
print('Loading data ...')
df, y = tp.load_data(tp.DATA_TRAIN, has_labels=True)
df, feat_cols = tp._merge_csv_features(df, FEATURES_TRAIN, tp.HEADING_COLS)
X = df[feat_cols].values

# Train current production model and get CV predictions
gb = GradientBoostingClassifier(
    n_estimators=153, max_depth=3, learning_rate=0.114,
    min_samples_leaf=7, subsample=0.819, random_state=RANDOM_STATE)
pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('clf', gb)])

print('Running CV predictions ...')
preds = cross_val_predict(pipe, X, y, cv=cv)
cm_new = confusion_matrix(y, preds, labels=[0, 1, 2, 3, 4])
cv_new = balanced_accuracy_score(y, preds)
per_class_new = [cm_new[k, k] / cm_new[k].sum() for k in range(5)]
print(f'Current CV balanced_acc = {cv_new:.4f}')
print(f'Baseline CV balanced_acc = {BASELINE_CV:.4f}')

# Fit full pipeline and get feature importances
pipe.fit(X, y)
importances = pipe['clf'].feature_importances_
feat_imp = dict(zip(feat_cols, importances))

# Categorize features
def categorize(feat):
    if feat.startswith('heading_seq_') or feat.startswith('heading_turnrate_'):
        return 'heading (trajectory, new)'
    if feat.startswith('heading_'):
        return 'heading (histogram/turns)'
    if feat.startswith('stair_frac_') or 'burst' in feat or 'stair_mass' in feat \
            or 'max_deriv' in feat:
        return 'stair (multi-threshold, new)'
    if feat.startswith('stair_') or 'stair' in feat:
        return 'stair (original)'
    if feat.startswith('prs_') or 'pressure' in feat:
        return 'pressure'
    if feat.startswith('alt') or feat.startswith('altitude'):
        return 'altitude'
    if feat.startswith('acc_'):
        return 'accelerometer'
    return 'other'


cat_totals = {}
for f, v in feat_imp.items():
    c = categorize(f)
    cat_totals[c] = cat_totals.get(c, 0.0) + float(v)

# --- Figure ---
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# (a) Per-class accuracy comparison (bar chart)
ax = fig.add_subplot(gs[0, 0])
xs = np.arange(5)
w = 0.38
ax.bar(xs - w/2, BASELINE_PER_CLASS, w, label=f'baseline ({BASELINE_CV:.3f})',
        color='lightgray', edgecolor='#555')
bars = ax.bar(xs + w/2, per_class_new, w, label=f'current ({cv_new:.3f})',
               color=PATH_COLORS)
ax.set_xticks(xs)
ax.set_xticklabels([PATH_LABELS[i] for i in range(5)], fontsize=8)
ax.set_ylim(0.7, 1.02)
ax.set_ylabel('Per-class recall')
ax.set_title('Per-class accuracy: baseline vs current')
ax.legend(loc='lower right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
for b, v_old, v_new in zip(bars, BASELINE_PER_CLASS, per_class_new):
    delta = v_new - v_old
    color = '#2ca02c' if delta > 0 else ('#d62728' if delta < -0.005 else 'black')
    ax.text(b.get_x() + b.get_width()/2, v_new + 0.008,
            f'{delta:+.3f}', ha='center', fontsize=8, color=color, fontweight='bold')

# (b) Confusion matrix: current
ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(cm_new, cmap='Blues')
for r in range(5):
    for c in range(5):
        ax.text(c, r, cm_new[r, c], ha='center', va='center',
                color='white' if cm_new[r, c] > cm_new.max()*0.5 else 'black',
                fontsize=12)
ax.set_xticks(range(5)); ax.set_xticklabels([f'P{i}' for i in range(5)])
ax.set_yticks(range(5)); ax.set_yticklabels([f'P{i}' for i in range(5)])
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Current confusion matrix\nbalanced acc = {cv_new:.3f}', fontsize=11)

# (c) Feature importance by category
ax = fig.add_subplot(gs[0, 2])
cats_sorted = sorted(cat_totals.items(), key=lambda x: -x[1])
labels  = [c for c, _ in cats_sorted]
values  = [v for _, v in cats_sorted]
colors = ['#2ca02c' if '(new)' in l else '#1f77b4' for l in labels]
ax.barh(labels[::-1], values[::-1], color=colors[::-1])
ax.set_xlabel('Total GBM feature importance')
ax.set_title('Importance by feature category\n(green = new in this iteration)', fontsize=11)
ax.grid(axis='x', alpha=0.3)

# (d-f) Mean heading trajectory per path with decile-indexed points overlaid
def resample_to(x, n_target):
    idx = np.linspace(0, len(x) - 1, n_target)
    return np.interp(idx, np.arange(len(x)), x)


def circ_diff(a, b):
    return (a - b + 180) % 360 - 180


# Load raw headings for visualization
import pickle
print('Loading raw headings ...')
TARGET_LEN = 300
N_SEG = 10
heading_by_path = {k: [] for k in range(5)}
for fname in sorted(os.listdir(tp.DATA_TRAIN)):
    if not fname.endswith('.pkl'):
        continue
    with open(os.path.join(tp.DATA_TRAIN, fname), 'rb') as f:
        raw = pickle.load(f)
    pmx = np.array(raw['data']['phone_mx']['values'], dtype=float)
    pmy = np.array(raw['data']['phone_my']['values'], dtype=float)
    n = min(len(pmx), len(pmy))
    if n < 30:
        continue
    heading = np.degrees(np.arctan2(pmy[:n], pmx[:n])) % 360
    heading_by_path[int(raw['labels']['path_idx'])].append(resample_to(heading, TARGET_LEN))

# Compute decile-indexed circular means (what the feature captures)
def circular_mean_over_window(arr, lo, hi):
    seg = arr[lo:hi]
    r = np.deg2rad(seg)
    return np.degrees(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))) % 360


# Single big panel: per-path mean trajectory + segment points
ax = fig.add_subplot(gs[1, :])
for p in range(5):
    stacked = np.array(heading_by_path[p])
    if len(stacked) == 0:
        continue
    # Circular mean at each time point
    s = stacked
    rad = np.deg2rad(s)
    mean_h = np.degrees(np.arctan2(
        np.mean(np.sin(rad), axis=0),
        np.mean(np.cos(rad), axis=0))) % 360
    t = np.linspace(0, 1, TARGET_LEN)
    ax.plot(t, mean_h, color=PATH_COLORS[p], lw=2.2,
            label=f'P{p} ({len(stacked)} rec, acc={per_class_new[p]:.2f})')

    # Overlay decile-indexed segment means (the actual features)
    seg_ts = []
    seg_hs = []
    for i in range(N_SEG):
        lo = int(TARGET_LEN * i / N_SEG)
        hi = int(TARGET_LEN * (i + 1) / N_SEG)
        seg_ts.append((lo + hi) / 2 / TARGET_LEN)
        seg_hs.append(circular_mean_over_window(mean_h, lo, hi))
    ax.scatter(seg_ts, seg_hs, color=PATH_COLORS[p], s=60,
               edgecolor='white', linewidth=1.5, zorder=5)

ax.set_xlabel('Normalized walk time')
ax.set_ylabel('Heading [deg]')
ax.set_ylim(0, 360)
ax.set_yticks([0, 90, 180, 270, 360])
ax.set_yticklabels(['N (0°)', 'E (90°)', 'S (180°)', 'W (270°)', 'N (360°)'])
ax.set_title('Mean heading trajectory per path (dots = decile-indexed features, new)', fontsize=12)
ax.legend(loc='lower left', ncol=5, fontsize=9)
ax.grid(alpha=0.3)

fig.suptitle(f'Path Classifier Progress: {BASELINE_CV:.3f} → {cv_new:.3f} (+{cv_new-BASELINE_CV:+.3f})',
             fontsize=14, fontweight='bold', y=0.995)
out = os.path.join(RESULTS, 'path_progress.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
