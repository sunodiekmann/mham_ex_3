"""
Per-feature discriminative power for path classification.

Three complementary metrics, all 5-fold CV on the 396 train recordings:

  1. Univariate CV balanced accuracy — fit a depth-3 GBM on JUST this single
     feature. Directly answers: "if I only had this feature, how well could I
     split the 5 paths?" Comparable apples-to-apples to the multivariate score.

  2. Mutual information with path label (sklearn `mutual_info_classif`) — picks
     up non-monotonic relationships that linear stats miss. Independent of
     classifier choice.

  3. Multivariate GBM importance — how much the full GBM model actually USES
     this feature when many features are available. Already in `audit_path_features.py`.

Univariate vs multivariate divergence is the diagnostic:

  | univariate | multivariate | interpretation                                |
  |-----------:|-------------:|-----------------------------------------------|
  | high       | high         | core feature — keep                           |
  | high       | low          | redundant with another stronger feature       |
  | low        | high         | useful only in interactions — keep but fragile|
  | low        | low          | dead weight — drop                            |

Run:  python sweeps/feature_discrimination.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import train_path as tp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_TRAIN = os.path.join(ROOT, 'results', 'features_train.csv')
OOF_CSV        = os.path.join(ROOT, 'results', 'oof_predictions.csv')

RANDOM_STATE = 42
N_FOLDS      = 5

UNIV_GBM_PARAMS = dict(
    n_estimators=80, learning_rate=0.12, max_depth=3,
    min_samples_leaf=10, subsample=0.85, random_state=RANDOM_STATE)


def categorize(feat):
    if feat.startswith('alt_rate_'):                    return 'alt-rate (new)'
    if feat.startswith('oof_proba_'):                   return 'activity proba (new)'
    if feat.startswith('phone_azim_'):                  return 'phone azimuth (new)'
    if feat.startswith('phone_'):                       return 'phone IMU'
    if feat.startswith('act_'):                         return 'activity'
    if feat.startswith('heading_seq_') or feat.startswith('heading_turnrate_'):
        return 'heading trajectory'
    if feat.startswith('heading_hist_'):                return 'heading histogram'
    if feat.startswith('heading_'):                     return 'heading aggregate'
    if feat.startswith('stair_pos_'):                   return 'stair position'
    if feat.startswith('stair_') or feat.startswith('acc_indep_stair') \
            or feat.startswith('acc_stair'):            return 'stair detection'
    if feat.startswith('pressure_') or feat.startswith('prs_'):
        return 'pressure'
    if feat.startswith('alt') or feat.startswith('altitude'):
        return 'altitude'
    if feat.startswith('acc_'):                         return 'accelerometer'
    if feat == 'duration' or feat == 'watch_loc':       return 'context'
    return 'other'


def univariate_balanced_acc(X, y, cv):
    """5-fold CV balanced accuracy for a depth-3 GBM on one feature."""
    pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('clf', GradientBoostingClassifier(**UNIV_GBM_PARAMS)),
    ])
    scores = cross_val_score(pipe, X.reshape(-1, 1), y, cv=cv,
                              scoring='balanced_accuracy', n_jobs=1)
    return float(scores.mean())


def main():
    print('Loading ...')
    df_raw, y = tp.load_data(tp.DATA_TRAIN, has_labels=True, oof_csv=OOF_CSV)
    df, feat_cols = tp._merge_csv_features(df_raw, FEATURES_TRAIN, tp.HEADING_COLS)

    # Optionally merge OOF activity probabilities so they get evaluated as
    # candidate features alongside everything else.
    proba_csv = os.path.join(ROOT, 'results', 'oof_activity_proba.csv')
    if os.path.exists(proba_csv):
        proba_df = pd.read_csv(proba_csv)
        df = df.merge(proba_df, on='filename', how='left')
        proba_cols = [c for c in proba_df.columns if c.startswith('oof_proba_')]
        feat_cols  = list(feat_cols) + proba_cols
        print(f'  + {len(proba_cols)} OOF activity-proba columns')
    else:
        print(f'  (skip) no {proba_csv} found — run sweeps/compute_oof_act_proba.py first')

    X = df[feat_cols].values.astype(float)
    print(f'  features={len(feat_cols)}  samples={len(y)}  classes={sorted(set(y))}')

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # --- Multivariate baseline + GBM importances ---
    full_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('clf', GradientBoostingClassifier(
            n_estimators=153, max_depth=3, learning_rate=0.114,
            min_samples_leaf=7, subsample=0.819, random_state=RANDOM_STATE)),
    ])
    base_scores = cross_val_score(full_pipe, X, y, cv=cv,
                                   scoring='balanced_accuracy')
    print(f'\nMultivariate baseline CV bal_acc: {base_scores.mean():.4f} ± {base_scores.std():.4f}')

    full_pipe.fit(X, y)
    multivar_imp = full_pipe['clf'].feature_importances_

    # --- Mutual information ---
    print('Computing mutual information ...')
    X_imp = SimpleImputer(strategy='median').fit_transform(X)
    mi = mutual_info_classif(X_imp, y, random_state=RANDOM_STATE)

    # --- Univariate balanced accuracy per feature ---
    print(f'Computing univariate CV balanced accuracy for {len(feat_cols)} features '
          f'(this takes a couple minutes) ...')
    univ_bal = np.zeros(len(feat_cols))
    for i, feat in enumerate(feat_cols):
        univ_bal[i] = univariate_balanced_acc(X[:, i], y, cv)
        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{len(feat_cols)}', flush=True)

    # Random-feature chance level: 5 classes balanced → 0.20
    chance = 0.20
    print(f'\nChance level (5 classes balanced): {chance:.2f}')

    # --- Assemble + sort ---
    out = pd.DataFrame({
        'feature':       feat_cols,
        'category':      [categorize(f) for f in feat_cols],
        'univ_bal_acc':  univ_bal.round(4),
        'mut_info':      mi.round(4),
        'multivar_imp':  multivar_imp.round(4),
    })
    out = out.sort_values('univ_bal_acc', ascending=False).reset_index(drop=True)

    # Save full ranking
    out_csv = os.path.join(ROOT, 'results', 'feature_discrimination.csv')
    out.to_csv(out_csv, index=False)
    print(f'\nSaved full ranking → {out_csv}')

    # --- Top-30 by univariate ---
    print('\n=== Top-30 features by univariate CV balanced accuracy ===')
    print(f'{"feature":<42} {"univ_bal":>10} {"mut_info":>10} {"mv_imp":>8}  category')
    for _, r in out.head(30).iterrows():
        print(f'{r["feature"]:<42} {r["univ_bal_acc"]:>10.4f} {r["mut_info"]:>10.4f} '
              f'{r["multivar_imp"]:>8.4f}  {r["category"]}')

    # --- Bottom features (likely dead weight) ---
    print('\n=== Bottom-15 by univariate (dead weight candidates) ===')
    bot = out.sort_values('univ_bal_acc', ascending=True).head(15)
    print(f'{"feature":<42} {"univ_bal":>10} {"mut_info":>10} {"mv_imp":>8}  category')
    for _, r in bot.iterrows():
        print(f'{r["feature"]:<42} {r["univ_bal_acc"]:>10.4f} {r["mut_info"]:>10.4f} '
              f'{r["multivar_imp"]:>8.4f}  {r["category"]}')

    # --- High-univariate but low-multivariate: redundant ---
    print('\n=== Suspicious: strong univariate but unused multivariate (redundant) ===')
    redundant = out[(out.univ_bal_acc > 0.45) & (out.multivar_imp < 0.005)]
    print(f'{"feature":<42} {"univ_bal":>10} {"mut_info":>10} {"mv_imp":>8}  category')
    for _, r in redundant.iterrows():
        print(f'{r["feature"]:<42} {r["univ_bal_acc"]:>10.4f} {r["mut_info"]:>10.4f} '
              f'{r["multivar_imp"]:>8.4f}  {r["category"]}')

    # --- Per-category summary ---
    print('\n=== Per-category summary ===')
    cat_stats = out.groupby('category').agg(
        n=('feature', 'count'),
        univ_max=('univ_bal_acc', 'max'),
        univ_mean=('univ_bal_acc', 'mean'),
        mv_total=('multivar_imp', 'sum'),
    ).round(4).sort_values('univ_max', ascending=False)
    print(cat_stats)

    # Spotlights on new feature families being trialled
    for cat in ('alt-rate (new)', 'activity proba (new)', 'phone azimuth (new)'):
        sub = out[out.category == cat]
        if len(sub):
            print(f'\n=== Spotlight: {cat} ===')
            print(f'{"feature":<42} {"univ_bal":>10} {"mut_info":>10} {"mv_imp":>8}')
            for _, r in sub.iterrows():
                print(f'{r["feature"]:<42} {r["univ_bal_acc"]:>10.4f} '
                      f'{r["mut_info"]:>10.4f} {r["multivar_imp"]:>8.4f}')


if __name__ == '__main__':
    main()
