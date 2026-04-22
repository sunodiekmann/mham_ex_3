#!/usr/bin/env python3
"""
Inference script for MHAM Exercise 3.

Loads all trained models and runs inference on train or test recordings.

Usage:
    python predict.py train              # evaluate on train set, print scores
    python predict.py test               # generate submission CSV
    python predict.py test --out my.csv  # custom output path
"""

import argparse
import os
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                              f1_score, classification_report)

# Import helpers from training scripts
from extract_features import extract_features, _heading_features
from classify_path import extract_path_features
from count_steps import count_steps_freq

# ---------------------------------------------------------------------------
TRAIN_DIR  = 'data/train'
TEST_DIR   = 'data/test'
ACTIVITIES = ['standing', 'walking', 'running', 'cycling']
ACT_INT_MAP = {0: 'standing', 1: 'walking', 2: 'running', 3: 'cycling'}
# ---------------------------------------------------------------------------


def load_models():
    print('Loading models ...')
    with open('model_watch_location.pkl', 'rb') as f:
        wl = pickle.load(f)
    with open('model_activities.pkl', 'rb') as f:
        acts = pickle.load(f)
    with open('step_count_params.pkl', 'rb') as f:
        sc = pickle.load(f)
    path = joblib.load('model_path.pkl')
    print('  watch_loc, activities, step_count, path — all loaded')
    return wl, acts, sc, path


def _normalize_activities(raw_acts):
    """Normalize activity labels from dict or list format to a bool dict."""
    if isinstance(raw_acts, dict):
        return {name: bool(raw_acts.get(name, False)) for name in ACTIVITIES}
    if isinstance(raw_acts, list):
        return {name: (i in raw_acts) for i, name in ACT_INT_MAP.items()}
    return {name: False for name in ACTIVITIES}


def _engineer_watch_loc_features(feat_df):
    """
    Compute derived features expected by the watch location model.
    These were engineered in classify_watch_location.py from CSV columns.
    """
    df = feat_df.copy()
    df['gyro_domfreq_iqr']       = df['gyro_domfreq_p75']  - df['gyro_domfreq_p25']
    df['watch_phone_gyro_ratio'] = df['gyro_std_p50']  / (df['phone_gyro_std_p50']  + 1e-9)
    df['watch_phone_acc_ratio']  = df['acc_std_p50']   / (df['phone_acc_std_p50']   + 1e-9)
    df['gyro_acc_std_ratio_p25'] = df['gyro_std_p25']  / (df['acc_std_p25'] + 1e-9)
    df['gyro_acc_std_ratio_p50'] = df['gyro_std_p50']  / (df['acc_std_p50'] + 1e-9)
    df['gyro_acc_std_ratio_p75'] = df['gyro_std_p75']  / (df['acc_std_p75'] + 1e-9)
    df['gyro_std_iqr']           = df['gyro_std_p75']  - df['gyro_std_p25']
    df['log_gyro_energy_p50']    = np.log1p(df['gyro_energy_p50'])
    df['log_acc_energy_p50']     = np.log1p(df['acc_energy_p50'])
    return df


def predict_recording(raw, wl_model, act_model, sc_data, path_model):
    """Run all four predictors on one raw recording dict."""
    data = raw['data']

    # ------------------------------------------------------------------ #
    # 1. Extract IMU features (used by watch_loc + activity classifiers)  #
    # ------------------------------------------------------------------ #
    imu_feats = extract_features(raw)
    feat_df   = pd.DataFrame([imu_feats])
    feat_df   = _engineer_watch_loc_features(feat_df)

    # ------------------------------------------------------------------ #
    # 2. Watch location                                                   #
    # ------------------------------------------------------------------ #
    wl_cols   = wl_model['feature_cols']
    watch_loc = int(wl_model['model'].predict(feat_df[wl_cols])[0])

    # ------------------------------------------------------------------ #
    # 3. Activities (watch_loc is used as a context feature)             #
    # ------------------------------------------------------------------ #
    feat_df['watch_loc'] = watch_loc
    act_cols   = act_model['feature_cols']
    activities = {}
    for act_name, pipe in act_model['models'].items():
        activities[act_name] = bool(pipe.predict(feat_df[act_cols])[0])

    # ------------------------------------------------------------------ #
    # 4. Step count (FREQ algorithm, location-specific params)           #
    # ------------------------------------------------------------------ #
    loc_entry  = sc_data['params_by_loc'].get(watch_loc,
                     sc_data['params_by_loc'][0])
    step_count = count_steps_freq(raw, loc_entry['params'])

    # ------------------------------------------------------------------ #
    # 5. Path classification                                              #
    # ------------------------------------------------------------------ #
    path_feats = extract_path_features(raw)
    # Add heading features from phone magnetometer (allowed per exercise rules)
    pmx = np.array(data['phone_mx']['values'])
    pmy = np.array(data['phone_my']['values'])
    path_feats.update(_heading_features(pmx, pmy))

    path_feat_cols = path_model['feature_cols']
    path_df = pd.DataFrame([path_feats])
    # Ensure every expected column is present (NaN if absent)
    for col in path_feat_cols:
        if col not in path_df.columns:
            path_df[col] = np.nan

    path_idx = int(path_model['model'].predict(path_df[path_feat_cols])[0])

    return {
        'watch_loc':  watch_loc,
        'path_idx':   path_idx,
        'standing':   activities.get('standing',  False),
        'walking':    activities.get('walking',   False),
        'running':    activities.get('running',   False),
        'cycling':    activities.get('cycling',   False),
        'step_count': step_count,
    }


def run_on_directory(directory, has_labels, wl_model, act_model, sc_data, path_model):
    files = sorted(f for f in os.listdir(directory) if f.endswith('.pkl'))
    predictions, labels_list = [], []

    for i, fname in enumerate(files):
        with open(os.path.join(directory, fname), 'rb') as f:
            raw = pickle.load(f)

        pred = predict_recording(raw, wl_model, act_model, sc_data, path_model)

        m = re.search(r'(\d{3})\.pkl$', fname)
        pred['Id']       = int(m.group(1)) if m else i
        pred['filename'] = fname
        predictions.append(pred)

        if has_labels and 'labels' in raw:
            lbl  = raw['labels']
            acts = _normalize_activities(lbl.get('activities', {}))
            sc   = lbl.get('step_count', -1)
            labels_list.append({
                'watch_loc':  int(lbl['watch_loc']),
                'path_idx':   int(lbl['path_idx']),
                'standing':   acts['standing'],
                'walking':    acts['walking'],
                'running':    acts['running'],
                'cycling':    acts['cycling'],
                'step_count': int(sc) if (sc is not None and sc != -1) else None,
            })

        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            print(f'  {i+1}/{len(files)}')

    preds_df  = pd.DataFrame(predictions)
    labels_df = pd.DataFrame(labels_list) if labels_list else None
    return preds_df, labels_df


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_scores(preds_df, labels_df):
    sep = '=' * 62
    print(f'\n{sep}')
    print('EVALUATION RESULTS  (train set)')
    print(sep)

    # --- Watch location ---
    yt = labels_df['watch_loc'].values
    yp = preds_df['watch_loc'].values
    print('\nWatch Location (0=wrist, 1=belt, 2=ankle):')
    print(f'  Balanced accuracy : {balanced_accuracy_score(yt, yp):.4f}')
    print(f'  Overall accuracy  : {(yt == yp).mean():.4f}')
    print('  Confusion matrix  (rows=true, cols=pred):')
    print(confusion_matrix(yt, yp))

    # --- Path index ---
    yt = labels_df['path_idx'].values
    yp = preds_df['path_idx'].values
    print('\nPath Index (0–4):')
    print(f'  Balanced accuracy : {balanced_accuracy_score(yt, yp):.4f}')
    print(f'  Overall accuracy  : {(yt == yp).mean():.4f}')
    print('  Confusion matrix  (rows=true, cols=pred):')
    print(confusion_matrix(yt, yp))

    # --- Activities ---
    print('\nActivities:')
    f1s = {}
    for act in ACTIVITIES:
        yt = labels_df[act].astype(int).values
        yp = preds_df[act].astype(int).values
        f1  = f1_score(yt, yp, zero_division=0)
        bal = balanced_accuracy_score(yt, yp)
        tp  = int(((yp == 1) & (yt == 1)).sum())
        fp  = int(((yp == 1) & (yt == 0)).sum())
        fn  = int(((yp == 0) & (yt == 1)).sum())
        f1s[act] = f1
        print(f'  {act:<10s}  F1={f1:.4f}  bal_acc={bal:.4f}'
              f'  TP={tp}  FP={fp}  FN={fn}')
    print(f'  Macro F1          : {np.mean(list(f1s.values())):.4f}')

    # --- Step count ---
    valid_mask = labels_df['step_count'].notna()
    # Also exclude zero true step counts (division by zero in per-sample MAPE)
    if valid_mask.sum() > 0:
        yt_all = labels_df.loc[valid_mask, 'step_count'].values.astype(float)
        valid_mask2 = valid_mask.copy()
        valid_mask2[valid_mask] = yt_all > 0
    else:
        valid_mask2 = valid_mask

    n_valid = int(valid_mask2.sum())
    if n_valid > 0:
        yt = labels_df.loc[valid_mask2, 'step_count'].values.astype(float)
        yp = preds_df.loc[valid_mask2.values, 'step_count'].values.astype(float)
        # WAPE (weighted / sum-based): robust to small true counts
        wape = np.sum(np.abs(yp - yt)) / np.sum(yt) * 100
        # Per-sample MAPE (clipped denominator)
        mape = np.mean(np.abs(yp - yt) / np.maximum(yt, 1)) * 100
        mae  = np.mean(np.abs(yp - yt))
        print(f'\nStep Count  (n={n_valid} labelled recordings, excl. zero counts):')
        print(f'  MAPE (per-sample) : {mape:.2f}%')
        print(f'  WAPE (sum-based)  : {wape:.2f}%')
        print(f'  MAE               : {mae:.1f} steps')

        # Per watch location
        wl_preds = preds_df.loc[valid_mask2.values, 'watch_loc'].values
        print('  Per watch location:')
        for loc, name in [(0, 'Wrist'), (1, 'Belt'), (2, 'Ankle')]:
            m = wl_preds == loc
            if m.sum() == 0:
                continue
            e = np.mean(np.abs(yp[m] - yt[m]) / np.maximum(yt[m], 1)) * 100
            print(f'    {name:<6s}  n={m.sum():2d}  MAPE={e:.1f}%')
    else:
        print('\nStep Count: no ground-truth labels available')

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='MHAM Ex3 inference')
    parser.add_argument('split', choices=['train', 'test'])
    parser.add_argument('--out', default='submission.csv',
                        help='Output CSV path (test mode only)')
    args = parser.parse_args()

    wl_model, act_model, sc_data, path_model = load_models()

    directory  = TRAIN_DIR if args.split == 'train' else TEST_DIR
    has_labels = args.split == 'train'

    print(f'\nRunning inference on {args.split} set ...')
    preds_df, labels_df = run_on_directory(
        directory, has_labels, wl_model, act_model, sc_data, path_model)

    if has_labels:
        compute_scores(preds_df, labels_df)
    else:
        cols  = ['Id', 'watch_loc', 'path_idx', 'standing', 'walking',
                 'running', 'cycling', 'step_count']
        out   = preds_df[cols].sort_values('Id').reset_index(drop=True)
        out.to_csv(args.out, index=False)
        print(f'\nSubmission written → {args.out}  ({len(out)} rows)')
        print(out.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
