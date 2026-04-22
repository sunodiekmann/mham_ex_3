"""
Path classifier for MHAM Exercise 3.

Core idea:
 - Barometric pressure is primary uphill/downhill signal (GPS is often noisy/frozen)
 - GPS altitude used when available but NaN'd if GPS appears corrupted (range < 10m)
 - Stair detection via sustained pressure derivative bursts (≥5s required to filter noise)
 - Stair position as fraction of PRESSURE progress (speed-invariant)
 - Accelerometer stair-band energy as independent stair evidence

Path descriptions:
 - Path 0: no stairs, uphill
 - Path 1: stairs early in ascent (Clausiusstrasse)
 - Path 2: stairs mid-ascent (Leonhardstrasse)
 - Path 3: reverse of path 2, downhill — stairs early in descent
 - Path 4: downhill, stairs in middle of descent

Forbidden inputs: latitude, longitude, bearing, speed, phone_steps.
"""

import os
import pickle
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

TRAIN_DIR  = 'data/train'
TEST_DIR   = 'data/test'
MODEL_PATH = 'model_path.pkl'

STAIR_THRESH_HPA_S  = 0.025   # hPa/s → rapid pressure change = stairs
MIN_STAIR_DURATION_S = 8.0    # ignore bursts shorter than this (filters 1Hz noise)
GPS_CORRUPTION_RANGE = 20.0   # m — altitude_range below this → GPS is frozen/broken
PRESSURE_TO_METRES  = -8.3    # hPa → m  (ΔP × -8.3 ≈ Δh)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _samplerate(raw_timestamps, values):
    dur = (raw_timestamps[-1][1] - raw_timestamps[0][1]) / 1000.0
    return len(values) / dur if dur > 0 else 0.0


def _net_change(arr, frac=0.1):
    n = max(1, int(len(arr) * frac))
    return float(np.mean(arr[-n:]) - np.mean(arr[:n]))


def _lp_filter(sig, sr, cutoff, order=4):
    nyq = sr / 2.0
    if cutoff >= nyq:
        return sig.copy()
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, sig)


def _sustained_mask(mask, min_samples):
    """Keep only runs of True in mask that last ≥ min_samples consecutively."""
    result = np.zeros_like(mask)
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i >= min_samples:
                result[i:j] = True
            i = j
        else:
            i += 1
    return result


# ---------------------------------------------------------------------------
# Pressure-based stair features
# ---------------------------------------------------------------------------

def _stair_features(prs, sr):
    """
    Stair detection from barometric pressure derivative.
    Requires sustained signal (≥ MIN_STAIR_DURATION_S) to suppress 1Hz noise.
    Key output: stair_pos_* = fraction of total pressure progress where stairs occur.
    """
    smooth_w = max(3, int(sr * 5))
    smooth   = uniform_filter1d(prs.astype(float), size=smooth_w)
    deriv    = np.diff(smooth) * sr          # hPa/s

    total_delta = smooth[-1] - smooth[0]
    raw_mask    = np.abs(deriv) > STAIR_THRESH_HPA_S

    # Require sustained signal to filter noise (especially at low sample rates)
    min_samp  = max(2, int(sr * MIN_STAIR_DURATION_S))
    stair_mask = _sustained_mask(raw_mask, min_samp)

    feats = {}
    feats['stair_frac']         = float(stair_mask.mean())
    feats['stair_detected']     = float(stair_mask.any())
    feats['pressure_deriv_p95'] = float(np.percentile(np.abs(deriv), 95))
    feats['pressure_deriv_p99'] = float(np.percentile(np.abs(deriv), 99))

    if stair_mask.any() and abs(total_delta) > 0.5:
        cum_prog   = (smooth[1:] - smooth[0]) / total_delta
        stair_prog = cum_prog[stair_mask]
        feats['stair_pos_first'] = float(stair_prog.min())
        feats['stair_pos_last']  = float(stair_prog.max())
        feats['stair_pos_p25']   = float(np.percentile(stair_prog, 25))
        feats['stair_pos_p50']   = float(np.percentile(stair_prog, 50))
        feats['stair_pos_p75']   = float(np.percentile(stair_prog, 75))
    else:
        for k in ('stair_pos_first', 'stair_pos_last',
                  'stair_pos_p25', 'stair_pos_p50', 'stair_pos_p75'):
            feats[k] = np.nan

    # Pressure profile shape at deciles (where in time does altitude progress happen)
    n = len(smooth)
    if abs(total_delta) > 0.5:
        for pct in (10, 20, 30, 40, 50, 60, 70, 80, 90):
            feats[f'prs_prog_{pct}'] = float((smooth[int(n * pct / 100)] - smooth[0]) / total_delta)
    else:
        for pct in (10, 20, 30, 40, 50, 60, 70, 80, 90):
            feats[f'prs_prog_{pct}'] = pct / 100.0

    # Burst ratio: stair signal is concentrated, flat-walk is uniform
    abs_deriv = np.abs(deriv)
    mean_abs  = float(abs_deriv.mean())
    feats['pressure_burst_ratio'] = float(np.percentile(abs_deriv, 99) / (mean_abs + 1e-6))
    n_top = max(1, int(len(abs_deriv) * 0.1))
    feats['pressure_top10_frac'] = float(
        np.sum(np.sort(abs_deriv)[-n_top:]) / (abs_deriv.sum() + 1e-9))

    return feats, stair_mask   # also return mask for acc analysis


# ---------------------------------------------------------------------------
# Accelerometer stair-band features
# From literature (Borenstein & Ojeda 2009; Brajdic & Harle 2013):
#   - Stair cadence ~0.7–1.3 Hz vs walking ~1.5–2.5 Hz
#   - Higher gyro std during stair climbing (body pitches with each step)
#   - More regular inter-step timing (fixed stair height)
# ---------------------------------------------------------------------------

def _acc_stair_features(data, stair_mask_prs, prs_sr):
    """
    Analyze watch acc/gyro signal during pressure-detected stair sections.
    stair_mask_prs: boolean array in pressure-time (len = len(prs)-1 after diff).
    """
    ax  = np.array(data['ax']['values'])
    ay  = np.array(data['ay']['values'])
    az  = np.array(data['az']['values'])
    gx  = np.array(data['gx']['values'])
    gy  = np.array(data['gy']['values'])
    gz  = np.array(data['gz']['values'])
    ts  = data['ax']['raw_timestamps']

    acc_sr  = _samplerate(ts, ax)
    n_acc   = min(len(ax), len(ay), len(az))
    acc_mag = np.sqrt(ax[:n_acc]**2 + ay[:n_acc]**2 + az[:n_acc]**2)
    n_gyr   = min(len(gx), len(gy), len(gz))
    gyr_mag = np.sqrt(gx[:n_gyr]**2 + gy[:n_gyr]**2 + gz[:n_gyr]**2)

    feats = {}

    # Whole-recording stair-band vs walk-band energy in LP-filtered acc
    # (independent of stair timing — useful even without pressure data)
    acc_lp = _lp_filter(acc_mag, acc_sr, cutoff=5.0)
    acc_centered = acc_lp - acc_lp.mean()
    n_fft = len(acc_centered)
    if n_fft >= 64:
        freqs   = np.fft.rfftfreq(n_fft, d=1.0 / acc_sr)
        fft_mag = np.abs(np.fft.rfft(acc_centered))
        stair_b = (freqs >= 0.6) & (freqs <= 1.4)
        walk_b  = (freqs >= 1.4) & (freqs <= 2.5)
        e_stair = float(fft_mag[stair_b].sum()) if stair_b.any() else 0.0
        e_walk  = float(fft_mag[walk_b].sum())  if walk_b.any()  else 1e-9
        feats['acc_stair_walk_ratio'] = e_stair / (e_walk + 1e-9)
    else:
        feats['acc_stair_walk_ratio'] = np.nan

    # Features specifically during pressure-detected stair sections
    if stair_mask_prs.any() and prs_sr > 0:
        # Resample stair mask from pressure time to accelerometer time
        prs_dur = len(stair_mask_prs) / prs_sr   # approximate
        acc_dur = n_acc / acc_sr
        use_dur = min(prs_dur, acc_dur)
        # Map stair mask to acc indices
        prs_time = np.linspace(0, prs_dur, len(stair_mask_prs))
        acc_time = np.linspace(0, acc_dur, n_acc)
        acc_stair_mask = np.interp(acc_time, prs_time,
                                   stair_mask_prs.astype(float)) > 0.5
        acc_stair_mask = acc_stair_mask[:n_acc]

        if acc_stair_mask.any():
            stair_acc = acc_mag[acc_stair_mask]
            flat_acc  = acc_mag[~acc_stair_mask]
            n_gyr_use = min(n_gyr, n_acc)
            stair_gyr = gyr_mag[:n_gyr_use][acc_stair_mask[:n_gyr_use]]

            feats['acc_stair_std']        = float(stair_acc.std())
            feats['acc_flat_std']         = float(flat_acc.std()) if len(flat_acc) > 10 else np.nan
            feats['acc_stair_mean']       = float(stair_acc.mean())  # ~1g always, but useful ratio
            feats['acc_stair_gyro_std']   = float(stair_gyr.std()) if len(stair_gyr) > 10 else np.nan

            # Dominant cadence during stairs: lower → more stair-like
            seg = _lp_filter(acc_mag[acc_stair_mask], acc_sr, cutoff=5.0)
            if len(seg) >= 64:
                f   = np.fft.rfftfreq(len(seg), d=1.0 / acc_sr)
                fm  = np.abs(np.fft.rfft(seg - seg.mean()))
                b   = (f >= 0.5) & (f <= 3.0)
                feats['acc_stair_cadence'] = float(f[b][np.argmax(fm[b])]) if b.any() else np.nan
            else:
                feats['acc_stair_cadence'] = np.nan
        else:
            for k in ('acc_stair_std', 'acc_flat_std', 'acc_stair_mean',
                      'acc_stair_gyro_std', 'acc_stair_cadence'):
                feats[k] = np.nan
    else:
        for k in ('acc_stair_std', 'acc_flat_std', 'acc_stair_mean',
                  'acc_stair_gyro_std', 'acc_stair_cadence'):
            feats[k] = np.nan

    return feats


# ---------------------------------------------------------------------------
# Per-recording feature extraction
# ---------------------------------------------------------------------------

def extract_path_features(raw):
    data  = raw['data']
    feats = {}

    ax_ts = data['ax']['raw_timestamps']
    feats['duration'] = (ax_ts[-1][1] - ax_ts[0][1]) / 1000.0

    # --- GPS altitude ---
    alt      = np.array(data['altitude']['values'], dtype=float)
    alt_range = float(alt.max() - alt.min())
    gps_ok   = alt_range >= GPS_CORRUPTION_RANGE   # False = GPS frozen/broken

    if gps_ok:
        feats['altitude_start']    = float(alt[0])
        feats['altitude_end']      = float(alt[-1])
        feats['altitude_net_gain'] = _net_change(alt)
        feats['altitude_range']    = alt_range
        n = len(alt)
        total_alt_delta = alt[-1] - alt[0]
        if abs(total_alt_delta) > 1:
            for q, frac in [(25, 0.25), (50, 0.5), (75, 0.75)]:
                feats[f'alt_prog_{q}'] = float((alt[int(n * frac)] - alt[0]) / total_alt_delta)
        else:
            for q in (25, 50, 75):
                feats[f'alt_prog_{q}'] = 0.5
    else:
        # GPS corrupted: NaN altitude features so model uses pressure instead
        for k in ('altitude_start', 'altitude_end', 'altitude_net_gain',
                  'alt_prog_25', 'alt_prog_50', 'alt_prog_75'):
            feats[k] = np.nan
        feats['altitude_range'] = alt_range  # keep range to signal corruption level

    # --- Barometric pressure (primary signal, more reliable than GPS) ---
    stair_mask_for_acc = np.array([], dtype=bool)
    prs_sr = 0.0

    if 'phone_pressure' in data:
        prs    = np.array(data['phone_pressure']['values'], dtype=float)
        prs_ts = data['phone_pressure']['raw_timestamps']
        prs_sr = _samplerate(prs_ts, prs)

        feats['pressure_start']     = float(prs[0])
        feats['pressure_net_delta'] = _net_change(prs)
        # Pressure-derived altitude gain: more robust than GPS
        feats['pressure_alt_gain']  = feats['pressure_net_delta'] * PRESSURE_TO_METRES

        stair_feats, stair_mask_for_acc = _stair_features(prs, prs_sr)
        feats.update(stair_feats)
    else:
        for k in ('pressure_start', 'pressure_net_delta', 'pressure_alt_gain',
                  'stair_frac', 'stair_detected',
                  'pressure_deriv_p95', 'pressure_deriv_p99',
                  'stair_pos_first', 'stair_pos_last',
                  'stair_pos_p25', 'stair_pos_p50', 'stair_pos_p75',
                  'prs_prog_10', 'prs_prog_20', 'prs_prog_30', 'prs_prog_40', 'prs_prog_50',
                  'prs_prog_60', 'prs_prog_70', 'prs_prog_80', 'prs_prog_90',
                  'pressure_local_var_p75', 'pressure_local_var_p90',
                  'pressure_burst_ratio', 'pressure_top10_frac'):
            feats[k] = np.nan

    # --- Accelerometer stair-band features ---
    feats.update(_acc_stair_features(data, stair_mask_for_acc, prs_sr))

    return feats


def load_data(directory, has_labels=True):
    files = sorted(f for f in os.listdir(directory) if f.endswith('.pkl'))
    rows, labels = [], []
    for fname in files:
        with open(os.path.join(directory, fname), 'rb') as f:
            raw = pickle.load(f)
        feats = extract_path_features(raw)
        feats['filename'] = fname
        rows.append(feats)
        if has_labels:
            labels.append(int(raw['labels']['path_idx']))
    df = pd.DataFrame(rows)
    return df, (np.array(labels) if has_labels else None)


# ---------------------------------------------------------------------------
# Feature column lists
# ---------------------------------------------------------------------------

FEATURE_COLS_RAW = [
    'duration',
    'altitude_start', 'altitude_end', 'altitude_net_gain', 'altitude_range',
    'alt_prog_25', 'alt_prog_50', 'alt_prog_75',
    'pressure_start', 'pressure_net_delta', 'pressure_alt_gain',
    'pressure_deriv_p95', 'pressure_deriv_p99',
    'stair_frac', 'stair_detected',
    'stair_pos_first', 'stair_pos_last',
    'stair_pos_p25', 'stair_pos_p50', 'stair_pos_p75',
    'prs_prog_10', 'prs_prog_20', 'prs_prog_30', 'prs_prog_40', 'prs_prog_50',
    'prs_prog_60', 'prs_prog_70', 'prs_prog_80', 'prs_prog_90',
    'pressure_burst_ratio', 'pressure_top10_frac',
    'acc_stair_walk_ratio',
    'acc_stair_std', 'acc_flat_std', 'acc_stair_mean',
    'acc_stair_gyro_std', 'acc_stair_cadence',
]

HEADING_COLS = ([f'heading_hist_{i}' for i in range(36)] +
                ['heading_mean', 'heading_std', 'heading_mode', 'heading_entropy'])


def _merge_csv_features(df_raw, csv_path, cols):
    if not os.path.exists(csv_path):
        return df_raw, [c for c in FEATURE_COLS_RAW if c in df_raw.columns]
    csv    = pd.read_csv(csv_path, usecols=['filename'] + cols)
    merged = df_raw.merge(csv, on='filename', how='left')
    return merged, FEATURE_COLS_RAW + cols


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Loading train data ...')
    df_train, y_train = load_data(TRAIN_DIR, has_labels=True)
    df_train, feat_cols = _merge_csv_features(df_train, 'features_train.csv', HEADING_COLS)
    X = df_train[feat_cols]

    print('Path distribution:', Counter(y_train.tolist()))
    df_train['path_idx'] = y_train
    print('\nKey feature means by path:')
    show = ['altitude_net_gain', 'pressure_alt_gain', 'stair_frac',
            'stair_pos_p50', 'pressure_burst_ratio', 'acc_stair_walk_ratio']
    print(df_train.groupby('path_idx')[show].mean().round(3))

    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        min_samples_leaf=4, subsample=0.8, random_state=42)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print()
    for name, model in [('GradientBoosting', gb), ('RandomForest', rf)]:
        scores = cross_val_score(
            Pipeline([('imp', SimpleImputer(strategy='median')), ('clf', model)]),
            X, y_train, cv=skf, scoring='balanced_accuracy')
        print(f'{name}: balanced_acc = {scores.mean():.3f} ± {scores.std():.3f}')

    pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('clf', gb)])
    pipe.fit(X, y_train)

    imps = sorted(zip(feat_cols, pipe['clf'].feature_importances_), key=lambda x: -x[1])
    print('\nTop feature importances (GBM):')
    for feat, val in imps[:15]:
        print(f'  {feat:35s}  {val:.4f}')

    y_cv    = cross_val_predict(pipe, X, y_train, cv=skf)
    cv_bal  = balanced_accuracy_score(y_train, y_cv)
    cm_cv   = confusion_matrix(y_train, y_cv)
    print(f'\nCV balanced_acc = {cv_bal:.3f}')
    print('CV confusion matrix (rows=true, cols=pred):')
    print(cm_cv)
    for p in range(5):
        mask = y_train == p
        print(f'  Path {p}: {mask.sum()} samples  acc={( y_cv[mask]==p).mean():.3f}')

    # --- plot: CV confusion matrix ---
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_cv, cmap='Blues')
    for r in range(5):
        for c in range(5):
            ax.text(c, r, str(cm_cv[r, c]), ha='center', va='center',
                    color='white' if cm_cv[r, c] > cm_cv.max() * 0.5 else 'black',
                    fontsize=11)
    ax.set_xticks(range(5)); ax.set_xticklabels([f'Path {i}' for i in range(5)])
    ax.set_yticks(range(5)); ax.set_yticklabels([f'Path {i}' for i in range(5)])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Path Classifier — CV Confusion Matrix\nBalanced Accuracy = {cv_bal:.3f}',
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('path_results.png', dpi=120, bbox_inches='tight')
    plt.close()

    # --- plot: feature importances ---
    top_feats = imps[:20]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([f for f, _ in top_feats][::-1], [v for _, v in top_feats][::-1])
    ax.set_xlabel('Importance')
    ax.set_title('Path Classifier — Top Feature Importances (GBM)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('path_importances.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('Saved: path_results.png, path_importances.png')

    # Save final model
    joblib.dump({'model': pipe, 'feature_cols': feat_cols}, MODEL_PATH)
    print(f'Saved → {MODEL_PATH}')

    # Test predictions
    print('\nLoading test data ...')
    df_test, _ = load_data(TEST_DIR, has_labels=False)
    df_test, _ = _merge_csv_features(df_test, 'features_test.csv', HEADING_COLS)
    y_test_pred = pipe.predict(df_test[feat_cols])
    print('Test path distribution:', Counter(y_test_pred.tolist()))
