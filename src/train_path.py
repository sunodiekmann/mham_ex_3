"""
Path classifier for MHAM Exercise 3.

Core idea:
 - Barometric pressure is primary uphill/downhill signal (GPS is often noisy/frozen)
 - GPS altitude used when available but NaN'd if GPS appears corrupted (range < 10m)
 - Stair detection via sustained pressure derivative bursts (>=5s required to filter noise)
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

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TRAIN  = os.path.join(ROOT, 'data', 'train')
DATA_TEST   = os.path.join(ROOT, 'data', 'test')
MODELS_DIR  = os.path.join(ROOT, 'models')
RESULTS_DIR = os.path.join(ROOT, 'results')
MODEL_PATH  = os.path.join(MODELS_DIR, 'path.pkl')

STAIR_THRESH_HPA_S   = 0.025   # hPa/s
MIN_STAIR_DURATION_S = 8.0     # ignore bursts shorter than this
GPS_CORRUPTION_RANGE = 10.0    # m — altitude_range below this → GPS is frozen/broken
PRESSURE_TO_METRES   = -8.3    # hPa → m  (ΔP × -8.3 ≈ Δh)


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
    """Keep only runs of True that last >= min_samples consecutively."""
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
    Sustained (>= MIN_STAIR_DURATION_S) bursts at threshold STAIR_THRESH_HPA_S.
    """
    smooth_w = max(3, int(sr * 5))
    smooth   = uniform_filter1d(prs.astype(float), size=smooth_w)
    deriv    = np.diff(smooth) * sr          # hPa/s

    total_delta = smooth[-1] - smooth[0]
    abs_deriv   = np.abs(deriv)

    feats = {}
    feats['pressure_deriv_p95'] = float(np.percentile(abs_deriv, 95))
    feats['pressure_deriv_p99'] = float(np.percentile(abs_deriv, 99))

    raw_mask = abs_deriv > STAIR_THRESH_HPA_S
    default_mask = _sustained_mask(raw_mask, max(2, int(sr * MIN_STAIR_DURATION_S)))
    feats['stair_frac']     = float(default_mask.mean())
    feats['stair_detected'] = float(default_mask.any())

    if default_mask.any() and abs(total_delta) > 0.5:
        cum_prog   = (smooth[1:] - smooth[0]) / total_delta
        stair_prog = cum_prog[default_mask]
        feats['stair_pos_first'] = float(stair_prog.min())
        feats['stair_pos_last']  = float(stair_prog.max())
        feats['stair_pos_p25']   = float(np.percentile(stair_prog, 25))
        feats['stair_pos_p50']   = float(np.percentile(stair_prog, 50))
        feats['stair_pos_p75']   = float(np.percentile(stair_prog, 75))
    else:
        for k in ('stair_pos_first', 'stair_pos_last',
                  'stair_pos_p25', 'stair_pos_p50', 'stair_pos_p75'):
            feats[k] = np.nan

    n = len(smooth)
    if abs(total_delta) > 0.5:
        for pct in (10, 20, 30, 40, 50, 60, 70, 80, 90):
            feats[f'prs_prog_{pct}'] = float((smooth[int(n * pct / 100)] - smooth[0]) / total_delta)
    else:
        for pct in (10, 20, 30, 40, 50, 60, 70, 80, 90):
            feats[f'prs_prog_{pct}'] = pct / 100.0

    mean_abs = float(abs_deriv.mean())
    feats['pressure_burst_ratio'] = float(np.percentile(abs_deriv, 99) / (mean_abs + 1e-6))
    n_top = max(1, int(len(abs_deriv) * 0.1))
    feats['pressure_top10_frac'] = float(
        np.sum(np.sort(abs_deriv)[-n_top:]) / (abs_deriv.sum() + 1e-9))

    return feats, default_mask


# ---------------------------------------------------------------------------
# Accelerometer stair-band features
# ---------------------------------------------------------------------------

def _acc_stair_detector(acc_mag, gyr_mag, acc_sr, gyr_sr,
                         watch_loc=None,
                         window_s=3.0, step_s=1.0,
                         stair_band=(0.6, 1.4),
                         walk_band=(1.5, 3.0)):
    """
    Independent accelerometer/gyro-based stair detector.

    Thresholds are watch-location specific. Ankle has the cleanest signal
    (heel-strike distinct on stairs); wrist is nearly useless (arm-swing
    cadence ~1 Hz overlaps with stair band); belt is intermediate.
      watch_loc 0 (wrist):  very strict — effectively skip unless signal is huge
      watch_loc 1 (belt):   moderate — torso sway visible but muddier
      watch_loc 2 (ankle):  standard — stairs produce big impacts + low cadence
      watch_loc None: use a conservative default (acts like belt).

    Returns per-window stair mask and window-start indices into the acc signal.
    """
    # Location-specific thresholds. Tuning based on stair-band vs walk-band
    # energy ratios, motion amplitude, and gyro activity expected per location.
    if watch_loc == 2:          # ankle
        motion_thresh       = 0.4
        gyro_active_thresh  = 25.0
        dominance_ratio     = 1.5
    elif watch_loc == 1:        # belt
        motion_thresh       = 0.25
        gyro_active_thresh  = 15.0
        dominance_ratio     = 2.0
    elif watch_loc == 0:        # wrist — effectively disable (arm swing overlaps)
        motion_thresh       = 1.0
        gyro_active_thresh  = 60.0
        dominance_ratio     = 3.0
    else:                       # conservative default
        motion_thresh       = 0.3
        gyro_active_thresh  = 20.0
        dominance_ratio     = 2.0

    n_acc = len(acc_mag)
    w = max(16, int(acc_sr * window_s))
    step = max(1, int(acc_sr * step_s))
    if n_acc < 2 * w:
        return np.array([], dtype=bool), np.array([], dtype=int)

    # Align gyro → acc time-base if sample rates differ
    if len(gyr_mag) > 0 and abs(gyr_sr - acc_sr) / max(gyr_sr, 1e-9) > 0.05:
        gyr_aligned = np.interp(
            np.arange(n_acc) / acc_sr,
            np.arange(len(gyr_mag)) / max(gyr_sr, 1e-9),
            gyr_mag,
        )
    else:
        gyr_aligned = gyr_mag[:n_acc] if len(gyr_mag) >= n_acc \
            else np.pad(gyr_mag, (0, n_acc - len(gyr_mag)), mode='edge')

    freqs = np.fft.rfftfreq(w, d=1.0 / acc_sr)
    stair_mask_band = (freqs >= stair_band[0]) & (freqs <= stair_band[1])
    walk_mask_band  = (freqs >= walk_band[0])  & (freqs <= walk_band[1])

    stair_windows = []
    window_starts = []
    for i in range(0, n_acc - w, step):
        seg_acc = acc_mag[i:i + w]
        seg_gyr = gyr_aligned[i:i + w]

        if seg_acc.std() < motion_thresh:
            stair_windows.append(False); window_starts.append(i); continue

        seg_c = seg_acc - seg_acc.mean()
        fft_m = np.abs(np.fft.rfft(seg_c))
        e_stair = float(fft_m[stair_mask_band].sum()) if stair_mask_band.any() else 0.0
        e_walk  = float(fft_m[walk_mask_band].sum())  if walk_mask_band.any()  else 1e-9

        # Tighter test: require the ARGMAX of FFT to land in the stair band
        # (not just that the band has energy). This excludes walking whose
        # dominant peak is at 2 Hz but which also has some 1 Hz energy.
        if fft_m.size > 1:
            peak_freq = freqs[np.argmax(fft_m[1:]) + 1]  # skip DC
            peak_in_stair = (stair_band[0] <= peak_freq <= stair_band[1])
        else:
            peak_in_stair = False

        stair_dominant = peak_in_stair and (e_stair > dominance_ratio * e_walk)
        gyro_active    = seg_gyr.std() > gyro_active_thresh

        stair_windows.append(bool(stair_dominant and gyro_active))
        window_starts.append(i)

    return np.array(stair_windows, dtype=bool), np.array(window_starts, dtype=int)


def _acc_independent_stair_features(data, stair_mask_prs, prs_sr, watch_loc=None):
    """
    Compute features from the INDEPENDENT acc-based stair detector, plus
    agreement/disagreement features with the pressure-based detector.

    This addresses the main bottleneck: ~22% false-positive rate on Path 0 and
    ~26% miss rate on Path 2 from pressure alone. An orthogonal acc signal
    should corroborate (or contradict) pressure and improve confidence.
    """
    keys = ['acc_indep_stair_frac',
            'acc_indep_stair_burst_count',
            'acc_indep_stair_longest_burst_s',
            'acc_indep_stair_pos_first', 'acc_indep_stair_pos_last',
            'acc_indep_stair_pos_mean',
            # Agreement features
            'stair_both_frac',           # corroborated: both detectors say stair
            'stair_prs_only_frac',       # only pressure: may be false positive
            'stair_acc_only_frac',       # only acc: pressure may have missed
            'stair_neither_frac',        # neither: confident non-stair
            'stair_agreement_score']     # (both + neither) / total

    ax = np.array(data['ax']['values'])
    ay = np.array(data['ay']['values'])
    az = np.array(data['az']['values'])
    gx = np.array(data['gx']['values'])
    gy = np.array(data['gy']['values'])
    gz = np.array(data['gz']['values'])

    acc_sr  = _samplerate(data['ax']['raw_timestamps'], ax)
    gyr_sr  = _samplerate(data['gx']['raw_timestamps'], gx)
    n_acc   = min(len(ax), len(ay), len(az))
    n_gyr   = min(len(gx), len(gy), len(gz))
    if n_acc < 64:
        return {k: np.nan for k in keys}

    acc_mag = np.sqrt(ax[:n_acc]**2 + ay[:n_acc]**2 + az[:n_acc]**2)
    gyr_mag = np.sqrt(gx[:n_gyr]**2 + gy[:n_gyr]**2 + gz[:n_gyr]**2)

    stair_windows, window_starts = _acc_stair_detector(
        acc_mag, gyr_mag, acc_sr, gyr_sr, watch_loc=watch_loc)
    if len(stair_windows) == 0:
        return {k: np.nan for k in keys}

    feats = {}
    feats['acc_indep_stair_frac'] = float(stair_windows.mean())

    # Burst analysis on acc stair mask
    bursts = []
    i = 0
    while i < len(stair_windows):
        if stair_windows[i]:
            j = i
            while j < len(stair_windows) and stair_windows[j]:
                j += 1
            bursts.append((i, j))
            i = j
        else:
            i += 1
    feats['acc_indep_stair_burst_count'] = float(len(bursts))
    step_s = 1.0
    feats['acc_indep_stair_longest_burst_s'] = (
        float(max((j - i) for i, j in bursts) * step_s) if bursts else 0.0
    )

    # Positional features (fraction of recording)
    if stair_windows.any():
        idx = np.where(stair_windows)[0]
        total = len(stair_windows)
        feats['acc_indep_stair_pos_first'] = float(idx[0] / total)
        feats['acc_indep_stair_pos_last']  = float(idx[-1] / total)
        feats['acc_indep_stair_pos_mean']  = float(idx.mean() / total)
    else:
        for k in ('acc_indep_stair_pos_first', 'acc_indep_stair_pos_last',
                  'acc_indep_stair_pos_mean'):
            feats[k] = np.nan

    # --- Agreement with pressure detector ---
    # Resample pressure mask onto the same per-window index as acc detector
    if len(stair_mask_prs) > 0 and prs_sr > 0:
        # Pressure-window centers in seconds: sample i of prs corresponds to
        # time i / prs_sr. Acc-detector windows start at window_starts[k] samples
        # of acc, i.e. seconds = window_starts[k] / acc_sr. Window length ~3s.
        t_acc_win = window_starts / acc_sr
        # Pressure-stair time mask at 1-sample resolution:
        t_prs = np.arange(len(stair_mask_prs)) / prs_sr
        prs_at_win = np.interp(t_acc_win, t_prs, stair_mask_prs.astype(float)) > 0.5

        both     = stair_windows & prs_at_win
        prs_only = (~stair_windows) & prs_at_win
        acc_only = stair_windows & (~prs_at_win)
        neither  = (~stair_windows) & (~prs_at_win)
        n = len(stair_windows)
        feats['stair_both_frac']      = float(both.mean())
        feats['stair_prs_only_frac']  = float(prs_only.mean())
        feats['stair_acc_only_frac']  = float(acc_only.mean())
        feats['stair_neither_frac']   = float(neither.mean())
        feats['stair_agreement_score'] = float((both.sum() + neither.sum()) / n)
    else:
        for k in ('stair_both_frac', 'stair_prs_only_frac',
                  'stair_acc_only_frac', 'stair_neither_frac',
                  'stair_agreement_score'):
            feats[k] = np.nan

    return feats


def _acc_stair_features(data, stair_mask_prs, prs_sr):
    """
    Analyze watch acc/gyro signal during pressure-detected stair sections.
    Based on Borenstein & Ojeda 2009; Brajdic & Harle 2013:
      - Stair cadence ~0.7-1.3 Hz vs walking ~1.5-2.5 Hz
      - Higher gyro std during stair climbing
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

    if stair_mask_prs.any() and prs_sr > 0:
        prs_dur = len(stair_mask_prs) / prs_sr
        acc_dur = n_acc / acc_sr
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

            feats['acc_stair_std']      = float(stair_acc.std())
            feats['acc_flat_std']       = float(flat_acc.std()) if len(flat_acc) > 10 else np.nan
            feats['acc_stair_mean']     = float(stair_acc.mean())
            feats['acc_stair_gyro_std'] = float(stair_gyr.std()) if len(stair_gyr) > 10 else np.nan

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

ACTIVITY_NAMES = ['standing', 'walking', 'running', 'cycling']


def _normalize_activities(raw_acts):
    """Handle the multiple activity-label formats (dict / list / None)."""
    if isinstance(raw_acts, dict):
        return {name: bool(raw_acts.get(name, False)) for name in ACTIVITY_NAMES}
    if isinstance(raw_acts, list):
        return {name: (i in raw_acts) for i, name in enumerate(ACTIVITY_NAMES)}
    return {name: False for name in ACTIVITY_NAMES}


def extract_path_features(raw, watch_loc=None, activities=None):
    data  = raw['data']
    feats = {}

    # watch_loc passed externally (predicted at inference) or read from labels (training)
    if watch_loc is None and 'labels' in raw:
        watch_loc = int(raw['labels'].get('watch_loc', -1))
    feats['watch_loc'] = int(watch_loc) if watch_loc is not None and watch_loc >= 0 else -1

    # Activity booleans (predicted at inference; from labels during training).
    # Strongest known constraint: "cycling only" forces Path 0 (no stairs),
    # since cycling is incompatible with stairs on Paths 1-4.
    if activities is None and 'labels' in raw:
        activities = raw['labels'].get('activities', {})
    acts = _normalize_activities(activities)
    for name in ACTIVITY_NAMES:
        feats[f'act_{name}'] = int(acts[name])
    # Hard rule feature: only cycling (no walking/running) → Path 0 (no stairs)
    feats['act_only_cycling'] = int(
        acts['cycling'] and not acts['walking'] and not acts['running']
    )

    ax_ts = data['ax']['raw_timestamps']
    feats['duration'] = (ax_ts[-1][1] - ax_ts[0][1]) / 1000.0

    # GPS altitude
    alt       = np.array(data['altitude']['values'], dtype=float)
    alt_range = float(alt.max() - alt.min())
    gps_ok    = alt_range >= GPS_CORRUPTION_RANGE

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
        for k in ('altitude_start', 'altitude_end', 'altitude_net_gain',
                  'alt_prog_25', 'alt_prog_50', 'alt_prog_75'):
            feats[k] = np.nan
        feats['altitude_range'] = alt_range

    # Barometric pressure (primary signal)
    stair_mask_for_acc = np.array([], dtype=bool)
    prs_sr = 0.0

    if 'phone_pressure' in data:
        prs    = np.array(data['phone_pressure']['values'], dtype=float)
        prs_ts = data['phone_pressure']['raw_timestamps']
        prs_sr = _samplerate(prs_ts, prs)

        feats['pressure_start']     = float(prs[0])
        feats['pressure_net_delta'] = _net_change(prs)
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

    feats.update(_acc_stair_features(data, stair_mask_for_acc, prs_sr))
    # Independent accelerometer-based stair detector + pressure/acc agreement.
    # Uses watch_loc-specific thresholds (wrist muted, ankle sensitive).
    feats.update(_acc_independent_stair_features(
        data, stair_mask_for_acc, prs_sr, watch_loc=watch_loc))

    return feats


def load_data(directory, has_labels=True, oof_csv=None):
    """
    Load and feature-extract recordings.

    If `oof_csv` is given (results/oof_predictions.csv), use the
    out-of-fold predictions for watch_loc and activities instead of the
    ground-truth labels. This makes path classifier training match the
    inference distribution (predicted-quality activities, not perfect ones).
    """
    oof = None
    if oof_csv and os.path.exists(oof_csv):
        oof = pd.read_csv(oof_csv).set_index('filename')

    files = sorted(f for f in os.listdir(directory) if f.endswith('.pkl'))
    rows, labels = [], []
    for fname in files:
        with open(os.path.join(directory, fname), 'rb') as f:
            raw = pickle.load(f)

        watch_loc_override, activities_override = None, None
        if oof is not None and fname in oof.index:
            row = oof.loc[fname]
            watch_loc_override = int(row['oof_watch_loc'])
            activities_override = {
                'standing': bool(row['oof_standing']),
                'walking':  bool(row['oof_walking']),
                'running':  bool(row['oof_running']),
                'cycling':  bool(row['oof_cycling']),
            }

        feats = extract_path_features(
            raw, watch_loc=watch_loc_override, activities=activities_override)
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
    'watch_loc',                       # true at train, predicted at inference
    # Activity bags: constrain path space. "cycling only" → Path 0 (hard rule).
    'act_standing', 'act_walking', 'act_running', 'act_cycling',
    'act_only_cycling',
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

N_HEADING_SEGMENTS = 10
HEADING_COLS = (
    [f'heading_hist_{i}' for i in range(36)] +
    ['heading_mean', 'heading_std', 'heading_mode', 'heading_entropy',
     'heading_circular_std', 'heading_num_modes',
     'heading_turns_45', 'heading_turns_90',
     'heading_total_change', 'heading_change_rate',
     'heading_first_turn_frac', 'heading_last_turn_frac',
     'heading_longest_straight_frac',
     'heading_early_mode_frac', 'heading_late_mode_frac'] +
    # Time-indexed heading trajectory: best single addition (+1pp CV)
    [f'heading_seq_{i}_sin' for i in range(N_HEADING_SEGMENTS)] +
    [f'heading_seq_{i}_cos' for i in range(N_HEADING_SEGMENTS)] +
    [f'heading_turnrate_{i}' for i in range(N_HEADING_SEGMENTS)] +
    ['heading_seq_mean_jump', 'heading_seq_max_jump', 'heading_seq_start_vs_end']
)

# Altitude-indexed heading and turn-sequence features were tried and
# reverted — net CV impact was within noise and they shifted errors
# between classes rather than fixing them. See git history for details.
ALT_HEADING_COLS = []


def _merge_csv_features(df_raw, csv_path, cols):
    # ALT_HEADING_COLS are computed by extract_path_features and live in df_raw.
    alt_cols = [c for c in ALT_HEADING_COLS if c in df_raw.columns]
    if not os.path.exists(csv_path):
        return df_raw, [c for c in FEATURE_COLS_RAW + alt_cols if c in df_raw.columns]
    csv    = pd.read_csv(csv_path, usecols=['filename'] + cols)
    merged = df_raw.merge(csv, on='filename', how='left')
    return merged, FEATURE_COLS_RAW + cols + alt_cols


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    features_train_csv = os.path.join(RESULTS_DIR, 'features_train.csv')
    features_test_csv  = os.path.join(RESULTS_DIR, 'features_test.csv')

    print('Loading train data ...')
    # Use OOF predictions for watch_loc/activities (matches inference dist)
    oof_csv = os.path.join(RESULTS_DIR, 'oof_predictions.csv')
    df_train, y_train = load_data(DATA_TRAIN, has_labels=True, oof_csv=oof_csv)
    df_train, feat_cols = _merge_csv_features(df_train, features_train_csv, HEADING_COLS)
    X = df_train[feat_cols]

    print('Path distribution:', Counter(y_train.tolist()))
    df_train['path_idx'] = y_train
    print('\nKey feature means by path:')
    show = ['altitude_net_gain', 'pressure_alt_gain', 'stair_frac',
            'stair_pos_p50', 'pressure_burst_ratio', 'acc_stair_walk_ratio']
    print(df_train.groupby('path_idx')[show].mean().round(3))

    # Best empirically on 199-feature set: shallower GBM with higher lr.
    # Retunes tried min_leaf=10 / lr=0.038 but those hurt Path 3 (6 confused
    # with Path 4). These params keep Path 3 intact at 0.93+.
    gb = GradientBoostingClassifier(
        n_estimators=153, max_depth=3, learning_rate=0.114,
        min_samples_leaf=7, subsample=0.819, random_state=42)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42)

    def build_pipe(clf):
        return Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('clf', clf),
        ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print()
    for name, model in [('GradientBoosting', gb), ('RandomForest', rf)]:
        scores = cross_val_score(build_pipe(model), X, y_train,
                                  cv=skf, scoring='balanced_accuracy')
        print(f'{name}: balanced_acc = {scores.mean():.3f} ± {scores.std():.3f}')

    pipe = build_pipe(gb)
    pipe.fit(X, y_train)

    imps = sorted(zip(feat_cols, pipe['clf'].feature_importances_), key=lambda x: -x[1])
    print('\nTop feature importances (GBM):')
    for feat, val in imps[:15]:
        print(f'  {feat:35s}  {val:.4f}')

    y_cv   = cross_val_predict(pipe, X, y_train, cv=skf)
    cv_bal = balanced_accuracy_score(y_train, y_cv)
    cm_cv  = confusion_matrix(y_train, y_cv)
    print(f'\nCV balanced_acc = {cv_bal:.3f}')
    print('CV confusion matrix (rows=true, cols=pred):')
    print(cm_cv)
    for p in range(5):
        mask = y_train == p
        print(f'  Path {p}: {mask.sum()} samples  acc={(y_cv[mask]==p).mean():.3f}')

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
    plt.savefig(os.path.join(RESULTS_DIR, 'path_results.png'), dpi=120, bbox_inches='tight')
    plt.close()

    top_feats = imps[:20]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([f for f, _ in top_feats][::-1], [v for _, v in top_feats][::-1])
    ax.set_xlabel('Importance')
    ax.set_title('Path Classifier — Top Feature Importances (GBM)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'path_importances.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print('Saved: path_results.png, path_importances.png')

    joblib.dump({'model': pipe, 'feature_cols': feat_cols}, MODEL_PATH)
    print(f'Saved → {MODEL_PATH}')

    print('\nLoading test data ...')
    df_test, _ = load_data(DATA_TEST, has_labels=False)
    df_test, _ = _merge_csv_features(df_test, features_test_csv, HEADING_COLS)
    y_test_pred = pipe.predict(df_test[feat_cols])
    print('Test path distribution:', Counter(y_test_pred.tolist()))
