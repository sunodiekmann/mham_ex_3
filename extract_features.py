"""
Feature extraction for MHAM Exercise 3.

Produces features_train.csv and features_test.csv.
Each row = one recording. No labels in the test CSV.

Feature groups:
  - duration                   : recording length [s]
  - acc_{mean/std/energy/domfreq}_{p25/p50/p75}  : watch acc magnitude, windowed
  - gyro_{std/energy/domfreq}_{p25/p50/p75}       : watch gyro magnitude, windowed
  - phone_acc_{std/energy/domfreq}_{p25/p50/p75}  : phone acc magnitude, windowed
  - phone_gyro_{std/energy/domfreq}_{p25/p50/p75} : phone gyro magnitude, windowed
  - phone_lacc_{std/domfreq}_{p25/p50/p75}        : phone linear acc (gravity removed), windowed
  - acc_high_std_frac          : fraction of windows with acc_std > 0.5 g  (running proxy)
  - altitude_net_gain          : elevation change end-start [m]
  - altitude_range             : max-min altitude [m]
  - pressure_net_delta         : pressure change end-start [hPa], NaN if absent
  - heading_{mean/std/mode/entropy}               : phone magnetometer heading statistics
  - heading_hist_{0..35}       : 36 × 10° normalised histogram of phone heading
"""

import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_DIR = 'data/train'
TEST_DIR  = 'data/test'

# ---------------------------------------------------------------------------
# Window parameters
# ---------------------------------------------------------------------------
WINDOW_S     = 5.0
OVERLAP      = 0.5
HEADING_BINS = 36
HIGH_STD_THRESH = 0.5   # g — threshold for "high-impact" window (running proxy)

ACT_INT_MAP = {0: 'standing', 1: 'walking', 2: 'running', 3: 'cycling'}

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _mag(x, y, z):
    n = min(len(x), len(y), len(z))
    return np.sqrt(x[:n]**2 + y[:n]**2 + z[:n]**2)


def _samplerate(raw_timestamps, n_values):
    duration = (raw_timestamps[-1][1] - raw_timestamps[0][1]) / 1000.0
    return n_values / duration if duration > 0 else 0.0


def _windowed(sig, sr):
    """
    Slide a WINDOW_S window with OVERLAP over sig.
    Returns (means, stds, energies, dom_freqs, spec_entropies) as numpy arrays.
    spec_entropies: Shannon entropy of the normalised FFT magnitude in 0.5–5 Hz —
    low = peaked spectrum (ankle heel-strike); high = flat/noisy (belt standing).
    """
    w    = int(WINDOW_S * sr)
    step = int(w * (1 - OVERLAP))
    if w < 8 or len(sig) < w:
        empty = np.array([np.nan])
        return empty, empty, empty, empty, empty

    means, stds, energies, dom_freqs, spec_entropies = [], [], [], [], []
    for start in range(0, len(sig) - w, step):
        seg = sig[start:start + w]
        means.append(np.mean(seg))
        stds.append(np.std(seg))
        freqs   = np.fft.rfftfreq(len(seg), d=1.0 / sr)
        fft_mag = np.abs(np.fft.rfft(seg))
        energies.append(np.sum(fft_mag**2) / len(seg))
        broad = (freqs >= 0.3) & (freqs <= 10.0)
        dom_freqs.append(freqs[broad][np.argmax(fft_mag[broad])] if broad.any() else 0.0)
        walk  = (freqs >= 0.5) & (freqs <= 5.0)
        if walk.any():
            p = fft_mag[walk]; p = p / (p.sum() + 1e-9)
            spec_entropies.append(float(scipy_entropy(p + 1e-9)))
        else:
            spec_entropies.append(np.nan)

    return (np.array(means), np.array(stds),
            np.array(energies), np.array(dom_freqs), np.array(spec_entropies))


def _percentiles(arr, prefix):
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return {f'{prefix}_p25': np.nan, f'{prefix}_p50': np.nan,
                f'{prefix}_p75': np.nan, f'{prefix}_p90': np.nan}
    return {
        f'{prefix}_p25': float(np.nanpercentile(arr, 25)),
        f'{prefix}_p50': float(np.nanpercentile(arr, 50)),
        f'{prefix}_p75': float(np.nanpercentile(arr, 75)),
        f'{prefix}_p90': float(np.nanpercentile(arr, 90)),
    }


def _windowed_corr(x, y, sr):
    """
    Per-window Pearson correlation between two axis signals (Ravi et al. 2005).
    Walking/running translate in ~one direction → high abs correlation.
    Standing: near-zero. Cycling: rotational pattern → moderate.
    """
    w    = int(WINDOW_S * sr)
    step = int(w * (1 - OVERLAP))
    n    = min(len(x), len(y))
    if w < 8 or n < w:
        return np.array([np.nan])
    corrs = []
    for start in range(0, n - w, step):
        sx, sy = x[start:start+w], y[start:start+w]
        sx_std, sy_std = sx.std(), sy.std()
        if sx_std > 1e-9 and sy_std > 1e-9:
            corrs.append(float(np.corrcoef(sx, sy)[0, 1]))
        else:
            corrs.append(0.0)
    return np.array(corrs)


def _net_change(vals, frac=0.1):
    """Mean of last `frac` fraction minus mean of first `frac` fraction."""
    n = max(1, int(len(vals) * frac))
    return float(np.mean(vals[-n:]) - np.mean(vals[:n]))


def _heading_features(mx, my):
    mx = np.ravel(mx).astype(float)
    my = np.ravel(my).astype(float)
    n  = min(len(mx), len(my))
    if n == 0:
        nan_feats = {f'heading_hist_{i}': np.nan for i in range(HEADING_BINS)}
        nan_feats.update({'heading_mean': np.nan, 'heading_std': np.nan,
                          'heading_mode': np.nan, 'heading_entropy': np.nan})
        return nan_feats
    heading = np.degrees(np.arctan2(my[:n], mx[:n])) % 360
    bins    = np.linspace(0, 360, HEADING_BINS + 1)
    hist, _ = np.histogram(heading, bins=bins)
    hist_n  = hist.astype(float) / (hist.sum() + 1e-9)

    feats = {f'heading_hist_{i}': hist_n[i] for i in range(HEADING_BINS)}
    feats['heading_mean']    = float(np.mean(heading))
    feats['heading_std']     = float(np.std(heading))
    feats['heading_mode']    = float(bins[np.argmax(hist)])
    feats['heading_entropy'] = float(scipy_entropy(hist_n + 1e-9))
    return feats


# ---------------------------------------------------------------------------
# Per-recording feature extraction
# ---------------------------------------------------------------------------

def extract_features(raw):
    """
    raw : dict loaded from a .pkl file  (keys: 'data', 'labels')
    Returns a flat feature dict.
    """
    data  = raw['data']
    feats = {}

    # ---- duration ----------------------------------------------------------
    ax_ts = data['ax']['raw_timestamps']
    duration = (ax_ts[-1][1] - ax_ts[0][1]) / 1000.0
    feats['duration'] = duration

    # ---- watch accelerometer -----------------------------------------------
    ax = np.array(data['ax']['values'])
    ay = np.array(data['ay']['values'])
    az = np.array(data['az']['values'])
    acc_sr  = _samplerate(ax_ts, len(ax))
    acc_mag = _mag(ax, ay, az)

    mn, sd, en, df, se = _windowed(acc_mag, acc_sr)
    feats.update(_percentiles(mn, 'acc_mean'))
    feats.update(_percentiles(sd,  'acc_std'))
    feats.update(_percentiles(en,  'acc_energy'))
    feats.update(_percentiles(df,  'acc_domfreq'))
    feats['acc_high_std_frac'] = float(np.nanmean(sd > HIGH_STD_THRESH))
    # Fraction of still windows: near-gravity mean (≈1g) + near-zero std.
    # Direct proxy for standing — even if mixed with walking in the recording,
    # 60+ seconds of standing creates enough still windows to be detectable.
    still_mask = (sd < 0.15) & (np.abs(mn - 1.0) < 0.15)
    feats['acc_still_frac'] = float(still_mask.mean()) if len(still_mask) > 0 else np.nan

    # Walking-selective features (Kunze et al. insight): only windows where
    # dominant freq is in the walking cadence band and motion is strong enough.
    WALK_MASK = (df >= 1.0) & (df <= 3.0) & (sd > 0.3)
    feats.update(_percentiles(sd[WALK_MASK],  'acc_walk_std'))
    feats.update(_percentiles(se[WALK_MASK],  'acc_walk_spec_entropy'))
    feats['acc_walk_frac'] = float(WALK_MASK.mean()) if len(WALK_MASK) > 0 else np.nan

    # Acc anisotropy: sort per-axis stds and take max/min ratio.
    # Arm swing (wrist) concentrates motion in one axis → high ratio.
    # Hip sway (belt) spreads evenly → low ratio.  Orientation-invariant.
    n3 = min(len(ax), len(ay), len(az))
    axis_stds = sorted([float(ax[:n3].std()), float(ay[:n3].std()), float(az[:n3].std())])
    feats['acc_anisotropy'] = axis_stds[2] / (axis_stds[0] + 1e-9)

    # Cross-axis correlation (Ravi et al. 2005): walking/running translate in
    # one dimension → high |corr| between aligned axes; standing → near zero.
    feats.update(_percentiles(_windowed_corr(ax, ay, acc_sr), 'acc_corr_xy'))
    feats.update(_percentiles(_windowed_corr(ax, az, acc_sr), 'acc_corr_xz'))
    feats.update(_percentiles(_windowed_corr(ay, az, acc_sr), 'acc_corr_yz'))

    # ---- watch gyroscope ---------------------------------------------------
    gx = np.array(data['gx']['values'])
    gy = np.array(data['gy']['values'])
    gz = np.array(data['gz']['values'])
    gyro_sr  = _samplerate(data['gx']['raw_timestamps'], len(gx))
    gyro_mag = _mag(gx, gy, gz)

    _, gsd, gen, gdf, gse = _windowed(gyro_mag, gyro_sr)
    feats.update(_percentiles(gsd, 'gyro_std'))
    feats.update(_percentiles(gen, 'gyro_energy'))
    feats.update(_percentiles(gdf, 'gyro_domfreq'))
    feats.update(_percentiles(gsd[WALK_MASK],  'gyro_walk_std'))
    feats.update(_percentiles(gse[WALK_MASK],  'gyro_walk_spec_entropy'))

    # ---- phone accelerometer -----------------------------------------------
    pax = np.array(data['phone_ax']['values'])
    pay = np.array(data['phone_ay']['values'])
    paz = np.array(data['phone_az']['values'])
    pacc_sr  = _samplerate(data['phone_ax']['raw_timestamps'], len(pax))
    pacc_mag = _mag(pax, pay, paz)

    _, psd, pen, pdf, _ = _windowed(pacc_mag, pacc_sr)
    feats.update(_percentiles(psd, 'phone_acc_std'))
    feats.update(_percentiles(pen, 'phone_acc_energy'))
    feats.update(_percentiles(pdf, 'phone_acc_domfreq'))
    feats.update(_percentiles(_windowed_corr(pax, pay, pacc_sr), 'phone_acc_corr_xy'))
    feats.update(_percentiles(_windowed_corr(pax, paz, pacc_sr), 'phone_acc_corr_xz'))
    feats.update(_percentiles(_windowed_corr(pay, paz, pacc_sr), 'phone_acc_corr_yz'))

    # ---- phone gyroscope ---------------------------------------------------
    pgx = np.array(data['phone_gx']['values'])
    pgy = np.array(data['phone_gy']['values'])
    pgz = np.array(data['phone_gz']['values'])
    pgyro_sr  = _samplerate(data['phone_gx']['raw_timestamps'], len(pgx))
    pgyro_mag = _mag(pgx, pgy, pgz)

    _, pgsd, pgen, pgdf, _ = _windowed(pgyro_mag, pgyro_sr)
    feats.update(_percentiles(pgsd, 'phone_gyro_std'))
    feats.update(_percentiles(pgen, 'phone_gyro_energy'))
    feats.update(_percentiles(pgdf, 'phone_gyro_domfreq'))

    # ---- phone linear acceleration (gravity removed) -----------------------
    plax = np.array(data['phone_lax']['values'])
    play = np.array(data['phone_lay']['values'])
    plaz = np.array(data['phone_laz']['values'])
    placc_sr  = _samplerate(data['phone_lax']['raw_timestamps'], len(plax))
    placc_mag = _mag(plax, play, plaz)

    _, plsd, _, pldf, _ = _windowed(placc_mag, placc_sr)
    feats.update(_percentiles(plsd, 'phone_lacc_std'))
    feats.update(_percentiles(pldf, 'phone_lacc_domfreq'))

    # ---- temperature -------------------------------------------------------
    # Mean ≈ contact/ambient temp. Slope: ankle warms during exercise (+),
    # wrist/belt cool outdoors (−). Both help distinguish ankle from others.
    tmp = np.array(data['temperature']['values'], dtype=float)
    feats['temp_mean']  = float(tmp.mean())
    feats['temp_slope'] = float(np.polyfit(np.arange(len(tmp)), tmp, 1)[0] * len(tmp))

    # ---- altitude ----------------------------------------------------------
    alt = np.array(data['altitude']['values'])
    feats['altitude_net_gain'] = _net_change(alt)
    feats['altitude_range']    = float(alt.max() - alt.min())

    # ---- barometric pressure (optional) ------------------------------------
    if 'phone_pressure' in data:
        prs = np.array(data['phone_pressure']['values'])
        feats['pressure_net_delta'] = _net_change(prs)
    else:
        feats['pressure_net_delta'] = np.nan

    # ---- phone magnetometer heading ----------------------------------------
    pmx = np.array(data['phone_mx']['values'])
    pmy = np.array(data['phone_my']['values'])
    feats.update(_heading_features(pmx, pmy))

    return feats


def extract_labels(lbl):
    def normalize_acts(raw):
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, list):
            return {name: (i in raw) for i, name in ACT_INT_MAP.items()}
        return {name: False for name in ACT_INT_MAP.values()}

    acts = normalize_acts(lbl.get('activities', {}))
    sc   = lbl.get('step_count', None)
    return {
        'watch_loc':  int(lbl['watch_loc']),
        'path_idx':   int(lbl['path_idx']),
        'standing':   bool(acts.get('standing', False)),
        'walking':    bool(acts.get('walking',  False)),
        'running':    bool(acts.get('running',  False)),
        'cycling':    bool(acts.get('cycling',  False)),
        'step_count': int(sc) if (sc is not None and sc != -1) else np.nan,
    }


# ---------------------------------------------------------------------------
# Directory processor
# ---------------------------------------------------------------------------

def process_directory(directory, has_labels):
    files = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith('.pkl')
    )
    print(f'Processing {len(files)} files from {directory} ...')
    rows = []
    for i, fpath in enumerate(files):
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(files)}')
        try:
            with open(fpath, 'rb') as f:
                raw = pickle.load(f)
            feats = extract_features(raw)
            feats['filename'] = os.path.basename(fpath)
            if has_labels:
                feats.update(extract_labels(raw['labels']))
            rows.append(feats)
        except Exception as e:
            print(f'  ERROR {os.path.basename(fpath)}: {e}')
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    df_train = process_directory(TRAIN_DIR, has_labels=True)
    df_test  = process_directory(TEST_DIR,  has_labels=False)

    # put filename first, then labels, then features
    label_cols   = ['watch_loc', 'path_idx', 'standing', 'walking',
                    'running', 'cycling', 'step_count']
    id_cols      = ['filename']
    feature_cols = [c for c in df_train.columns
                    if c not in id_cols + label_cols]

    df_train = df_train[id_cols + label_cols + feature_cols]
    df_test  = df_test[ id_cols + feature_cols]

    df_train.to_csv('features_train.csv', index=False)
    df_test.to_csv( 'features_test.csv',  index=False)

    print(f'\nTrain : {df_train.shape}  →  features_train.csv')
    print(f'Test  : {df_test.shape}   →  features_test.csv')
    print(f'Features ({len(feature_cols)}): {sorted(feature_cols)}')
    print(f'\nMissing values in train:\n'
          f'{df_train[feature_cols].isna().sum()[df_train[feature_cols].isna().sum() > 0]}')
