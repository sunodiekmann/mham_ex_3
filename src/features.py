"""
Feature extraction for MHAM Exercise 3.

Produces results/features_train.csv and results/features_test.csv.
Each row = one recording. No labels in the test CSV.

Feature groups:
  - duration
  - acc_{mean/std/energy/domfreq}_{p25/p50/p75/p90}  : watch acc magnitude, windowed
  - gyro_{std/energy/domfreq}_{p25/p50/p75}           : watch gyro magnitude, windowed
  - phone_acc_{std/energy/domfreq}_{p25/p50/p75}      : phone acc magnitude, windowed
  - phone_gyro_{std/energy/domfreq}_{p25/p50/p75}     : phone gyro magnitude, windowed
  - phone_lacc_{std/domfreq}_{p25/p50/p75}            : phone linear acc (gravity removed)
  - acc_high_std_frac    : fraction of windows with acc_std > 0.5 g  (running proxy)
  - altitude_net_gain    : elevation change end-start [m]
  - altitude_range       : max-min altitude [m]
  - pressure_net_delta   : pressure change end-start [hPa], NaN if absent
  - heading_{mean/std/mode/entropy} : phone magnetometer heading statistics
  - heading_hist_{0..35} : 36 × 10° normalised histogram of phone heading
"""

import os
import pickle

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import entropy as scipy_entropy

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TRAIN  = os.path.join(ROOT, 'data', 'train')
DATA_TEST   = os.path.join(ROOT, 'data', 'test')
RESULTS_DIR = os.path.join(ROOT, 'results')

WINDOW_S        = 5.0
OVERLAP         = 0.5
HEADING_BINS    = 36
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
    spec_entropies: Shannon entropy of the normalised FFT magnitude in 0.5-5 Hz.
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
    """Per-window Pearson correlation (Ravi et al. 2005)."""
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
    n = max(1, int(len(vals) * frac))
    return float(np.mean(vals[-n:]) - np.mean(vals[:n]))


def _extract_gravity(ax, ay, az, sr, cutoff=0.3):
    """
    Extract gravity vector components per axis via low-pass filter.
    Walking/running spectra sit at 1-3 Hz, well above 0.3 Hz, so the LP
    output is the slowly-changing gravity projection onto each watch axis.
    """
    n = min(len(ax), len(ay), len(az))
    ax, ay, az = ax[:n], ay[:n], az[:n]
    nyq = sr / 2.0
    if nyq <= cutoff:
        return ax.copy(), ay.copy(), az.copy()
    b, a = butter(4, cutoff / nyq, btype='low')
    return filtfilt(b, a, ax), filtfilt(b, a, ay), filtfilt(b, a, az)


def _orientation_features(ax, ay, az, sr):
    """
    Orientation-aware features from the spec-constrained watch placement.
    Uses LP-filtered accel as the gravity estimate (robust during movement).

    Per spec:
      - Wrist: display normal horizontal, forearm-aligned axis carries gravity
      - Ankle: display faces left foot, leg-aligned axis carries gravity
      - Belt : rotation unspecified → gravity axis varies across users
    """
    gx, gy, gz = _extract_gravity(ax, ay, az, sr, cutoff=0.3)
    n = len(gx)
    if n == 0:
        keys = ['grav_abs_x', 'grav_abs_y', 'grav_abs_z',
                'grav_frac_x', 'grav_frac_y', 'grav_frac_z',
                'grav_secondary_ratio', 'grav_spread',
                'grav_dom_axis_x', 'grav_dom_axis_y', 'grav_dom_axis_z',
                'grav_axis_consistency', 'grav_tilt']
        return {k: np.nan for k in keys}

    # Mean |gravity| per axis over the whole recording
    abs_means = np.array([np.abs(gx).mean(), np.abs(gy).mean(), np.abs(gz).mean()])
    feats = {
        'grav_abs_x': float(abs_means[0]),
        'grav_abs_y': float(abs_means[1]),
        'grav_abs_z': float(abs_means[2]),
    }

    total = abs_means.sum() + 1e-9
    feats['grav_frac_x'] = float(abs_means[0] / total)
    feats['grav_frac_y'] = float(abs_means[1] / total)
    feats['grav_frac_z'] = float(abs_means[2] / total)

    # Sorted for axis-agnostic shape stats
    sorted_abs = np.sort(abs_means)[::-1]
    # Belt: rotation unspecified → gravity spread across axes → high secondary ratio
    # Wrist/Ankle: rotation fixed → one axis dominates → low secondary ratio
    feats['grav_secondary_ratio'] = float(sorted_abs[1] / (sorted_abs[0] + 1e-9))
    # Entropy-like spread: 0 = single-axis, 1 = uniform across 3 axes
    fracs = abs_means / total
    feats['grav_spread'] = float(-np.sum(fracs * np.log(fracs + 1e-9)) / np.log(3))

    # Windowed dominant-axis consistency
    w = int(5.0 * sr)
    step = w // 2
    if w < 8 or n < w:
        # Fall back to whole-recording dominant axis
        dom = int(np.argmax(abs_means))
        feats['grav_dom_axis_x'] = float(dom == 0)
        feats['grav_dom_axis_y'] = float(dom == 1)
        feats['grav_dom_axis_z'] = float(dom == 2)
        feats['grav_axis_consistency'] = 1.0
    else:
        dom_axes = []
        for s in range(0, n - w, step):
            win_abs = np.array([np.abs(gx[s:s+w]).mean(),
                                np.abs(gy[s:s+w]).mean(),
                                np.abs(gz[s:s+w]).mean()])
            dom_axes.append(int(np.argmax(win_abs)))
        counts = np.bincount(dom_axes, minlength=3)
        mode = int(counts.argmax())
        feats['grav_dom_axis_x'] = float(mode == 0)
        feats['grav_dom_axis_y'] = float(mode == 1)
        feats['grav_dom_axis_z'] = float(mode == 2)
        feats['grav_axis_consistency'] = float(counts[mode] / len(dom_axes))

    # Gravity tilt: angle between gravity and the dominant axis (0 = pure alignment)
    dom_component = sorted_abs[0]
    grav_norm = np.sqrt((abs_means**2).sum()) + 1e-9
    feats['grav_tilt'] = float(np.arccos(np.clip(dom_component / grav_norm, 0, 1)))

    return feats


def _per_axis_dynamic_features(ax, ay, az, sr, prefix='axis'):
    """
    Per-axis dynamic signal features.
    Which axis carries the walking/stride periodicity is orientation-dependent:
      - Ankle: heel-strike → high std + strong 1-3 Hz on leg-vertical axis
      - Wrist: arm swing  → strong 0.5-2 Hz on forearm-sagittal axis
      - Belt : torso bob  → weaker signal, vertical axis
    For gyro (prefix='gyro_axis'), the dominant axis is the rotation axis:
      - Ankle: leg swing around hip (single consistent axis)
      - Wrist: forearm rotation + hand pronation (mixed axes)
      - Belt : minimal rotation overall
    """
    n = min(len(ax), len(ay), len(az))
    keys_template = ['std_x', 'std_y', 'std_z',
                     'walkband_energy_x', 'walkband_energy_y', 'walkband_energy_z',
                     'walkband_frac_x', 'walkband_frac_y', 'walkband_frac_z',
                     'walkband_dom_x', 'walkband_dom_y', 'walkband_dom_z']
    keys = [f'{prefix}_{k}' for k in keys_template]
    if n < int(sr * 2):
        return {k: np.nan for k in keys}

    feats = {}
    stds = [float(ax[:n].std()), float(ay[:n].std()), float(az[:n].std())]
    feats[f'{prefix}_std_x'] = stds[0]
    feats[f'{prefix}_std_y'] = stds[1]
    feats[f'{prefix}_std_z'] = stds[2]

    ax_c = ax[:n] - ax[:n].mean()
    ay_c = ay[:n] - ay[:n].mean()
    az_c = az[:n] - az[:n].mean()
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    band  = (freqs >= 1.0) & (freqs <= 3.0)

    energies = []
    for sig in (ax_c, ay_c, az_c):
        if band.any():
            fm = np.abs(np.fft.rfft(sig))
            energies.append(float((fm[band]**2).sum() / n))
        else:
            energies.append(0.0)
    feats[f'{prefix}_walkband_energy_x'] = energies[0]
    feats[f'{prefix}_walkband_energy_y'] = energies[1]
    feats[f'{prefix}_walkband_energy_z'] = energies[2]

    total_e = sum(energies) + 1e-9
    feats[f'{prefix}_walkband_frac_x'] = energies[0] / total_e
    feats[f'{prefix}_walkband_frac_y'] = energies[1] / total_e
    feats[f'{prefix}_walkband_frac_z'] = energies[2] / total_e

    dom = int(np.argmax(energies))
    feats[f'{prefix}_walkband_dom_x'] = float(dom == 0)
    feats[f'{prefix}_walkband_dom_y'] = float(dom == 1)
    feats[f'{prefix}_walkband_dom_z'] = float(dom == 2)

    return feats


def _circular_diff(a, b):
    """Signed smallest difference between two angles in degrees, in [-180, 180]."""
    d = (a - b + 180) % 360 - 180
    return d


def _heading_features(mx, my):
    mx = np.ravel(mx).astype(float)
    my = np.ravel(my).astype(float)
    n  = min(len(mx), len(my))
    nan_keys = ([f'heading_hist_{i}' for i in range(HEADING_BINS)]
                + ['heading_mean', 'heading_std', 'heading_mode', 'heading_entropy',
                   'heading_circular_std',
                   'heading_num_modes', 'heading_turns_45', 'heading_turns_90',
                   'heading_total_change', 'heading_change_rate',
                   'heading_first_turn_frac', 'heading_last_turn_frac',
                   'heading_longest_straight_frac',
                   'heading_early_mode_frac', 'heading_late_mode_frac'])
    if n == 0:
        return {k: np.nan for k in nan_keys}

    heading = np.degrees(np.arctan2(my[:n], mx[:n])) % 360
    bins    = np.linspace(0, 360, HEADING_BINS + 1)
    hist, _ = np.histogram(heading, bins=bins)
    hist_n  = hist.astype(float) / (hist.sum() + 1e-9)

    feats = {f'heading_hist_{i}': hist_n[i] for i in range(HEADING_BINS)}
    feats['heading_mean']    = float(np.mean(heading))
    feats['heading_std']     = float(np.std(heading))
    feats['heading_mode']    = float(bins[np.argmax(hist)])
    feats['heading_entropy'] = float(scipy_entropy(hist_n + 1e-9))

    # Circular std (linear std is wrong near 0/360 wrap). Mardia's definition:
    h_rad = np.deg2rad(heading)
    R = np.sqrt(np.mean(np.cos(h_rad))**2 + np.mean(np.sin(h_rad))**2)
    feats['heading_circular_std'] = float(np.sqrt(-2 * np.log(max(R, 1e-9))))

    # Number of modes: bins holding >= 5% of samples (distinct heading segments)
    feats['heading_num_modes'] = float(np.sum(hist_n >= 0.05))

    # Temporal/turn features — smooth heading to ~1 Hz and scan for jumps
    # Magnetometer sample rate: variable; we compute turns from consecutive
    # signed circular diffs. A turn is >= some delta sustained for a few samples.
    if n < 10:
        for k in ('heading_turns_45', 'heading_turns_90',
                  'heading_total_change', 'heading_change_rate',
                  'heading_first_turn_frac', 'heading_last_turn_frac',
                  'heading_longest_straight_frac',
                  'heading_early_mode_frac', 'heading_late_mode_frac'):
            feats[k] = np.nan
        return feats

    # Downsample heading to ~50 samples for turn analysis (robust to noise)
    target = min(200, n)
    idx = np.linspace(0, n - 1, target).astype(int)
    h_ds = heading[idx]

    # Consecutive signed differences on downsampled heading
    diffs = _circular_diff(h_ds[1:], h_ds[:-1])
    total_change = float(np.sum(np.abs(diffs)))
    feats['heading_total_change'] = total_change
    feats['heading_change_rate']  = total_change / max(1, len(diffs))

    # Turn detection: sum cumulative diffs over a short window (~5 samples);
    # if magnitude crosses threshold, call it a turn.
    window_diff = uniform_filter1d_cumsum(diffs, 5)
    # Zero-crossing logic for counting distinct turns:
    above_45 = np.abs(window_diff) > 45
    above_90 = np.abs(window_diff) > 90
    feats['heading_turns_45'] = float(_count_regions(above_45))
    feats['heading_turns_90'] = float(_count_regions(above_90))

    # Position of first / last turn (fraction of recording)
    if above_45.any():
        feats['heading_first_turn_frac'] = float(np.argmax(above_45) / len(above_45))
        feats['heading_last_turn_frac']  = float(
            (len(above_45) - 1 - np.argmax(above_45[::-1])) / len(above_45))
    else:
        feats['heading_first_turn_frac'] = 1.0
        feats['heading_last_turn_frac']  = 0.0

    # Longest straight-line segment (fraction of recording where heading varies little)
    straight_mask = np.abs(window_diff) < 15
    longest = _longest_run(straight_mask)
    feats['heading_longest_straight_frac'] = float(longest / max(1, len(straight_mask)))

    # Dominant heading mode in early vs late half — captures path start/end direction
    half = len(h_ds) // 2
    early_mode = int(np.bincount((h_ds[:half] // 10).astype(int), minlength=36).argmax())
    late_mode  = int(np.bincount((h_ds[half:] // 10).astype(int), minlength=36).argmax())
    feats['heading_early_mode_frac'] = float(early_mode) / 36.0
    feats['heading_late_mode_frac']  = float(late_mode)  / 36.0

    return feats


def uniform_filter1d_cumsum(arr, w):
    """Centred rolling sum of length w (NumPy-only, no scipy dependency here)."""
    w = max(1, int(w))
    if w == 1 or len(arr) < w:
        return arr.copy()
    pad = np.concatenate([np.zeros(w // 2), arr, np.zeros(w // 2)])
    c = np.cumsum(np.insert(pad, 0, 0))
    return (c[w:] - c[:-w])[:len(arr)]


def _count_regions(mask):
    """Number of contiguous True regions in a boolean array."""
    if len(mask) == 0:
        return 0
    padded = np.concatenate([[False], mask, [False]])
    transitions = np.diff(padded.astype(int))
    return int(np.sum(transitions == 1))


def _longest_run(mask):
    """Length of the longest contiguous True region."""
    if not mask.any():
        return 0
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _longest_run_with_tolerance(mask, tolerance_samples):
    """
    Length of the longest True run, allowing False gaps up to
    `tolerance_samples` samples to be absorbed into the run.

    Mirrors the spec's "up to 8 s of interruption still counts as
    uninterrupted activity" rule.
    """
    if len(mask) == 0 or not mask.any():
        return 0
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    if len(starts) == 0:
        return 0
    merged_start, merged_end = starts[0], ends[0]
    best = merged_end - merged_start
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged_end <= tolerance_samples:
            merged_end = e              # absorb gap into current merged run
        else:
            merged_start, merged_end = s, e
        best = max(best, merged_end - merged_start)
    return best


# Per-sensor activity-signature thresholds. Watch is in g/deg-s, phone is in
# m/s² and rad/s (Android defaults), so thresholds differ by ~10×.
# Tuned on per-class median statistics (sweeps/tune_activity_thresholds.py).
_SEG_THRESH = {
    'watch': {     # acc in g, gyro in deg/s
        'run_freq':   2.2,   'run_acc':   0.5,  'run_gyro':   30.0,
        'walk_freq_lo': 1.3, 'walk_freq_hi': 2.5, 'walk_acc': 0.25,
        'stand_acc':  0.15,  'stand_mean': 1.0, 'stand_tol':  0.15,
        'cycle_acc_max': 0.4, 'cycle_gyro_max': 45.0,
    },
    'phone': {     # acc in m/s² (gravity included → ~9.8 baseline mean)
        'run_freq':   2.0,   'run_acc':   3.7,  'run_gyro':   0.55,
        'walk_freq_lo': 1.3, 'walk_freq_hi': 2.5, 'walk_acc': 2.8,
        'stand_acc':  2.0,   'stand_mean': 9.8, 'stand_tol':  2.0,
        'cycle_acc_max': 2.5, 'cycle_gyro_max': 0.6,
    },
}


def _aerial_phase_features(acc_mag, sr, sensor=''):
    """
    Aerial-phase detection: during running, all feet are off ground briefly,
    so acc magnitude drops to near-zero (free fall). Walking never has this.

    For each 0.3-second sliding sub-window, compute the MIN of acc magnitude.
    Running stride: large peaks (heel-strike) + brief drops (aerial phase).
    Walking: peaks but no drop (one foot always on ground).

    Per-sensor scale and signal:
      watch:     raw acc magnitude in g, threshold 0.5 g (gravity present)
      phone_l:   LINEAR acc magnitude (gravity removed), threshold 3 m/s²
                 (free fall → linear acc near 0)
    """
    keys = [f'aerial_{sensor}_min_p05', f'aerial_{sensor}_min_p25',
            f'aerial_{sensor}_below_thresh_frac',
            f'aerial_{sensor}_min_per_window_count']
    if len(acc_mag) < int(sr * 1.0):
        return {k: np.nan for k in keys}

    threshold = 0.5 if sensor == 'watch' else 3.0

    # Rolling min over 0.3 s windows (~aerial phase duration in running)
    sub_w = max(2, int(sr * 0.3))
    n = len(acc_mag)
    if n < sub_w:
        return {k: np.nan for k in keys}

    # Cheap rolling min via stride trick (use minimum_filter1d analogue)
    # Just sample the rolling min at sub_w/2 step to keep it fast.
    step = max(1, sub_w // 2)
    mins = []
    for i in range(0, n - sub_w + 1, step):
        mins.append(float(acc_mag[i:i + sub_w].min()))
    mins = np.array(mins)

    feats = {}
    feats[f'aerial_{sensor}_min_p05'] = float(np.percentile(mins, 5))
    feats[f'aerial_{sensor}_min_p25'] = float(np.percentile(mins, 25))
    feats[f'aerial_{sensor}_below_thresh_frac'] = float((mins < threshold).mean())
    # How many "aerial phases" did we see? (each stride during running has one)
    feats[f'aerial_{sensor}_min_per_window_count'] = int((mins < threshold).sum())
    return feats


def _activity_segment_features(acc_mean, acc_std, acc_dom_freq, gyro_std,
                                window_s=5.0, overlap=0.5, sensor=''):
    """
    Activity-signature detection at SUB-WINDOW level, summarised to a
    recording-level feature matching the >=60s label definition.

    Signatures (per-sensor thresholds in _SEG_THRESH):
      running  : fast cadence + high amplitude + active gyro
      walking  : moderate cadence + moderate amplitude
      standing : near-zero motion (acc near-gravity-only)
      cycling  : LOW gyro (legs constrained by pedals) + low-to-moderate acc
                 (no heel-strikes)
    """
    th = _SEG_THRESH.get(sensor, _SEG_THRESH['watch'])
    step_s = window_s * (1 - overlap)
    n = len(acc_std)
    if n == 0:
        keys = []
        for name in ('running', 'walking', 'standing', 'cycling'):
            keys += [f'seg_{sensor}_{name}_longest_s',
                     f'seg_{sensor}_{name}_any_60s',
                     f'seg_{sensor}_{name}_frac']
        return {k: np.nan for k in keys}

    acc_mean = np.nan_to_num(acc_mean, nan=0.0)
    acc_std  = np.nan_to_num(acc_std,  nan=0.0)
    acc_freq = np.nan_to_num(acc_dom_freq, nan=0.0)
    gyro_std_arr = np.nan_to_num(gyro_std, nan=0.0) if len(gyro_std) else np.zeros(n)
    if len(gyro_std_arr) < n:
        gyro_std_arr = np.pad(gyro_std_arr, (0, n - len(gyro_std_arr)), mode='edge')
    else:
        gyro_std_arr = gyro_std_arr[:n]

    running_mask  = ((acc_freq >= th['run_freq']) & (acc_std > th['run_acc'])
                     & (gyro_std_arr > th['run_gyro']))
    walking_mask  = ((acc_freq >= th['walk_freq_lo']) & (acc_freq <= th['walk_freq_hi'])
                     & (acc_std > th['walk_acc']) & ~running_mask)
    standing_mask = ((acc_std < th['stand_acc'])
                     & (np.abs(acc_mean - th['stand_mean']) < th['stand_tol']))
    # Cycling: low gyro AND acc below cycling-max. NOT mutually exclusive with
    # walking-cadence (pedaling can be in walking-cadence band) — we only
    # exclude running, since "cycling-and-running" is impossible.
    cycling_mask  = ((acc_std < th['cycle_acc_max'])
                     & (gyro_std_arr < th['cycle_gyro_max'])
                     & ~running_mask)

    # Spec: up to 8 s of interruption still counts as uninterrupted.
    tolerance_samples = max(1, int(8.0 / step_s))

    feats = {}
    for name, mask in [('running',  running_mask),
                        ('walking',  walking_mask),
                        ('standing', standing_mask),
                        ('cycling',  cycling_mask)]:
        longest_w = _longest_run_with_tolerance(mask, tolerance_samples)
        longest_s = longest_w * step_s
        feats[f'seg_{sensor}_{name}_longest_s'] = float(longest_s)
        feats[f'seg_{sensor}_{name}_any_60s']   = int(longest_s >= 60)
        feats[f'seg_{sensor}_{name}_frac']      = float(mask.mean())
    return feats


def _circular_mean_deg(angles_deg):
    """Circular mean of an array of angles in degrees."""
    r = np.deg2rad(angles_deg)
    return float(np.degrees(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))) % 360)


def _extract_turn_sequence(heading_deg,
                            sr,
                            min_straight_duration_s=3.0,
                            straight_std_thresh_deg=20.0,
                            min_turn_magnitude=45.0,
                            smoothing_window_s=2.0,
                            probe_step_s=0.5):
    """
    Detect the ordered sequence of significant turns via STABLE-SEGMENT analysis.

    A real street-corner turn takes 5-10 seconds as the walker rounds it, so
    sustained-high-velocity detection misses slow turns. Instead we:
      1. Find contiguous "stable heading" regions: circular std < threshold
         over a >= min_straight_duration_s window (i.e. walking straight).
      2. A turn is the difference between adjacent stable regions' mean
         headings. Gentle 10-second turns are captured naturally this way.

    Returns list of signed turn magnitudes in deg.
    Convention: positive = right (clockwise) turn in compass frame.
    """
    n = len(heading_deg)
    if n < int(sr * min_straight_duration_s * 2):
        return []

    # Smooth heading (wrap-safe)
    w = max(3, int(sr * smoothing_window_s))
    h_rad = np.deg2rad(heading_deg)
    sin_s = uniform_filter1d_cumsum(np.sin(h_rad), w) / w
    cos_s = uniform_filter1d_cumsum(np.cos(h_rad), w) / w
    heading_s = np.degrees(np.arctan2(sin_s, cos_s)) % 360

    window_samples = int(sr * min_straight_duration_s)
    step = max(1, int(sr * probe_step_s))

    def _circ_stats(seg):
        r = np.deg2rad(seg)
        c, s = np.mean(np.cos(r)), np.mean(np.sin(r))
        R = np.sqrt(c*c + s*s)
        mean_deg = float(np.degrees(np.arctan2(s, c)) % 360)
        c_std_deg = float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-9)))))
        return mean_deg, c_std_deg

    # Greedy: scan forward, find first stable window, grow it while still stable,
    # then jump to end and search for the next stable window.
    stable_regions = []
    i = 0
    while i + window_samples <= n:
        _, cstd = _circ_stats(heading_s[i:i + window_samples])
        if cstd < straight_std_thresh_deg:
            # Grow this stable region
            j = i + window_samples
            while j + step <= n:
                _, cstd2 = _circ_stats(heading_s[j - window_samples:j])
                if cstd2 < straight_std_thresh_deg:
                    j += step
                else:
                    break
            # Record the stable region's representative heading
            mean_deg, _ = _circ_stats(heading_s[i:j])
            stable_regions.append((i, j, mean_deg))
            i = j
        else:
            i += step

    # Turns = signed-smallest diff between consecutive stable-region headings
    turns = []
    for k in range(1, len(stable_regions)):
        prev_h = stable_regions[k - 1][2]
        curr_h = stable_regions[k][2]
        diff = float(_circular_diff(curr_h, prev_h))
        if abs(diff) >= min_turn_magnitude:
            turns.append(diff)
    return turns


def _turn_sequence_features(mx, my, sr, max_turns=6):
    """
    Extract speed-invariant turn-sequence features:
      - turn_i_angle: signed magnitude of the i-th detected turn (0 if absent).
        Discriminates paths directly: e.g. P0/P1 both start with ~+150° right,
        P2 starts with ~+80° right, so turn_1_angle alone separates P2.
      - num_significant_turns: overall turn count
        (P1 has 5 from the Clausiusstrasse loop; P0/P2 have 3).
      - total_signed_rotation / total_abs_rotation: net & absolute rotation.
      - turns_positive / turns_negative: counts of right vs left turns.
    """
    mx = np.ravel(mx).astype(float)
    my = np.ravel(my).astype(float)
    n  = min(len(mx), len(my))
    keys = ([f'turn_{i+1}_angle' for i in range(max_turns)] +
            [f'turn_{i+1}_abs' for i in range(max_turns)] +
            ['num_significant_turns', 'total_signed_rotation',
             'total_abs_rotation', 'turns_positive', 'turns_negative'])
    if n < 20 or sr <= 0:
        return {k: np.nan for k in keys}

    heading = np.degrees(np.arctan2(my[:n], mx[:n])) % 360
    turns = _extract_turn_sequence(heading, sr=sr)

    feats = {}
    for i in range(max_turns):
        if i < len(turns):
            feats[f'turn_{i+1}_angle'] = float(turns[i])
            feats[f'turn_{i+1}_abs']   = float(abs(turns[i]))
        else:
            feats[f'turn_{i+1}_angle'] = 0.0
            feats[f'turn_{i+1}_abs']   = 0.0
    feats['num_significant_turns'] = float(len(turns))
    feats['total_signed_rotation'] = float(sum(turns)) if turns else 0.0
    feats['total_abs_rotation']    = float(sum(abs(t) for t in turns)) if turns else 0.0
    feats['turns_positive']        = float(sum(1 for t in turns if t > 0))
    feats['turns_negative']        = float(sum(1 for t in turns if t < 0))
    return feats


def _heading_trajectory_features(mx, my, n_segments=10):
    """
    Decile-indexed heading trajectory features: captures the TEMPORAL shape
    of the walk (which the histogram discards). This is the unique path
    signature — see sweeps/visualize_heading.py row 2.

    Emits per-segment circular-mean heading, per-segment turn intensity,
    and the mean pairwise angular diff between consecutive segments.
    """
    mx = np.ravel(mx).astype(float)
    my = np.ravel(my).astype(float)
    n  = min(len(mx), len(my))
    keys = (
        [f'heading_seq_{i}_sin' for i in range(n_segments)] +
        [f'heading_seq_{i}_cos' for i in range(n_segments)] +
        [f'heading_turnrate_{i}' for i in range(n_segments)] +
        ['heading_seq_mean_jump', 'heading_seq_max_jump',
         'heading_seq_start_vs_end']
    )
    if n < n_segments * 5:
        return {k: np.nan for k in keys}

    heading = np.degrees(np.arctan2(my[:n], mx[:n])) % 360

    # Circular mean per segment. We store sin/cos components (not the raw
    # angle) so the classifier sees a continuous feature even near 0/360 wrap.
    # Turn intensity = mean |Δheading| within the segment.
    seg_means  = []
    seg_turns  = []
    feats = {}
    for i in range(n_segments):
        lo = int(n * i / n_segments)
        hi = int(n * (i + 1) / n_segments)
        seg = heading[lo:hi]
        if len(seg) < 2:
            mean_deg = 0.0
            tr = 0.0
        else:
            mean_deg = _circular_mean_deg(seg)
            diffs = _circular_diff(seg[1:], seg[:-1])
            tr = float(np.mean(np.abs(diffs)))
        seg_means.append(mean_deg)
        seg_turns.append(tr)
        rad = np.deg2rad(mean_deg)
        feats[f'heading_seq_{i}_sin']    = float(np.sin(rad))
        feats[f'heading_seq_{i}_cos']    = float(np.cos(rad))
        feats[f'heading_turnrate_{i}']   = tr

    # Jumps between consecutive segments (signed-smallest angular differences)
    jumps = [abs(_circular_diff(seg_means[i + 1], seg_means[i]))
             for i in range(n_segments - 1)]
    feats['heading_seq_mean_jump']    = float(np.mean(jumps)) if jumps else 0.0
    feats['heading_seq_max_jump']     = float(np.max(jumps))  if jumps else 0.0
    feats['heading_seq_start_vs_end'] = float(abs(_circular_diff(seg_means[-1],
                                                                  seg_means[0])))
    return feats


# ---------------------------------------------------------------------------
# Per-recording feature extraction
# ---------------------------------------------------------------------------

def extract_features(raw):
    """raw: dict loaded from a .pkl file. Returns a flat feature dict."""
    data  = raw['data']
    feats = {}

    ax_ts    = data['ax']['raw_timestamps']
    duration = (ax_ts[-1][1] - ax_ts[0][1]) / 1000.0
    feats['duration'] = duration

    # Watch accelerometer
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
    still_mask = (sd < 0.15) & (np.abs(mn - 1.0) < 0.15)
    feats['acc_still_frac'] = float(still_mask.mean()) if len(still_mask) > 0 else np.nan

    WALK_MASK = (df >= 1.0) & (df <= 3.0) & (sd > 0.3)
    feats.update(_percentiles(sd[WALK_MASK],  'acc_walk_std'))
    feats.update(_percentiles(se[WALK_MASK],  'acc_walk_spec_entropy'))
    feats['acc_walk_frac'] = float(WALK_MASK.mean()) if len(WALK_MASK) > 0 else np.nan

    n3 = min(len(ax), len(ay), len(az))
    axis_stds = sorted([float(ax[:n3].std()), float(ay[:n3].std()), float(az[:n3].std())])
    feats['acc_anisotropy'] = axis_stds[2] / (axis_stds[0] + 1e-9)

    feats.update(_percentiles(_windowed_corr(ax, ay, acc_sr), 'acc_corr_xy'))
    feats.update(_percentiles(_windowed_corr(ax, az, acc_sr), 'acc_corr_xz'))
    feats.update(_percentiles(_windowed_corr(ay, az, acc_sr), 'acc_corr_yz'))

    # Orientation-aware features (from spec-constrained placement)
    feats.update(_orientation_features(ax, ay, az, acc_sr))
    feats.update(_per_axis_dynamic_features(ax, ay, az, acc_sr, prefix='axis'))

    # Save watch-sub-window stats for segment features (populated after gyro)
    _watch_sub_window_stats = (mn, sd, df)

    # Watch gyroscope
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

    # Per-axis gyro features: rotation axis tells you mounting location
    feats.update(_per_axis_dynamic_features(gx, gy, gz, gyro_sr, prefix='gyro_axis'))

    # Segment-based activity features (WATCH IMU): longest sustained
    # running/walking/standing/cycling region, matches the >=60 s label.
    mn_w, sd_w, df_w = _watch_sub_window_stats
    feats.update(_activity_segment_features(
        mn_w, sd_w, df_w, gsd, window_s=WINDOW_S, overlap=OVERLAP, sensor='watch'))

    # Aerial-phase detection (running-specific physics)
    feats.update(_aerial_phase_features(acc_mag, acc_sr, sensor='watch'))

    # Phone accelerometer
    pax = np.array(data['phone_ax']['values'])
    pay = np.array(data['phone_ay']['values'])
    paz = np.array(data['phone_az']['values'])
    pacc_sr  = _samplerate(data['phone_ax']['raw_timestamps'], len(pax))
    pacc_mag = _mag(pax, pay, paz)

    pmn, psd, pen, pdf, _ = _windowed(pacc_mag, pacc_sr)
    feats.update(_percentiles(psd, 'phone_acc_std'))
    feats.update(_percentiles(pen, 'phone_acc_energy'))
    feats.update(_percentiles(pdf, 'phone_acc_domfreq'))
    feats.update(_percentiles(_windowed_corr(pax, pay, pacc_sr), 'phone_acc_corr_xy'))
    feats.update(_percentiles(_windowed_corr(pax, paz, pacc_sr), 'phone_acc_corr_xz'))
    feats.update(_percentiles(_windowed_corr(pay, paz, pacc_sr), 'phone_acc_corr_yz'))

    # Phone gyroscope
    pgx = np.array(data['phone_gx']['values'])
    pgy = np.array(data['phone_gy']['values'])
    pgz = np.array(data['phone_gz']['values'])
    pgyro_sr  = _samplerate(data['phone_gx']['raw_timestamps'], len(pgx))
    pgyro_mag = _mag(pgx, pgy, pgz)

    _, pgsd, pgen, pgdf, _ = _windowed(pgyro_mag, pgyro_sr)
    feats.update(_percentiles(pgsd, 'phone_gyro_std'))
    feats.update(_percentiles(pgen, 'phone_gyro_energy'))
    feats.update(_percentiles(pgdf, 'phone_gyro_domfreq'))

    # Segment features on phone IMU (location-independent: phone is in pocket)
    feats.update(_activity_segment_features(
        pmn, psd, pdf, pgsd, window_s=WINDOW_S, overlap=OVERLAP, sensor='phone'))
    # Aerial phase: use linear acc (gravity removed) so free fall → ~0

    # Phone linear acceleration (gravity removed)
    plax = np.array(data['phone_lax']['values'])
    play = np.array(data['phone_lay']['values'])
    plaz = np.array(data['phone_laz']['values'])
    placc_sr  = _samplerate(data['phone_lax']['raw_timestamps'], len(plax))
    placc_mag = _mag(plax, play, plaz)

    _, plsd, _, pldf, _ = _windowed(placc_mag, placc_sr)
    feats.update(_percentiles(plsd, 'phone_lacc_std'))
    feats.update(_percentiles(pldf, 'phone_lacc_domfreq'))

    # Aerial-phase on PHONE LINEAR acc (gravity removed) — drops to ~0 during
    # the airborne moment of running. Walking lacks this entirely.
    feats.update(_aerial_phase_features(placc_mag, placc_sr, sensor='phone'))

    # Temperature
    tmp = np.array(data['temperature']['values'], dtype=float)
    feats['temp_mean']  = float(tmp.mean())
    feats['temp_slope'] = float(np.polyfit(np.arange(len(tmp)), tmp, 1)[0] * len(tmp))

    # Altitude
    alt = np.array(data['altitude']['values'])
    feats['altitude_net_gain'] = _net_change(alt)
    feats['altitude_range']    = float(alt.max() - alt.min())

    # Barometric pressure
    if 'phone_pressure' in data:
        prs = np.array(data['phone_pressure']['values'])
        feats['pressure_net_delta'] = _net_change(prs)
    else:
        feats['pressure_net_delta'] = np.nan

    # Phone magnetometer heading (histogram + turn features)
    pmx = np.array(data['phone_mx']['values'])
    pmy = np.array(data['phone_my']['values'])
    mag_sr = _samplerate(data['phone_mx']['raw_timestamps'], len(pmx))
    feats.update(_heading_features(pmx, pmy))
    # Decile-indexed heading trajectory — path-unique temporal signature
    feats.update(_heading_trajectory_features(pmx, pmy, n_segments=10))
    # Ordered turn sequence (speed-invariant; distinguishes Path 0/1/2 by
    # first-turn magnitude and total count)
    feats.update(_turn_sequence_features(pmx, pmy, sr=mag_sr, max_turns=6))

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
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df_train = process_directory(DATA_TRAIN, has_labels=True)
    df_test  = process_directory(DATA_TEST,  has_labels=False)

    label_cols   = ['watch_loc', 'path_idx', 'standing', 'walking',
                    'running', 'cycling', 'step_count']
    id_cols      = ['filename']
    feature_cols = [c for c in df_train.columns
                    if c not in id_cols + label_cols]

    df_train = df_train[id_cols + label_cols + feature_cols]
    df_test  = df_test[ id_cols + feature_cols]

    out_train = os.path.join(RESULTS_DIR, 'features_train.csv')
    out_test  = os.path.join(RESULTS_DIR, 'features_test.csv')
    df_train.to_csv(out_train, index=False)
    df_test.to_csv( out_test,  index=False)

    print(f'\nTrain : {df_train.shape}  →  {out_train}')
    print(f'Test  : {df_test.shape}   →  {out_test}')
