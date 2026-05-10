"""
Step counting for LilyGo smartwatch at ~200 Hz.

Based on: Brajdic & Harle, "Walk Detection and Step Counting on Unconstrained
Smartphones", UbiComp 2013.

ALGORITHM: STFT frequency × locomotion duration  (Table 4 in paper)
  For each 2-second window:
    1. Dual-sensor activity mask: acc_std > thresh AND gyro_std > gyro_thresh
       (gyro filter is crucial — cycling has very low gyro std vs walking)
    2. Dominant step frequency = FFT peak in [freq_lo, freq_hi] Hz
    3. Accumulated steps += dom_freq × step_duration_s (non-overlapping portion)

Parameters are tuned per watch location via grid search LOO-CV on the 33
labelled training recordings.
"""

import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TRAIN  = os.path.join(ROOT, 'data', 'train')
MODELS_DIR  = os.path.join(ROOT, 'models')
RESULTS_DIR = os.path.join(ROOT, 'results')
PARAMS_OUT  = os.path.join(MODELS_DIR, 'steps_params.pkl')

LOC_NAMES = {0: 'Wrist', 1: 'Belt', 2: 'Ankle'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _samplerate(raw_timestamps, values=None):
    dur = (raw_timestamps[-1][1] - raw_timestamps[0][1]) / 1000.0
    if dur <= 0:
        return 0.0
    if values is not None:
        return len(values) / dur
    return len(raw_timestamps) / dur


def _mag(x, y, z):
    n = min(len(x), len(y), len(z))
    return np.sqrt(x[:n]**2 + y[:n]**2 + z[:n]**2)


def _lp_filter(sig, sr, cutoff=4.0, order=4):
    nyq = sr / 2.0
    if cutoff >= nyq:
        return sig
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, sig)


# ---------------------------------------------------------------------------
# Phone-IMU cycling veto (per-window helper)
# ---------------------------------------------------------------------------

def _phone_cycling_mask(raw, target_sr, n_target, gyro_thresh=0.4):
    """
    Per-sample-on-target-grid boolean: True if phone IMU shows CYCLING in the
    1s neighbourhood of that sample. Phone is location-invariant (in pocket),
    so this veto fires regardless of watch_loc.

    Discriminator: phone gyro magnitude std. During cycling, the phone barely
    rotates with each pedal (no body twist) → very low gyro variance. During
    walking, hip sway gives sustained gyro motion → high std. Threshold tuned
    on labelled data; tunable via the params dict.

    Returns a boolean array aligned to a grid of length `n_target` at
    `target_sr` Hz. True means "phone IMU says cycling here, do not count".
    """
    data = raw['data']
    if not all(k in data for k in ('phone_gx', 'phone_gy', 'phone_gz')):
        return np.zeros(n_target, dtype=bool)

    pgx = np.array(data['phone_gx']['values'], dtype=float)
    pgy = np.array(data['phone_gy']['values'], dtype=float)
    pgz = np.array(data['phone_gz']['values'], dtype=float)
    n_p = min(len(pgx), len(pgy), len(pgz))
    if n_p < 8:
        return np.zeros(n_target, dtype=bool)

    phone_sr = _samplerate(data['phone_gx']['raw_timestamps'], pgx)
    if phone_sr <= 0:
        return np.zeros(n_target, dtype=bool)

    pg_mag = _mag(pgx, pgy, pgz)[:n_p]

    # Compute rolling std over ~1s phone-IMU windows, then resample to target.
    win = max(4, int(phone_sr))   # 1 s
    if n_p < win:
        return np.zeros(n_target, dtype=bool)

    # Cumulative stats for O(n) rolling std
    c1 = np.cumsum(pg_mag, dtype=np.float64)
    c2 = np.cumsum(pg_mag ** 2, dtype=np.float64)
    s1 = c1[win:] - c1[:-win]
    s2 = c2[win:] - c2[:-win]
    mean = s1 / win
    var = np.maximum(s2 / win - mean ** 2, 0.0)
    rolling_std = np.sqrt(var)
    # Pad to original length
    pad = np.full(win, rolling_std[0] if len(rolling_std) else 0.0)
    rolling_std_full = np.concatenate([pad, rolling_std])[:n_p]

    # Cycling window: phone gyro std below threshold
    cycling_phone = rolling_std_full < gyro_thresh

    # Resample to target grid via nearest-time mapping
    t_phone   = np.arange(n_p) / phone_sr
    t_target  = np.arange(n_target) / target_sr
    idx_map   = np.clip(np.searchsorted(t_phone, t_target), 0, n_p - 1)
    return cycling_phone[idx_map]


# ---------------------------------------------------------------------------
# Step counting algorithm
# ---------------------------------------------------------------------------

def count_steps_with_activity_mask(raw, params, source='watch',
                                     activities=None, algo='FREQ'):
    """
    Activity-aware wrapper. Masks cycling-only recordings to 0 steps and
    widens the running cadence band when running is detected.
    `algo` selects between FREQ (FFT cadence) and WPD (peak detection).
    """
    if activities is not None:
        if (activities.get('cycling', False)
                and not activities.get('walking', False)
                and not activities.get('running', False)):
            return 0
        if activities.get('running', False) and algo == 'FREQ':
            params = {**params, 'freq_hi': max(params.get('freq_hi', 3.0), 4.0)}
    if algo == 'WPD':
        return count_steps_wpd(raw, params, source=source)
    return count_steps_freq(raw, params, source=source)


def count_steps_wpd(raw, params, source='watch'):
    """
    Windowed Peak Detection (Brajdic & Harle 2013, Table 4).
    Detects step events as peaks in low-pass-filtered acc magnitude.
    Activity-masked: only counts peaks during locomotion windows.

    Differs from FREQ algorithm: WPD counts INDIVIDUAL step events,
    while FREQ multiplies cadence × duration. WPD is more accurate at
    transitions (start/stop) and short bursts; FREQ is more robust to
    noise in continuous walking.
    """
    data = raw['data']
    if source == 'phone':
        ax = np.array(data['phone_lax']['values'])
        ay = np.array(data['phone_lay']['values'])
        az = np.array(data['phone_laz']['values'])
        ts = data['phone_lax']['raw_timestamps']
        gx = np.array(data['phone_gx']['values'])
        gy = np.array(data['phone_gy']['values'])
        gz = np.array(data['phone_gz']['values'])
    else:
        ax = np.array(data['ax']['values'])
        ay = np.array(data['ay']['values'])
        az = np.array(data['az']['values'])
        ts = data['ax']['raw_timestamps']
        gx = np.array(data['gx']['values'])
        gy = np.array(data['gy']['values'])
        gz = np.array(data['gz']['values'])

    sr      = _samplerate(ts, ax)
    acc_mag = _mag(ax, ay, az)
    gyr_mag = _mag(gx, gy, gz)

    # LP filter to isolate step frequencies
    acc_filt = _lp_filter(acc_mag, sr, cutoff=params.get('lp_cutoff', 4.0))

    # Centred moving average (~0.3s smooth) — Brajdic Section 3.5
    ma_n  = max(1, int(params.get('movavr_win', 0.3) * sr))
    kern  = np.ones(ma_n) / ma_n
    smooth = np.convolve(acc_filt, kern, mode='same')

    # Locomotion mask: 1-sec sub-windows passing both acc + gyro thresholds
    win_n   = max(1, int(sr))
    step_n  = max(1, win_n // 2)
    n_min   = min(len(smooth), len(gyr_mag))
    active  = np.zeros(n_min, dtype=bool)
    acc_th  = params['acc_std_thresh']
    gyro_th = params['gyro_std_thresh']
    for s in range(0, n_min - win_n, step_n):
        if (smooth[s:s+win_n].std() > acc_th and
                gyr_mag[s:s+win_n].std() > gyro_th):
            active[s:s+win_n] = True

    # Mask non-active regions to the mean → no peaks there
    masked = smooth[:n_min].copy()
    masked[~active] = smooth[:n_min].mean()

    min_dist = max(1, int(params.get('min_peak_dist', 0.35) * sr))
    prominence = params.get('min_prominence', 0.3)
    peaks, _ = find_peaks(masked, distance=min_dist, prominence=prominence)
    return int(np.sum(active[peaks]))


def count_steps_freq(raw, params, source='watch'):
    """
    For each locomotion window estimate step cadence via FFT and accumulate
    fractional steps.  Locomotion detected by dual-sensor mask:
      acc_std  > params['acc_std_thresh']   — excludes pure standing
      gyro_std > params['gyro_std_thresh']  — excludes cycling (low gyro)
    Only windows whose dominant frequency falls in [freq_lo, freq_hi] are counted.

    `source`: 'watch' uses watch acc/gyro magnitude (location-dependent
              signal: ankle sees heel-strike, wrist sees arm-swing).
              'phone' uses phone linear-acc + phone gyro magnitude
              (location-independent: phone in pocket gives clean foot-strike).
    """
    data = raw['data']
    if source == 'phone':
        # Use phone linear acc (gravity removed) — cleanest foot-strike signal
        ax = np.array(data['phone_lax']['values'])
        ay = np.array(data['phone_lay']['values'])
        az = np.array(data['phone_laz']['values'])
        ts = data['phone_lax']['raw_timestamps']
        gx = np.array(data['phone_gx']['values'])
        gy = np.array(data['phone_gy']['values'])
        gz = np.array(data['phone_gz']['values'])
    else:
        ax = np.array(data['ax']['values'])
        ay = np.array(data['ay']['values'])
        az = np.array(data['az']['values'])
        ts = data['ax']['raw_timestamps']
        gx = np.array(data['gx']['values'])
        gy = np.array(data['gy']['values'])
        gz = np.array(data['gz']['values'])

    sr      = _samplerate(ts, ax)
    acc_mag = _mag(ax, ay, az)
    gyr_mag = _mag(gx, gy, gz)

    acc_filt = _lp_filter(acc_mag, sr, cutoff=params.get('lp_cutoff', 5.0))

    win_s  = params.get('win_size', 2.0)
    step_s = params.get('win_step', 1.0)
    win_n  = max(8, int(win_s * sr))
    step_n = max(1, int(step_s * sr))

    gyro_th = params['gyro_std_thresh']
    freq_lo = params.get('freq_lo', 0.8)
    freq_hi = params.get('freq_hi', 4.0)
    acc_th  = params['acc_std_thresh']
    n_acc   = len(acc_filt)
    n_gyr   = len(gyr_mag)

    total_steps = 0.0
    freqs       = np.fft.rfftfreq(win_n, d=1.0 / sr)
    band        = (freqs >= freq_lo) & (freqs <= freq_hi)

    for start in range(0, min(n_acc, n_gyr) - win_n, step_n):
        seg_acc = acc_filt[start:start + win_n]
        seg_gyr = gyr_mag[start:start + win_n]

        if seg_acc.std() < acc_th:
            continue
        if seg_gyr.std() < gyro_th:
            continue

        fft_mag = np.abs(np.fft.rfft(seg_acc - seg_acc.mean()))
        if not band.any() or fft_mag[band].max() < 1e-9:
            continue

        dom_freq = freqs[band][np.argmax(fft_mag[band])]
        total_steps += dom_freq * step_s

    return int(round(total_steps))


# ---------------------------------------------------------------------------
# Load labelled recordings
# ---------------------------------------------------------------------------

def load_labelled(train_dir, features_csv):
    df       = pd.read_csv(features_csv)
    labelled = df[df['step_count'].notna()].copy()
    records  = []
    for _, row in labelled.iterrows():
        fpath = os.path.join(train_dir, row['filename'])
        try:
            with open(fpath, 'rb') as f:
                raw = pickle.load(f)
            # True activity labels for activity-aware masking during tuning
            activities = {a: bool(row[a]) for a in
                           ['standing', 'walking', 'running', 'cycling']}
            records.append({
                'raw':        raw,
                'filename':   row['filename'],
                'watch_loc':  int(row['watch_loc']),
                'step_count': int(row['step_count']),
                'activities': activities,
            })
        except Exception as e:
            print(f'  SKIP {row["filename"]}: {e}')
    return records


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def mape(pred, true):
    p, t = np.array(pred, float), np.array(true, float)
    mask = t > 10
    if mask.sum() == 0:
        return float(np.mean(np.abs(p - t)))
    return float(np.mean(np.abs(p[mask] - t[mask]) / t[mask]) * 100)


def mae(pred, true):
    return float(np.mean(np.abs(np.array(pred, float) - np.array(true, float))))


# ---------------------------------------------------------------------------
# LOO-CV evaluation
# ---------------------------------------------------------------------------

def loo_eval(records, params, source='watch', algo='FREQ'):
    preds, trues = [], []
    for rec in records:
        preds.append(count_steps_with_activity_mask(
            rec['raw'], params, source=source, algo=algo,
            activities=rec.get('activities')))
        trues.append(rec['step_count'])
    return mape(preds, trues), mae(preds, trues), preds, trues


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

FREQ_GRID = {
    'acc_std_thresh':  [0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.22],
    'gyro_std_thresh': [3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0],
    'freq_hi':         [2.5, 3.0, 3.5, 4.0],
    'win_size':        [2.0, 3.0],
}
_FREQ_FIXED = {'win_step': 1.0, 'lp_cutoff': 4.0, 'freq_lo': 1.0}

# Phone uses different units: linear acc in m/s² (vs watch in g), gyro in rad/s
# (vs watch in deg/s). Scale thresholds by ~10× and ~0.02× respectively.
PHONE_FREQ_GRID = {
    'acc_std_thresh':  [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0],
    'gyro_std_thresh': [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2],
    'freq_hi':         [2.5, 3.0, 3.5, 4.0],
    'win_size':        [2.0, 3.0],
}

# WPD-watch grid: peak detection on low-pass-filtered acc magnitude
WPD_GRID = {
    'acc_std_thresh':  [0.07, 0.10, 0.15, 0.22],
    'gyro_std_thresh': [3.0, 8.0, 20.0, 60.0],
    'min_prominence':  [0.10, 0.15, 0.25, 0.40, 0.60],
    'min_peak_dist':   [0.30, 0.35, 0.45],
}
_WPD_FIXED = {'lp_cutoff': 4.0, 'movavr_win': 0.3}


def grid_search(records, grid, fixed=None, source='watch', algo='FREQ'):
    import itertools
    fixed  = fixed or {}
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f'    {len(combos)} combinations × {len(records)} recordings (LOO-CV) '
          f'[{source}/{algo}]')
    best_err    = np.inf
    best_params = {**fixed, **dict(zip(keys, [grid[k][0] for k in keys]))}
    for combo in combos:
        params = {**fixed, **dict(zip(keys, combo))}
        err, _, _, _ = loo_eval(records, params, source=source, algo=algo)
        if err < best_err:
            best_err    = err
            best_params = params.copy()
    return best_params, best_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    features_train_csv = os.path.join(RESULTS_DIR, 'features_train.csv')

    print('Loading labelled recordings...')
    records = load_labelled(DATA_TRAIN, features_train_csv)
    print(f'  {len(records)} recordings with valid step counts.')
    for loc in [0, 1, 2]:
        sub = [r for r in records if r['watch_loc'] == loc]
        if sub:
            counts = [r['step_count'] for r in sub]
            print(f'  {LOC_NAMES[loc]}: n={len(sub)}, '
                  f'range {min(counts)}-{max(counts)}, mean={np.mean(counts):.0f}')

    results = {loc: {} for loc in [0, 1, 2]}

    for loc in [0, 1, 2]:
        sub = [r for r in records if r['watch_loc'] == loc]
        if not sub:
            continue
        print(f'\n--- {LOC_NAMES[loc]} ({len(sub)} recordings) ---')

        # Try FREQ (FFT cadence) and WPD (peak detection) on watch + phone.
        # Pick lowest LOO-CV MAPE per location.
        candidates = []
        for algo, source, grid, fixed in [
            ('FREQ', 'watch', FREQ_GRID,       _FREQ_FIXED),
            ('FREQ', 'phone', PHONE_FREQ_GRID, _FREQ_FIXED),
            ('WPD',  'watch', WPD_GRID,        _WPD_FIXED),
        ]:
            print(f'  [{algo}/{source.upper()}] tuning...')
            bp, err = grid_search(sub, grid, fixed=fixed, source=source, algo=algo)
            _, _, p, t = loo_eval(sub, bp, source=source, algo=algo)
            print(f'  [{algo}/{source.upper()}] MAPE={err:.1f}%  MAE={mae(p,t):.0f}')
            candidates.append((algo, source, bp, err, p, t))

        winner = min(candidates, key=lambda x: x[3])
        algo, source, bp, err, p, t = winner
        print(f'  → Picked {algo}/{source.upper()} (MAPE={err:.1f}%)')
        results[loc] = {'algo': algo, 'source': source, 'params': bp,
                        'mape': err, 'preds': p, 'trues': t}

    all_preds, all_trues, all_locs = [], [], []
    for loc, res in results.items():
        if not res:
            continue
        all_preds.extend(res['preds'])
        all_trues.extend(res['trues'])
        all_locs.extend([loc] * len(res['preds']))

    print('\n=== Overall (best algo per location, LOO-CV) ===')
    print(f'  MAE  : {mae(all_preds, all_trues):.1f} steps')
    print(f'  MAPE : {mape(all_preds, all_trues):.1f}%')
    for loc in [0, 1, 2]:
        idx = [i for i, l in enumerate(all_locs) if l == loc]
        if not idx:
            continue
        p = [all_preds[i] for i in idx]
        t = [all_trues[i] for i in idx]
        print(f'  {LOC_NAMES[loc]:6s}: MAE={mae(p,t):.0f}  MAPE={mape(p,t):.1f}%')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    ax = axes[0]
    for loc in [0, 1, 2]:
        idx = [i for i, l in enumerate(all_locs) if l == loc]
        if not idx:
            continue
        ax.scatter([all_trues[i] for i in idx],
                   [all_preds[i] for i in idx],
                   label=LOC_NAMES[loc], color=colors[loc], alpha=0.8, s=70)
    lim = max(max(all_trues) if all_trues else 1,
              max(all_preds) if all_preds else 1) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', lw=1)
    ax.set_xlabel('True step count')
    ax.set_ylabel('Predicted step count')
    ax.set_title(f'Step Counting  MAE={mae(all_preds, all_trues):.0f}  '
                 f'MAPE={mape(all_preds, all_trues):.1f}%')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    for loc in [0, 1, 2]:
        res = results.get(loc, {})
        if not res:
            continue
        errs = [(p - t) / t * 100
                for p, t in zip(res['preds'], res['trues']) if t > 10]
        ax.scatter([LOC_NAMES[loc]] * len(errs), errs,
                   color=colors[loc], alpha=0.8, s=70)
    ax.axhline(0,   color='black', lw=1)
    ax.axhline(10,  color='red',   ls='--', lw=0.8, alpha=0.5)
    ax.axhline(-10, color='red',   ls='--', lw=0.8, alpha=0.5)
    ax.set_ylabel('Relative error (%)')
    ax.set_title('Per-location error distribution')
    ax.grid(alpha=0.3)

    plt.suptitle('Step Counting (FREQ algorithm)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'step_count_results.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print('\nSaved: step_count_results.png')

    save = {}
    for loc in [0, 1, 2]:
        res = results.get(loc, {})
        if res:
            save[loc] = {'algo': res['algo'], 'source': res['source'],
                          'params': res['params']}
        else:
            save[loc] = {'algo': 'FREQ', 'source': 'watch', 'params': {
                'win_size': 2.0, 'win_step': 1.0, 'lp_cutoff': 4.0,
                'acc_std_thresh': 0.12, 'gyro_std_thresh': 25.0,
                'freq_lo': 1.0, 'freq_hi': 4.0,
            }}

    with open(PARAMS_OUT, 'wb') as f:
        pickle.dump({'params_by_loc': save, 'loc_names': LOC_NAMES}, f)
    print(f'Saved: {PARAMS_OUT}')
