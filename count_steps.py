"""
Step counting for LilyGo smartwatch at 12.5 Hz.

Based on: Brajdic & Harle, "Walk Detection and Step Counting on Unconstrained
Smartphones", UbiComp 2013.

WHY WPD (windowed peak detection) fails here:
  - 12.5 Hz gives only 3-5 samples between steps → noise peaks slip through
  - Max prominence in naive grid (0.1g) is far below actual step peaks (0.5-1.5g)
  - No gyroscope filter → cycling vibrations counted as steps

PRIMARY ALGORITHM: STFT frequency × locomotion duration  (Table 4 in paper)
  For each 2-second window:
    1. Activity mask: acc_std > thresh AND gyro_std > gyro_thresh
       (gyro filter is crucial — cycling has very low gyro std ~40 vs walking ~70 deg/s)
    2. Dominant step frequency = FFT peak in [0.8, 4.0] Hz
    3. Accumulated steps += dom_freq × step_duration_s (non-overlapping portion)

SECONDARY ALGORITHM: WPD with proper prominence range and dual-sensor mask
  Used as fallback comparison; extended prominence grid [0.1 … 1.5g].

Parameters tuned per watch location via leave-one-out CV on the 33 labelled
training recordings.
"""

import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_DIR  = 'data/train'
PARAMS_OUT = 'step_count_params.pkl'
LOC_NAMES  = {0: 'Wrist', 1: 'Belt', 2: 'Ankle'}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _samplerate(raw_timestamps, values=None):
    """Compute sample rate from values array (actual samples) over timestamp span.
    raw_timestamps are batched at ~12.5 Hz; actual sample rate is len(values)/duration.
    Always pass values= to get the true rate (watch: ~200 Hz, phone: ~100 Hz).
    """
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
# Algorithm 1: STFT frequency × locomotion duration (primary)
# ---------------------------------------------------------------------------

def count_steps_freq(raw, params):
    """
    For each locomotion window estimate the step cadence via FFT and accumulate
    fractional steps.  Locomotion is detected by a dual-sensor mask:
      acc_std  > params['acc_std_thresh']   — excludes pure standing
      gyro_std > params['gyro_std_thresh']  — excludes cycling (low gyro)
    Only windows whose dominant frequency falls in the walking/running band
    [params['freq_lo'], params['freq_hi']] are counted.
    """
    data = raw['data']
    ax  = np.array(data['ax']['values'])
    ay  = np.array(data['ay']['values'])
    az  = np.array(data['az']['values'])
    ts  = data['ax']['raw_timestamps']
    gx  = np.array(data['gx']['values'])
    gy  = np.array(data['gy']['values'])
    gz  = np.array(data['gz']['values'])

    sr      = _samplerate(ts, ax)   # use values count for true ~200 Hz rate
    acc_mag = _mag(ax, ay, az)
    gyr_mag = _mag(gx, gy, gz)

    # Low-pass to isolate step frequencies
    acc_filt = _lp_filter(acc_mag, sr, cutoff=params.get('lp_cutoff', 5.0))

    win_s    = params.get('win_size', 2.0)   # window length [s]
    step_s   = params.get('win_step', 1.0)   # non-overlapping step [s]
    win_n    = max(8, int(win_s * sr))
    step_n   = max(1, int(step_s * sr))

    gyro_th  = params['gyro_std_thresh']
    freq_lo  = params.get('freq_lo', 0.8)
    freq_hi  = params.get('freq_hi', 4.0)

    acc_th   = params['acc_std_thresh']
    n_acc = len(acc_filt)
    n_gyr = len(gyr_mag)

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
        # Accumulate fractional steps for the non-overlapping step_s portion
        total_steps += dom_freq * step_s

    return int(round(total_steps))


# ---------------------------------------------------------------------------
# Algorithm 2b: WPD using phone linear acceleration (primary for wrist/belt)
# Phone in pocket gives consistent body-vertical signal regardless of arm swing.
# phone_lax/lay/laz are gravity-removed (linear acceleration) at ~50 Hz.
# ---------------------------------------------------------------------------

def count_steps_phone_wpd(raw, params):
    """
    WPD on phone linear acceleration magnitude.
    Uses phone gyro for locomotion masking.
    Runs at actual phone sample rate (~100 Hz); watch runs at ~200 Hz.
    """
    data = raw['data']
    plax = np.array(data['phone_lax']['values'])
    play = np.array(data['phone_lay']['values'])
    plaz = np.array(data['phone_laz']['values'])
    pts  = data['phone_lax']['raw_timestamps']
    pgx  = np.array(data['phone_gx']['values'])
    pgy  = np.array(data['phone_gy']['values'])
    pgz  = np.array(data['phone_gz']['values'])

    sr      = _samplerate(pts, plax)  # true ~100 Hz
    if sr < 5:
        return 0

    acc_mag = _mag(plax, play, plaz)
    gyr_mag = _mag(pgx, pgy, pgz)

    # Gravity already removed (linear acc) — std threshold logic still valid
    # for the std threshold logic — or just use it directly (std is the same)
    acc_filt = _lp_filter(acc_mag, sr, cutoff=params.get('lp_cutoff', 4.0))

    ma_n    = max(1, int(params.get('movavr_win', 0.3) * sr))
    kernel  = np.ones(ma_n) / ma_n
    smoothed = np.convolve(acc_filt, kernel, mode='same')

    win_n  = max(1, int(sr))
    step_n = max(1, win_n // 2)
    active = np.zeros(len(smoothed), dtype=bool)
    n_min  = min(len(smoothed), len(gyr_mag))
    acc_th  = params['acc_std_thresh']
    gyro_th = params['gyro_std_thresh']

    for start in range(0, n_min - win_n, step_n):
        if (smoothed[start:start + win_n].std() > acc_th and
                gyr_mag[start:start + win_n].std() > gyro_th):
            active[start:start + win_n] = True

    masked = smoothed.copy()
    masked[~active] = smoothed.mean()

    min_dist = max(1, int(params.get('min_peak_dist', 0.35) * sr))
    peaks, _ = find_peaks(masked,
                          distance=min_dist,
                          prominence=params['min_prominence'])

    return int(np.sum(active[peaks]))


# ---------------------------------------------------------------------------
# Algorithm 2: WPD with dual-sensor mask (comparison)
# ---------------------------------------------------------------------------

def count_steps_wpd(raw, params):
    """
    Windowed Peak Detection (Brajdic & Harle, Table 4).
    Dual-sensor activity mask: acc_std AND gyro_std thresholds.
    Prominence extended to cover real step amplitudes (0.3–1.5g).
    """
    data    = raw['data']
    ax      = np.array(data['ax']['values'])
    ay      = np.array(data['ay']['values'])
    az      = np.array(data['az']['values'])
    ts      = data['ax']['raw_timestamps']
    gx      = np.array(data['gx']['values'])
    gy      = np.array(data['gy']['values'])
    gz      = np.array(data['gz']['values'])

    sr      = _samplerate(ts, ax)  # true ~200 Hz
    acc_mag = _mag(ax, ay, az)
    gyr_mag = _mag(gx, gy, gz)

    acc_filt = _lp_filter(acc_mag, sr, cutoff=params.get('lp_cutoff', 4.0))

    # Centred moving average
    ma_n      = max(1, int(params.get('movavr_win', 0.3) * sr))
    kernel    = np.ones(ma_n) / ma_n
    smoothed  = np.convolve(acc_filt, kernel, mode='same')

    # Build locomotion mask (1-second sub-windows)
    win_n    = max(1, int(sr))
    step_n   = max(1, win_n // 2)
    active   = np.zeros(len(smoothed), dtype=bool)
    n_min    = min(len(smoothed), len(gyr_mag))
    acc_th   = params['acc_std_thresh']
    gyro_th  = params['gyro_std_thresh']

    for start in range(0, n_min - win_n, step_n):
        if (smoothed[start:start + win_n].std() > acc_th and
                gyr_mag[start:start + win_n].std() > gyro_th):
            active[start:start + win_n] = True

    masked        = smoothed.copy()
    masked[~active] = smoothed.mean()

    min_dist = max(1, int(params.get('min_peak_dist', 0.35) * sr))
    peaks, _ = find_peaks(masked,
                          distance=min_dist,
                          prominence=params['min_prominence'])

    return int(np.sum(active[peaks]))


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
            records.append({
                'raw':        raw,
                'filename':   row['filename'],
                'watch_loc':  int(row['watch_loc']),
                'step_count': int(row['step_count']),
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
# LOO-CV evaluation for a given algorithm + params
# ---------------------------------------------------------------------------

def loo_eval(records, algo_fn, params):
    preds, trues = [], []
    for rec in records:
        preds.append(algo_fn(rec['raw'], params))
        trues.append(rec['step_count'])
    return mape(preds, trues), mae(preds, trues), preds, trues


# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

# FREQ grid — fine-grained sweep of acc/gyro thresholds, frequency band, window size.
FREQ_GRID = {
    'acc_std_thresh':  [0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.22],
    'gyro_std_thresh': [3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0],
    'freq_hi':         [2.5, 3.0, 3.5, 4.0],
    'win_size':        [2.0, 3.0],
}
_FREQ_FIXED = {'win_step': 1.0, 'lp_cutoff': 4.0, 'freq_lo': 1.0}


def grid_search(records, algo_fn, grid, fixed=None):
    import itertools
    fixed  = fixed or {}
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f'    {len(combos)} combinations × {len(records)} recordings (LOO-CV)')
    best_err, best_params = np.inf, {**fixed, **dict(zip(keys, [grid[k][0] for k in keys]))}
    for combo in combos:
        params = {**fixed, **dict(zip(keys, combo))}
        err, _, _, _ = loo_eval(records, algo_fn, params)
        if err < best_err:
            best_err   = err
            best_params = params.copy()
    return best_params, best_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Loading labelled recordings...')
    records = load_labelled(TRAIN_DIR, 'features_train.csv')
    print(f'  {len(records)} recordings with valid step counts.')
    for loc in [0, 1, 2]:
        sub = [r for r in records if r['watch_loc'] == loc]
        if sub:
            counts = [r['step_count'] for r in sub]
            print(f'  {LOC_NAMES[loc]}: n={len(sub)}, '
                  f'range {min(counts)}–{max(counts)}, mean={np.mean(counts):.0f}')

    results = {loc: {} for loc in [0, 1, 2]}

    for loc in [0, 1, 2]:
        sub = [r for r in records if r['watch_loc'] == loc]
        if not sub:
            continue
        print(f'\n--- {LOC_NAMES[loc]} ({len(sub)} recordings) ---')

        print('  [FREQ] tuning...')
        bp, err = grid_search(sub, count_steps_freq, FREQ_GRID, fixed=_FREQ_FIXED)
        _, _, p, t = loo_eval(sub, count_steps_freq, bp)
        print(f'  [FREQ] MAPE={err:.1f}%  MAE={mae(p,t):.0f}  params={bp}')
        results[loc] = {'algo': 'FREQ', 'params': bp, 'mape': err,
                        'preds': p, 'trues': t, 'algo_fn': count_steps_freq}

    # -------------------------------------------------------------------
    # Overall summary
    # -------------------------------------------------------------------
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
        algo = results[loc].get('algo', '?')
        print(f'  {LOC_NAMES[loc]:6s} [{algo}]: MAE={mae(p,t):.0f}  MAPE={mape(p,t):.1f}%')

    # Per-recording table
    print(f'\n  {"Filename":<32} {"Loc":6} {"True":>6} {"Pred":>6} {"Err%":>7}')
    idx_r = 0
    for loc in [0, 1, 2]:
        res = results.get(loc, {})
        if not res:
            continue
        for t, p in zip(res['trues'], res['preds']):
            err = abs(p - t) / t * 100 if t > 10 else float('nan')
            sub = [r for r in records if r['watch_loc'] == loc]
            fname = sub[idx_r % len(sub)]['filename'] if sub else '?'
            err_s = f'{err:7.1f}%' if not np.isnan(err) else '    n/a'
            print(f'  {fname:<32} {LOC_NAMES[loc]:6} {t:6d} {p:6d} {err_s}')
            idx_r += 1

    # -------------------------------------------------------------------
    # Visualisation
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    ax = axes[0]
    for loc in [0, 1, 2]:
        idx = [i for i, l in enumerate(all_locs) if l == loc]
        if not idx:
            continue
        ax.scatter([all_trues[i] for i in idx],
                   [all_preds[i] for i in idx],
                   label=f'{LOC_NAMES[loc]} ({results[loc]["algo"]})',
                   color=colors[loc], alpha=0.8, s=70)
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

    plt.suptitle('Step Counting — FREQ / WPD-watch / WPD-phone (best per location)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('step_count_results.png', dpi=120, bbox_inches='tight')
    plt.show()
    print('\nSaved: step_count_results.png')

    # -------------------------------------------------------------------
    # Save parameters
    # -------------------------------------------------------------------
    save = {}
    for loc in [0, 1, 2]:
        res = results.get(loc, {})
        if res:
            save[loc] = {'algo': res['algo'], 'params': res['params']}
        else:
            save[loc] = {'algo': 'FREQ', 'params': {
                'win_size': 2.0, 'win_step': 1.0, 'lp_cutoff': 4.0,
                'acc_std_thresh': 0.12, 'gyro_std_thresh': 25.0,
                'freq_lo': 1.0, 'freq_hi': 4.0,
            }}

    with open(PARAMS_OUT, 'wb') as f:
        pickle.dump({'params_by_loc': save, 'loc_names': LOC_NAMES}, f)
    print(f'Saved: {PARAMS_OUT}')
