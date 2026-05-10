"""
Wrist-only step-counter source comparison.

The current production winner for wrist (loc=0) is WPD/watch. Wrist watch sees
arm-swing — noisy, walking-cycling-mixed recordings cause the biggest LOO
errors. Phone-in-pocket sees clean foot-strike on every watch_loc.

This sweep retunes the wrist subset across:
  - FREQ/phone   (existing PHONE_FREQ_GRID)
  - WPD/phone    (new — phone units: linear acc m/s², gyro rad/s)
  - WPD/watch    (existing wrist baseline, for comparison)
  - FREQ/watch   (also tried for completeness)

Scoring: competition step-count score (capped relative error) AND MAPE.
The production grid search uses MAPE, which can disagree with competition
score on the wrist subset where over/under-counts get capped differently.

Outputs the best params per algorithm/source and a recommendation.
"""

import itertools
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import train_steps as ts
import train_path as tp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_TRAIN = os.path.join(ROOT, 'results', 'features_train.csv')

# WPD grid for phone (m/s² for linear acc, rad/s for gyro)
PHONE_WPD_GRID = {
    'acc_std_thresh':  [0.5, 1.0, 1.5, 2.0],
    'gyro_std_thresh': [0.05, 0.1, 0.2, 0.3, 0.5],
    'min_prominence':  [0.5, 1.0, 1.5, 2.5, 4.0],
    'min_peak_dist':   [0.30, 0.35, 0.45],
}
_WPD_FIXED = {'lp_cutoff': 4.0, 'movavr_win': 0.3}


def step_count_score(preds, trues):
    p, t = np.asarray(preds, float), np.asarray(trues, float)
    rel = np.abs(p - t) / np.maximum(1.0, t)
    return float(1.0 - np.minimum(1.0, rel).mean())


def grid_search_score(records, grid, fixed, source, algo, score_fn):
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    best = (-np.inf, None, None, None, None)
    for combo in combos:
        params = {**(fixed or {}), **dict(zip(keys, combo))}
        preds, trues = [], []
        for rec in records:
            p = ts.count_steps_with_activity_mask(
                rec['raw'], params, source=source, algo=algo,
                activities=rec.get('activities'))
            preds.append(p); trues.append(rec['step_count'])
        s = score_fn(preds, trues)
        m = ts.mape(preds, trues)
        if s > best[0]:
            best = (s, m, params.copy(), preds, trues)
    return best


def main():
    print('Loading labelled recordings ...')
    records = ts.load_labelled(tp.DATA_TRAIN, FEATURES_TRAIN)
    wrist = [r for r in records if r['watch_loc'] == 0]
    print(f'  {len(wrist)} wrist-labelled recordings')

    candidates = [
        ('FREQ', 'watch', ts.FREQ_GRID,        ts._FREQ_FIXED),
        ('FREQ', 'phone', ts.PHONE_FREQ_GRID,  ts._FREQ_FIXED),
        ('WPD',  'watch', ts.WPD_GRID,         _WPD_FIXED),
        ('WPD',  'phone', PHONE_WPD_GRID,      _WPD_FIXED),
    ]

    print(f'\n{"algo":<6}{"source":<8}{"score":>10}{"mape":>10}  best_params')
    print('-' * 80)
    results = []
    for algo, source, grid, fixed in candidates:
        score, mape_v, params, preds, trues = grid_search_score(
            wrist, grid, fixed, source, algo, step_count_score)
        keep = {k: v for k, v in params.items() if k in grid}
        print(f'{algo:<6}{source:<8}{score:>10.4f}{mape_v:>10.1f}  {keep}')
        results.append((algo, source, score, mape_v, params, preds, trues))

    print()
    winner = max(results, key=lambda x: x[2])
    algo, source, score, mape_v, params, preds, trues = winner
    print(f'Recommended for wrist: {algo}/{source}  score={score:.4f}  MAPE={mape_v:.1f}%')
    print('Params:')
    for k, v in params.items():
        print(f'  {k}: {v}')

    # Per-recording delta vs current production (WPD/watch)
    print('\nWrist per-recording predictions (winner vs current WPD/watch):')
    cur_algo, cur_src, _, _, cur_params, cur_preds, _ = next(
        r for r in results if r[0] == 'WPD' and r[1] == 'watch')
    for i, rec in enumerate(wrist):
        a_str = ','.join(k for k, v in rec['activities'].items() if v)
        cur_e = min(1.0, abs(cur_preds[i] - trues[i]) / max(1, trues[i]))
        new_e = min(1.0, abs(preds[i] - trues[i]) / max(1, trues[i]))
        marker = '↓' if new_e < cur_e - 0.005 else ('↑' if new_e > cur_e + 0.005 else '·')
        print(f'  {rec["filename"]:<24} true={trues[i]:>5}  '
              f'cur={cur_preds[i]:>5} (e={cur_e:.3f})  '
              f'new={preds[i]:>5} (e={new_e:.3f})  {marker}  [{a_str}]')


if __name__ == '__main__':
    main()
