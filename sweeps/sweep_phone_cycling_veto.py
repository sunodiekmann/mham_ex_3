"""
Sweep the phone-IMU cycling-veto threshold for step counting.

Per-window veto in count_steps_freq/wpd: if phone gyro std (rolling 1s window)
< threshold, treat as cycling and skip. Targets wrist+walking+cycling
recordings where pedaling motion is being miscounted as steps.

Reports: LOO-CV competition score (capped relative error) per threshold,
plus delta vs no-veto baseline. Threshold is in rad/s.
"""

import os
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import train_steps as ts
import train_path as tp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_PATH = os.path.join(ROOT, 'models', 'steps_params.pkl')
FEATURES_TRAIN = os.path.join(ROOT, 'results', 'features_train.csv')


def step_count_score(preds, trues):
    p = np.asarray(preds, float)
    t = np.asarray(trues, float)
    rel = np.abs(p - t) / np.maximum(1.0, t)
    return float(1.0 - np.minimum(1.0, rel).mean())


def evaluate(records, params_by_loc, veto_thresh):
    preds, trues, locs = [], [], []
    for rec in records:
        loc = rec['watch_loc']
        e   = params_by_loc[loc]
        params = dict(e['params'])
        if veto_thresh is not None:
            params['phone_cycling_gyro_thresh'] = veto_thresh
        else:
            params.pop('phone_cycling_gyro_thresh', None)
        pred = ts.count_steps_with_activity_mask(
            rec['raw'], params,
            source=e.get('source', 'watch'),
            algo=e.get('algo', 'FREQ'),
            activities=rec['activities'])
        preds.append(pred); trues.append(rec['step_count']); locs.append(loc)
    return preds, trues, locs


def main():
    print('Loading labelled recordings + saved params ...')
    records = ts.load_labelled(tp.DATA_TRAIN, FEATURES_TRAIN)
    with open(PARAMS_PATH, 'rb') as f:
        sc = pickle.load(f)
    params_by_loc = sc['params_by_loc']

    # Baseline (no veto)
    preds_base, trues, locs = evaluate(records, params_by_loc, None)
    score_base = step_count_score(preds_base, trues)
    print(f'\nBaseline (no veto):                  score={score_base:.4f}')

    # Sweep
    print(f'\n{"thresh":>8} {"score":>8} {"Δ":>8} | wrist err  belt err  ankle err  | top errors changed')
    for thresh in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.80]:
        preds, _, _ = evaluate(records, params_by_loc, thresh)

        score = step_count_score(preds, trues)

        # Per-loc capped error contribution
        per_loc = {0: [], 1: [], 2: []}
        for p_b, p_v, t, lc in zip(preds_base, preds, trues, locs):
            per_loc[lc].append((p_b, p_v, t))
        loc_str = []
        for lc in [0, 1, 2]:
            if not per_loc[lc]:
                loc_str.append('   -    '); continue
            base_e = np.mean([min(1.0, abs(b - t) / max(1, t))
                              for b, _, t in per_loc[lc]])
            new_e  = np.mean([min(1.0, abs(v - t) / max(1, t))
                              for _, v, t in per_loc[lc]])
            loc_str.append(f'{base_e:.3f}→{new_e:.3f}')

        # How many recordings changed?
        diff = sum(1 for b, v in zip(preds_base, preds) if b != v)
        print(f'{thresh:>8.2f} {score:>8.4f} {score - score_base:>+8.4f} | '
              f'{loc_str[0]}  {loc_str[1]}  {loc_str[2]} | {diff} changed')


if __name__ == '__main__':
    main()
