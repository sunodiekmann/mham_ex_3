"""
Sanity check: extract turn sequences from real recordings and verify they
match the hand-derived expectations from the user:
  Path 0: ~[+150, -90, +90] right, left, right (3 turns)
  Path 1: ~[+150, -90, +90, -90, +90] adds a Clausiusstrasse loop (5 turns)
  Path 2: ~[+80, -90, +90] different start (3 turns, smaller first turn)
"""

import os
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from features import _extract_turn_sequence, _samplerate

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TRAIN = os.path.join(ROOT, 'data', 'train')


def main():
    by_path = {k: [] for k in range(5)}
    for fname in sorted(os.listdir(DATA_TRAIN)):
        if not fname.endswith('.pkl'):
            continue
        with open(os.path.join(DATA_TRAIN, fname), 'rb') as f:
            raw = pickle.load(f)
        pmx = np.array(raw['data']['phone_mx']['values'], dtype=float)
        pmy = np.array(raw['data']['phone_my']['values'], dtype=float)
        n = min(len(pmx), len(pmy))
        if n < 30:
            continue
        heading = np.degrees(np.arctan2(pmy[:n], pmx[:n])) % 360
        sr = _samplerate(raw['data']['phone_mx']['raw_timestamps'], n)
        turns = _extract_turn_sequence(heading, sr=sr)
        by_path[int(raw['labels']['path_idx'])].append(turns)

    print('Expected (user description):')
    print('  Path 0: [+150, -90, +90]  (3 turns)')
    print('  Path 1: [+150, -90, +90, -90, +90]  (5 turns)')
    print('  Path 2: [+80, -90, +90]  (3 turns)\n')

    for p in range(5):
        n_turns_list = [len(t) for t in by_path[p]]
        # Distribution of first turn magnitude
        first_turns = [t[0] for t in by_path[p] if len(t) > 0]
        # Pad and aggregate to show typical pattern
        padded = np.zeros((len(by_path[p]), 6))
        for i, t in enumerate(by_path[p]):
            for j, v in enumerate(t[:6]):
                padded[i, j] = v
        median = np.median(padded, axis=0)
        mean   = np.mean(padded, axis=0)
        print(f'Path {p}: n={len(by_path[p])}')
        print(f'  turn-count median={int(np.median(n_turns_list))}, '
              f'p25={int(np.percentile(n_turns_list, 25))}, '
              f'p75={int(np.percentile(n_turns_list, 75))}')
        print(f'  first-turn median={np.median(first_turns):+.1f}°, '
              f'mean={np.mean(first_turns):+.1f}°'
              if first_turns else '  no turns found')
        print(f'  median turn sequence: '
              f'[{", ".join(f"{v:+.0f}" for v in median)}]')
        print(f'  mean   turn sequence: '
              f'[{", ".join(f"{v:+.0f}" for v in mean)}]')
        print()


if __name__ == '__main__':
    main()
