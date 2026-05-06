"""
Verify phone_orientationx availability + that it's compass yaw (azimuth).

Tests:
  1. How many recordings have phone_orientation?
  2. Is phone_orientationx in [0, 360] (azimuth) or [-pi, pi] (radians)?
  3. Compare phone_orientationx with atan2(phone_my, phone_mx) on a few samples.
  4. Fallback availability: phone_pressure vs altitude.
"""

import os
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TRAIN = os.path.join(ROOT, 'data', 'train')
DATA_TEST  = os.path.join(ROOT, 'data', 'test')


def inspect_keys(directory):
    files = sorted(f for f in os.listdir(directory) if f.endswith('.pkl'))
    has_orient = 0
    has_pressure = 0
    has_altitude = 0
    has_neither = 0
    has_mx = 0

    orient_ranges = []   # collect (min, max) per recording
    for fname in files:
        with open(os.path.join(directory, fname), 'rb') as f:
            raw = pickle.load(f)
        keys = raw['data'].keys()
        if 'phone_orientationx' in keys:
            has_orient += 1
            v = np.array(raw['data']['phone_orientationx']['values'])
            if len(v) > 0:
                orient_ranges.append((v.min(), v.max(), v.mean()))
        if 'phone_pressure' in keys:
            has_pressure += 1
        if 'altitude' in keys:
            has_altitude += 1
        if 'phone_pressure' not in keys and 'altitude' not in keys:
            has_neither += 1
        if 'phone_mx' in keys:
            has_mx += 1
    print(f'  Files                : {len(files)}')
    print(f'  has phone_orientation: {has_orient} / {len(files)}')
    print(f'  has phone_pressure   : {has_pressure} / {len(files)}')
    print(f'  has altitude (GPS)   : {has_altitude} / {len(files)}')
    print(f'  has neither pressure nor altitude: {has_neither}')
    print(f'  has phone_mx         : {has_mx} / {len(files)}')
    return orient_ranges


print('=== Train set ===')
train_ranges = inspect_keys(DATA_TRAIN)

print('\n=== Test set ===')
test_ranges = inspect_keys(DATA_TEST)

# Range stats
all_ranges = train_ranges + test_ranges
if all_ranges:
    mins = [r[0] for r in all_ranges]
    maxs = [r[1] for r in all_ranges]
    means = [r[2] for r in all_ranges]
    print(f'\nphone_orientationx ranges across all recordings:')
    print(f'  Per-recording min:  median={np.median(mins):.2f}, '
          f'p5={np.percentile(mins, 5):.2f}, p95={np.percentile(mins, 95):.2f}')
    print(f'  Per-recording max:  median={np.median(maxs):.2f}, '
          f'p5={np.percentile(maxs, 5):.2f}, p95={np.percentile(maxs, 95):.2f}')
    print(f'  Per-recording mean: median={np.median(means):.2f}')

    # Detect units: degrees [0, 360] vs radians [-pi, pi]
    if np.percentile(maxs, 95) > 7:
        print('  → looks like DEGREES [0, 360] (azimuth)')
    else:
        print('  → looks like RADIANS [-pi, pi]')

# Compare phone_orientationx to atan2(my, mx) on a sample
print('\n=== Sample comparison: phone_orientationx vs atan2(my, mx) ===')
files = sorted(f for f in os.listdir(DATA_TRAIN) if f.endswith('.pkl'))[:3]
for fname in files:
    with open(os.path.join(DATA_TRAIN, fname), 'rb') as f:
        raw = pickle.load(f)
    if 'phone_orientationx' not in raw['data'] or 'phone_mx' not in raw['data']:
        continue
    ox = np.array(raw['data']['phone_orientationx']['values'])
    mx = np.array(raw['data']['phone_mx']['values'])
    my = np.array(raw['data']['phone_my']['values'])
    n = min(len(ox), len(mx), len(my))
    if n < 100:
        continue
    yaw_calc = np.degrees(np.arctan2(my[:n], mx[:n])) % 360
    print(f'\n{fname}:')
    # Sample at 5 evenly-spaced indices
    idx = np.linspace(0, n-1, 5).astype(int)
    print(f'  idx     phone_orientationx   atan2(my,mx)')
    for i in idx:
        print(f'  {i:6d}  {ox[i]:>16.2f}     {yaw_calc[i]:>10.2f}')
    print(f'  ranges: orient [{ox.min():.1f}, {ox.max():.1f}]  '
          f'atan2 [{yaw_calc.min():.1f}, {yaw_calc.max():.1f}]')
