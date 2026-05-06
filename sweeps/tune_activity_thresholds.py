"""
Examine distributions of acc_std, gyro_std per activity-true/false
to find better segment-detection thresholds.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = os.path.join(ROOT, 'results', 'features_train.csv')

df = pd.read_csv(FEATURES)
print(f'n={len(df)}\n')

# Check the windowed stats medians per activity
print('=== Per-activity median stats (watch acc/gyro) ===')
print(f'{"Activity":<15} {"n":>5} {"acc_std_p50":>12} {"gyro_std_p50":>12} '
      f'{"acc_domfreq_p50":>16}')
for act in ['standing', 'walking', 'running', 'cycling']:
    for val in (True, False):
        sub = df[df[act] == val]
        print(f'  {act + ("=T" if val else "=F"):<13} {len(sub):>5d} '
              f'{sub["acc_std_p50"].median():>12.3f} '
              f'{sub["gyro_std_p50"].median():>12.3f} '
              f'{sub["acc_domfreq_p50"].median():>16.3f}')
    print()

# For cycling specifically: look at distribution to pick threshold
print('\n=== Cycling: acc_std_p50 and gyro_std_p50 distributions ===')
cyc_T = df[df['cycling'] == True]
cyc_F = df[df['cycling'] == False]
print(f'cycling=True   (n={len(cyc_T)}): acc_std_p50 '
      f'q25={cyc_T["acc_std_p50"].quantile(0.25):.3f} '
      f'q50={cyc_T["acc_std_p50"].quantile(0.50):.3f} '
      f'q75={cyc_T["acc_std_p50"].quantile(0.75):.3f}'
      f'  |  gyro_std_p50 '
      f'q25={cyc_T["gyro_std_p50"].quantile(0.25):.3f} '
      f'q50={cyc_T["gyro_std_p50"].quantile(0.50):.3f} '
      f'q75={cyc_T["gyro_std_p50"].quantile(0.75):.3f}')
print(f'cycling=False  (n={len(cyc_F)}): acc_std_p50 '
      f'q25={cyc_F["acc_std_p50"].quantile(0.25):.3f} '
      f'q50={cyc_F["acc_std_p50"].quantile(0.50):.3f} '
      f'q75={cyc_F["acc_std_p50"].quantile(0.75):.3f}'
      f'  |  gyro_std_p50 '
      f'q25={cyc_F["gyro_std_p50"].quantile(0.25):.3f} '
      f'q50={cyc_F["gyro_std_p50"].quantile(0.50):.3f} '
      f'q75={cyc_F["gyro_std_p50"].quantile(0.75):.3f}')

# Check phone stats separately
print('\n=== Per-activity median stats (PHONE acc/gyro) ===')
print(f'{"Activity":<15} {"n":>5} {"phone_acc_std":>14} {"phone_gyro_std":>14} '
      f'{"phone_acc_df":>14}')
for act in ['standing', 'walking', 'running', 'cycling']:
    for val in (True, False):
        sub = df[df[act] == val]
        print(f'  {act + ("=T" if val else "=F"):<13} {len(sub):>5d} '
              f'{sub["phone_acc_std_p50"].median():>14.3f} '
              f'{sub["phone_gyro_std_p50"].median():>14.3f} '
              f'{sub["phone_acc_domfreq_p50"].median():>14.3f}')
    print()
