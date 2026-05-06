"""
Elevation profile matching for path classification.

Compares each recording's elevation profile (from phone pressure or GPS
altitude) against the 5 GPX-derived path templates using multiple distance
metrics.

Why this works where the trajectory approach struggles:
  - Pressure / altitude are reliable scalars (no phone-pocket-orientation issue)
  - Each path has a distinctive elevation signature:
      P0/P1: +39m total, gradual + (P1) Clausiusstrasse stair
      P2:    +54m total, long flat start + Walchetor stairs
      P3:    -54m total (reverse of P2), early Walchetor descent
      P4:    -49m total, different mid-route descent
  - DTW handles speed variation
  - Both signed (magnitude-preserving) and shape-normalised distances are
    emitted — the classifier picks whichever discriminates the case at hand.
"""

import os
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPX_DIR = os.path.join(ROOT, 'data', 'path_gpx')
PRESSURE_TO_METRES = -8.3
N_PATHS    = 5
N_RESAMPLE = 100        # resample curves to fixed length for distance metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resample_curve(values, n_target=N_RESAMPLE):
    """Resample 1D curve to n_target points (linear interp on [0, 1])."""
    n = len(values)
    if n < 2:
        return np.full(n_target, np.nan)
    x = np.linspace(0, 1, n)
    target = np.linspace(0, 1, n_target)
    return np.interp(target, x, values)


def _normalize_shape(elev):
    """Centre at start, divide by absolute span. Output in roughly [-1, 1]."""
    rel = elev - elev[0]
    span = max(abs(rel).max(), 0.1)
    return rel / span


def _dtw(seq1, seq2, max_warp=0.3):
    """DTW distance, length-normalised. Always reachable: band >= |n1 - n2|+3."""
    n1, n2 = len(seq1), len(seq2)
    if n1 == 0 or n2 == 0:
        return float('inf')
    band = max(int(max(n1, n2) * max_warp), abs(n1 - n2) + 3)
    cost = np.full((n1 + 1, n2 + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n1 + 1):
        j_lo = max(1, i - band)
        j_hi = min(n2, i + band)
        for j in range(j_lo, j_hi + 1):
            d = abs(seq1[i - 1] - seq2[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n1, n2] / max(n1, n2))


def _slope_features(elev_resampled, total_change):
    """
    Position-of-steepest, flat-fraction etc. on the resampled elevation curve.
    These directly capture path-distinguishing structure (e.g. P2 has a long
    flat section at the start, P0 has a uniform climb).
    """
    feats = {}
    diffs = np.diff(elev_resampled)
    abs_diffs = np.abs(diffs)
    if len(diffs) > 0 and abs_diffs.max() > 0:
        feats['elev_steepest_pos']  = float(np.argmax(abs_diffs) / len(diffs))
        feats['elev_steepest_mag']  = float(abs_diffs.max())
        # Flat fraction: per-step elevation change < 0.05 of total span
        flat_thresh = max(0.005, 0.05 * abs(total_change) / len(diffs))
        feats['elev_flat_frac']     = float((abs_diffs < flat_thresh).mean())
        # Where in the curve does the most climb happen? (cum-of-abs argmax of cumsum / total)
        cum = np.cumsum(abs_diffs)
        cum_norm = cum / max(cum[-1], 1e-9)
        # Position of 50% cumulative climb (median climb position)
        feats['elev_climb_p50_pos'] = float(np.searchsorted(cum_norm, 0.5) / len(diffs))
        feats['elev_climb_p25_pos'] = float(np.searchsorted(cum_norm, 0.25) / len(diffs))
        feats['elev_climb_p75_pos'] = float(np.searchsorted(cum_norm, 0.75) / len(diffs))
    else:
        for k in ('elev_steepest_pos', 'elev_steepest_mag', 'elev_flat_frac',
                  'elev_climb_p50_pos', 'elev_climb_p25_pos', 'elev_climb_p75_pos'):
            feats[k] = np.nan
    return feats


# ---------------------------------------------------------------------------
# Templates (built once from GPX)
# ---------------------------------------------------------------------------

def build_elevation_templates(gpx_dir=GPX_DIR):
    """For each path, resample elevation profile + cache shape statistics."""
    templates = {}
    for p in range(N_PATHS):
        path_file = os.path.join(gpx_dir, f'path{p}.csv')
        if not os.path.exists(path_file):
            continue
        df = pd.read_csv(path_file)
        elev = df['elevation_m'].values.astype(float)
        # Light smoothing on GPX (already low-resolution)
        elev_smooth = uniform_filter1d(elev, size=3, mode='nearest')
        templates[p] = {
            'raw_resampled':   _resample_curve(elev_smooth - elev_smooth[0]),
            'shape_resampled': _resample_curve(_normalize_shape(elev_smooth)),
            'total_change':    float(elev_smooth[-1] - elev_smooth[0]),
            'abs_total':       float(np.abs(np.diff(elev_smooth)).sum()),
        }
    return templates


# ---------------------------------------------------------------------------
# Recording-side extraction
# ---------------------------------------------------------------------------

def extract_recording_elevation(raw):
    """
    Extract elevation series (m above start) from sensors.

    Pressure preferred (high res, low noise after smoothing). Falls back to
    GPS altitude for the ~16 % of recordings missing pressure.
    """
    data = raw['data']
    if 'phone_pressure' in data and len(data['phone_pressure']['values']) > 10:
        prs = np.array(data['phone_pressure']['values'], dtype=float)
        # Heavy smooth: ~5 s window at 5 Hz = 25 samples
        prs_smooth = uniform_filter1d(prs, size=max(3, min(25, len(prs) // 2)),
                                        mode='nearest')
        return (prs_smooth - prs_smooth[0]) * PRESSURE_TO_METRES
    elif 'altitude' in data and len(data['altitude']['values']) > 10:
        alt = np.array(data['altitude']['values'], dtype=float)
        alt_smooth = uniform_filter1d(alt, size=max(3, min(5, len(alt) // 2)),
                                        mode='nearest')
        return alt_smooth - alt_smooth[0]
    return np.array([])


# ---------------------------------------------------------------------------
# Match a recording against all 5 templates
# ---------------------------------------------------------------------------

def match_recording_elevation(raw, templates):
    """
    Distance features comparing recording elevation profile to each template.

    Emitted features per template (5 templates → 4×5 = 20 distance features):
      elev_l2_raw_p{p}    : L2 on raw signed elevation (magnitude-aware)
      elev_l2_shape_p{p}  : L2 on shape-normalised elevation (magnitude-blind)
      elev_corr_p{p}      : Pearson correlation with template shape
      elev_dtw_p{p}       : DTW on shape-normalised elevation

    Plus aggregate features:
      elev_total_change, elev_abs_total
      elev_best_match_l2, elev_best_match_corr
      slope/position descriptors (6 features)
    """
    feats = {}
    elev = extract_recording_elevation(raw)
    keys_dist = ([f'elev_l2_raw_p{p}'   for p in range(N_PATHS)] +
                  [f'elev_l2_shape_p{p}' for p in range(N_PATHS)] +
                  [f'elev_corr_p{p}'     for p in range(N_PATHS)] +
                  [f'elev_dtw_p{p}'      for p in range(N_PATHS)])

    if len(elev) < 10:
        for k in keys_dist:
            feats[k] = np.nan
        feats.update({
            'elev_best_match_l2':   -1,
            'elev_best_match_corr': -1,
            'elev_total_change':    0.0,
            'elev_abs_total':       0.0,
        })
        feats.update(_slope_features(np.zeros(N_RESAMPLE), 0.0))
        return feats

    raw_resampled   = _resample_curve(elev - elev[0])
    shape_resampled = _resample_curve(_normalize_shape(elev))
    total_change    = float(elev[-1] - elev[0])

    feats['elev_total_change'] = total_change
    feats['elev_abs_total']    = float(np.abs(np.diff(elev)).sum())

    for p, tmpl in templates.items():
        feats[f'elev_l2_raw_p{p}']   = float(np.sqrt(np.mean(
            (raw_resampled - tmpl['raw_resampled']) ** 2)))
        feats[f'elev_l2_shape_p{p}'] = float(np.sqrt(np.mean(
            (shape_resampled - tmpl['shape_resampled']) ** 2)))
        # Correlation (only meaningful with non-trivial variation)
        if np.std(shape_resampled) > 1e-6 and np.std(tmpl['shape_resampled']) > 1e-6:
            feats[f'elev_corr_p{p}'] = float(np.corrcoef(
                shape_resampled, tmpl['shape_resampled'])[0, 1])
        else:
            feats[f'elev_corr_p{p}'] = 0.0
        feats[f'elev_dtw_p{p}'] = _dtw(shape_resampled, tmpl['shape_resampled'])

    # Best-match indices
    l2_d   = {p: feats[f'elev_l2_raw_p{p}']  for p in range(N_PATHS)}
    corr_d = {p: feats[f'elev_corr_p{p}']    for p in range(N_PATHS)}
    feats['elev_best_match_l2']   = int(min(l2_d, key=l2_d.get))
    feats['elev_best_match_corr'] = int(max(corr_d, key=corr_d.get))

    # Shape descriptors
    feats.update(_slope_features(raw_resampled, total_change))
    return feats


ELEV_FEAT_COLS = (
    [f'elev_l2_raw_p{p}'   for p in range(N_PATHS)] +
    [f'elev_l2_shape_p{p}' for p in range(N_PATHS)] +
    [f'elev_corr_p{p}'     for p in range(N_PATHS)] +
    [f'elev_dtw_p{p}'      for p in range(N_PATHS)] +
    ['elev_best_match_l2', 'elev_best_match_corr',
     'elev_total_change', 'elev_abs_total',
     'elev_steepest_pos', 'elev_steepest_mag', 'elev_flat_frac',
     'elev_climb_p25_pos', 'elev_climb_p50_pos', 'elev_climb_p75_pos']
)
