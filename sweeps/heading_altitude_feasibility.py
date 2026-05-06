"""
Feasibility test for the heading × altitude path-reconstruction idea.

Per user's intuition: for each ~1m of elevation gain, advance one unit in
the average heading direction during that elevation bin. Compare the
resulting shape to the actual GPS path.

Validates:
  1. Per-path elevation profile (how many distinct 1m bins?)
  2. Heading-vs-elevation curve (per path, does it look distinctive?)
  3. Reconstructed shape using only heading + altitude vs ground truth
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPX_DIR = os.path.join(ROOT, 'data', 'path_gpx')
RESULTS = os.path.join(ROOT, 'results')

PATH_NAMES = {
    0: 'P0 Central→ETH',
    1: 'P1 +Clausiusstr',
    2: 'P2 Walchebrücke→ETH',
    3: 'P3 reverse P2',
    4: 'P4 downhill stairs-mid',
}
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def haversine_xy(lat, lon, lat0, lon0):
    """Convert lat/lon to local meters east/north relative to (lat0, lon0)."""
    R = 6371000.0
    dlat = np.deg2rad(lat - lat0)
    dlon = np.deg2rad(lon - lon0)
    lat0r = np.deg2rad(lat0)
    x_east  = R * dlon * np.cos(lat0r)
    y_north = R * dlat
    return x_east, y_north


def bearing(lat1, lon1, lat2, lon2):
    """Initial bearing from point1 to point2 in compass degrees (0=N, 90=E)."""
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlon = np.deg2rad(lon2 - lon1)
    x = np.sin(dlon) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y)) % 360


def load_path(p):
    df = pd.read_csv(os.path.join(GPX_DIR, f'path{p}.csv'))
    # Convert to local meters relative to first point
    lat0, lon0 = df['latitude'].iloc[0], df['longitude'].iloc[0]
    df['x_east'], df['y_north'] = haversine_xy(
        df['latitude'].values, df['longitude'].values, lat0, lon0)
    return df


def compute_headings(df):
    """Per-segment compass bearing between consecutive GPS points."""
    lat = df['latitude'].values
    lon = df['longitude'].values
    bearings = bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return bearings


# ---------------------------------------------------------------------------
# Reconstruction: walk through altitude in 1-meter bins, accumulate heading
# ---------------------------------------------------------------------------
def reconstruct_by_altitude(df, bin_m=1.0):
    """
    User's idea: for each ~bin_m of elevation change, advance one unit in
    the average heading direction during that elevation bin.

    Returns a list of (x, y) reconstructed positions, one per bin.
    """
    elev = df['elevation_m'].values
    headings_seg = compute_headings(df)   # n-1 bearings
    # Mid-elevation per segment
    elev_mid = (elev[:-1] + elev[1:]) / 2.0

    # Cumulative absolute elevation change (so reverses count as progress)
    delev = np.abs(np.diff(elev))
    cum_elev = np.concatenate([[0], np.cumsum(delev)])
    total_elev = cum_elev[-1]
    if total_elev <= 0:
        return np.array([]), np.array([])

    # Define bins along cumulative elevation (every bin_m meters)
    n_bins = int(total_elev / bin_m)
    if n_bins < 2:
        return np.array([]), np.array([])

    bin_edges = np.linspace(0, total_elev, n_bins + 1)
    # For each bin, average the heading of segments falling in that elevation slice
    # Use midpoint of each segment's cumulative elevation
    cum_mid = (cum_elev[:-1] + cum_elev[1:]) / 2.0    # one per segment

    xs, ys = [0.0], [0.0]
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (cum_mid >= lo) & (cum_mid < hi)
        if not in_bin.any():
            # No segments in this bin — repeat previous direction
            xs.append(xs[-1]); ys.append(ys[-1])
            continue
        # Circular mean of bearings in the bin
        h = np.deg2rad(headings_seg[in_bin])
        mean_h = np.arctan2(np.mean(np.sin(h)), np.mean(np.cos(h)))
        # Compass: 0=N (y+), 90=E (x+) → step
        step_x = np.sin(mean_h)   # east component
        step_y = np.cos(mean_h)   # north component
        xs.append(xs[-1] + step_x); ys.append(ys[-1] + step_y)
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# Diagnostics + visualization
# ---------------------------------------------------------------------------
def main():
    print('=== Per-path elevation profile diagnostics ===')
    print(f'{"Path":<25} {"n_pts":>6} {"start_alt":>10} {"end_alt":>9} '
          f'{"net":>6} {"abs_total":>11} {"n_1m_bins":>10}')
    for p in range(5):
        df = load_path(p)
        elev = df['elevation_m'].values
        net  = elev[-1] - elev[0]
        delev_abs_total = float(np.abs(np.diff(elev)).sum())
        n_bins = int(delev_abs_total / 1.0)
        print(f'  Path {p} ({PATH_NAMES[p]:<18}) {len(df):>6}  '
              f'{elev[0]:>10.0f} {elev[-1]:>9.0f} {net:>+6.0f} '
              f'{delev_abs_total:>11.0f}m {n_bins:>10}')

    # ---- Plot ----
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 5, hspace=0.35, wspace=0.25)

    # Row 1: Actual GPS paths (in local meters, rotated north-up)
    for p in range(5):
        df = load_path(p)
        ax = fig.add_subplot(gs[0, p])
        ax.plot(df['x_east'], df['y_north'], color=COLORS[p], lw=2)
        ax.scatter([df['x_east'].iloc[0]], [df['y_north'].iloc[0]], color='green',
                    s=80, zorder=5, label='start')
        ax.scatter([df['x_east'].iloc[-1]], [df['y_north'].iloc[-1]], color='red',
                    s=80, zorder=5, label='end')
        ax.set_title(f'{PATH_NAMES[p]}\nground truth (GPS)')
        ax.set_xlabel('East (m)'); ax.set_ylabel('North (m)')
        ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # Row 2: Reconstructed via heading × altitude bins (1m)
    for p in range(5):
        df = load_path(p)
        rx, ry = reconstruct_by_altitude(df, bin_m=1.0)
        ax = fig.add_subplot(gs[1, p])
        if len(rx) > 0:
            ax.plot(rx, ry, color=COLORS[p], lw=2)
            ax.scatter([rx[0]], [ry[0]], color='green', s=80, zorder=5)
            ax.scatter([rx[-1]], [ry[-1]], color='red', s=80, zorder=5)
        ax.set_title(f'reconstructed @ 1m elev-bins\n({len(rx)} bins)')
        ax.set_xlabel('Σ sin(h) per bin'); ax.set_ylabel('Σ cos(h) per bin')
        ax.set_aspect('equal'); ax.grid(alpha=0.3)

    # Row 3: heading vs cumulative elevation (the actual feature)
    for p in range(5):
        df = load_path(p)
        elev = df['elevation_m'].values
        headings = compute_headings(df)
        cum = np.concatenate([[0], np.cumsum(np.abs(np.diff(elev)))])
        cum_mid = (cum[:-1] + cum[1:]) / 2.0
        ax = fig.add_subplot(gs[2, p])
        ax.plot(cum_mid, headings, '.-', color=COLORS[p], lw=1.2, ms=4)
        ax.set_title('heading vs cumulative |Δelev|')
        ax.set_xlabel('Cumulative |Δelev| (m)')
        ax.set_ylabel('Bearing (deg)')
        ax.set_ylim(0, 360); ax.set_yticks([0, 90, 180, 270, 360])
        ax.grid(alpha=0.3)

    fig.suptitle('Heading × Altitude path reconstruction feasibility',
                 fontsize=14, fontweight='bold', y=0.995)
    out = os.path.join(RESULTS, 'heading_altitude_feasibility.png')
    plt.savefig(out, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
