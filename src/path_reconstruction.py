"""
Heading × elevation path reconstruction & template matching.

Idea (per user): a path's *shape* can be reconstructed from heading + elevation
alone, with no step detection. For each ~1m of elevation change (signed),
advance one unit in the average compass heading during that elevation bin.
The resulting 2D trajectory is used to match against pre-computed templates
of the 5 routes (built from user-supplied GPX files).

Inputs at inference:
  - Heading: phone_orientationx (Android compass yaw, tilt-compensated).
             Fallback: atan2(phone_my, phone_mx) for the few records missing it.
  - Elevation: phone_pressure (1 hPa ≈ -8.3 m). Fallback: altitude (GPS).

Templates come from data/path_gpx/path{0..4}.csv.
"""

import os
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPX_DIR  = os.path.join(ROOT, 'data', 'path_gpx')
PRESSURE_TO_METRES = -8.3            # 1 hPa drop ≈ +8.3 m elevation gain

ELEV_BIN_M = 1.0                     # 1-meter elevation bins
N_PATHS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bearing(lat1, lon1, lat2, lon2):
    """Initial compass bearing from point1→point2 in degrees [0, 360)."""
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlon = np.deg2rad(lon2 - lon1)
    x = np.sin(dlon) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y)) % 360


def _circular_mean_deg(angles):
    r = np.deg2rad(angles)
    return float(np.degrees(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))) % 360)


def _circular_diff(a, b):
    """Signed smallest angular difference (a - b), result in [-180, 180]."""
    return (a - b + 180) % 360 - 180


# ---------------------------------------------------------------------------
# Reconstruction core: heading sequence binned by signed cumulative elevation
# ---------------------------------------------------------------------------

def _bin_heading_by_elevation(heading_seq, elev_seq, bin_m=ELEV_BIN_M,
                                smooth_samples=5):
    """
    Bin headings by ABSOLUTE cumulative elevation change. Returns:
        bin_headings: array of (circular-mean) heading per bin [deg]
        sign_arr    : array of +1/-1 per bin marking ascending/descending

    `smooth_samples` controls elevation smoothing.  Recordings sample at
    ~100 Hz so noise on each sample creates spurious cumulative-elevation
    inflation if smoothing is short.  Templates from GPX have ~50-100 points
    spread over many minutes, so a much smaller smoothing is appropriate.
    Caller should pass the right value:
      - GPX templates:  smooth_samples=3 (sample-level)
      - phone recordings @ 100 Hz: smooth_samples=500 (~5 s window)
    """
    heading_seq = np.asarray(heading_seq, dtype=float) % 360
    elev_seq    = np.asarray(elev_seq, dtype=float)
    n = min(len(heading_seq), len(elev_seq))
    if n < 4:
        return np.array([]), np.array([])

    heading_seq = heading_seq[:n]
    elev_seq    = elev_seq[:n]

    smooth_n = max(3, min(smooth_samples, n // 2))
    if n >= smooth_n:
        elev_seq = uniform_filter1d(elev_seq, size=smooth_n, mode='nearest')

    d_elev    = np.diff(elev_seq)
    abs_d     = np.abs(d_elev)
    cum_abs   = np.concatenate([[0.0], np.cumsum(abs_d)])
    total_abs = cum_abs[-1]
    if total_abs < 5.0:                 # essentially flat → cannot reconstruct
        return np.array([]), np.array([])

    overall_sign = 1 if (elev_seq[-1] - elev_seq[0]) > 0 else -1
    n_bins = int(total_abs / bin_m)
    if n_bins < 5:
        return np.array([]), np.array([])

    cum_mid = (cum_abs[:-1] + cum_abs[1:]) / 2.0   # one per segment

    bin_headings = np.full(n_bins, np.nan)
    for i in range(n_bins):
        lo = i * bin_m
        hi = (i + 1) * bin_m
        mask = (cum_mid >= lo) & (cum_mid < hi)
        if mask.any():
            bin_headings[i] = _circular_mean_deg(heading_seq[:-1][mask])

    # Forward-fill any empty bins
    last = np.nan
    for i in range(n_bins):
        if not np.isnan(bin_headings[i]):
            last = bin_headings[i]
        else:
            bin_headings[i] = last if not np.isnan(last) else 0.0
    return bin_headings, np.full(n_bins, overall_sign, dtype=int)


def _heading_to_xy(bin_headings):
    """Integrate unit-step heading sequence to produce 2D positions."""
    if len(bin_headings) == 0:
        return np.array([]), np.array([])
    rad = np.deg2rad(bin_headings)
    dx  = np.sin(rad)        # east component
    dy  = np.cos(rad)        # north component
    x = np.concatenate([[0.0], np.cumsum(dx)])
    y = np.concatenate([[0.0], np.cumsum(dy)])
    return x, y


def reconstruct_trajectory(heading_seq, elev_seq, bin_m=ELEV_BIN_M,
                             smooth_samples=5):
    """
    Returns:
        positions: (N+1) × 2 array of (x_east, y_north) trajectory points
        bin_headings: N headings per bin (deg)
        sign:    +1 or -1 (overall ascending or descending)
    """
    bh, sign = _bin_heading_by_elevation(heading_seq, elev_seq, bin_m=bin_m,
                                           smooth_samples=smooth_samples)
    if len(bh) == 0:
        return np.zeros((0, 2)), np.array([]), 0
    x, y = _heading_to_xy(bh)
    return np.column_stack([x, y]), bh, int(sign[0]) if len(sign) > 0 else 0


# ---------------------------------------------------------------------------
# Templates from GPX
# ---------------------------------------------------------------------------

def _gpx_to_heading_elev(df):
    """
    Convert a GPX dataframe (lat, lon, elevation_m, time) into per-segment
    heading + per-point elevation aligned to a common grid (length = n).
    """
    lat = df['latitude'].values
    lon = df['longitude'].values
    elev = df['elevation_m'].values
    n = len(df)
    headings = np.full(n, np.nan)
    headings[:-1] = _bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])
    headings[-1]  = headings[-2] if n > 1 else 0.0
    # Forward-fill NaN (when consecutive identical points: bearing is undefined)
    last = headings[0] if not np.isnan(headings[0]) else 0.0
    for i in range(n):
        if np.isnan(headings[i]):
            headings[i] = last
        else:
            last = headings[i]
    return headings, elev


def build_path_templates(gpx_dir=GPX_DIR, bin_m=ELEV_BIN_M):
    """
    Build 5 path templates from GPX files.
    GPX has ~50-100 points across minutes → minimal smoothing (3 samples).
    """
    templates = {}
    for p in range(N_PATHS):
        path = os.path.join(gpx_dir, f'path{p}.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        headings, elev = _gpx_to_heading_elev(df)
        positions, bin_headings, sign = reconstruct_trajectory(
            headings, elev, bin_m=bin_m, smooth_samples=3)
        templates[p] = {
            'bin_headings': bin_headings,
            'positions':    positions,
            'sign':         sign,
            'n_bins':       len(bin_headings),
        }
    return templates


# ---------------------------------------------------------------------------
# Heading + elevation extraction from sensor data
# ---------------------------------------------------------------------------

def _ts_seconds(raw_timestamps, n_values):
    """Linear time axis (in seconds) for n_values samples spanning ts range."""
    ts_ms = np.array([t[1] for t in raw_timestamps], dtype=float)
    if len(ts_ms) < 2:
        return np.linspace(0, 1, n_values)
    return np.linspace(ts_ms[0], ts_ms[-1], n_values) / 1000.0


# ---------------------------------------------------------------------------
# Walking-direction estimation (Renaudin / Kang style PCA on horizontal acc)
# ---------------------------------------------------------------------------

def _device_to_world_rotation(grav_xyz, mag_xyz):
    """
    Build the rotation matrix R such that R @ v_device = v_world,
    where world frame is (East, North, Up).

    Mirrors Android's SensorManager.getRotationMatrix logic:
      - gravity vector g (in device frame, points DOWN-in-world)
      - magnetic vector m (in device frame)
      - East axis  e_dev = m × g (cross product, in device frame), then normalised
      - North axis n_dev = g × e_dev / |g|
      - Up axis    u_dev = -g / |g|  (gravity points down → up is opposite)

    R rows are [E, N, U]_world expressed in device frame.
    Then R @ v_device gives v in world frame.
    """
    g = np.asarray(grav_xyz, dtype=float)
    m = np.asarray(mag_xyz,  dtype=float)
    g_norm = np.linalg.norm(g)
    if g_norm < 1e-6:
        return np.eye(3)

    e_dev = np.cross(m, g)
    e_norm = np.linalg.norm(e_dev)
    if e_norm < 1e-6:
        return np.eye(3)
    e_dev /= e_norm
    n_dev = np.cross(g, e_dev) / g_norm
    u_dev = -g / g_norm
    R = np.stack([e_dev, n_dev, u_dev], axis=0)   # R @ v_dev = [E, N, U]_world
    return R


def gyro_integrated_heading(raw):
    """
    Heading (cumulative yaw) from integrating phone gyroscope projected onto
    the gravity axis. Rotation around vertical = compass yaw rate.

    Why this beats magnetometer/orientation-based heading:
      - Robust to phone-in-pocket orientation (gyro measures rotation
        regardless of which way phone is pointing)
      - Immune to magnetic interference (no mag dependency)
      - Smooth (no pocket-shift artefacts)

    Trade-off: slow drift (~0.1°/s typical bias × 10min ≈ 60° drift). We
    debias by subtracting the median yaw rate (robust to outliers and to
    sustained turns).

    Returns
    -------
    heading_deg : array (n,) cumulative heading from arbitrary reference
                  (rotation-invariant since we use turn angles in matching)
    ts          : array (n,) time axis in seconds
    """
    data = raw['data']
    gx = np.array(data['phone_gx']['values'], dtype=float)
    gy = np.array(data['phone_gy']['values'], dtype=float)
    gz = np.array(data['phone_gz']['values'], dtype=float)
    grx = np.array(data['phone_gravx']['values'], dtype=float)
    gry = np.array(data['phone_gravy']['values'], dtype=float)
    grz = np.array(data['phone_gravz']['values'], dtype=float)

    n_g = min(len(gx), len(gy), len(gz))
    n_r = min(len(grx), len(gry), len(grz))
    if n_g < 50 or n_r < 50:
        return np.array([]), np.array([])

    # Gyro and gravity sample at ~100 Hz; might differ slightly. Resample
    # gravity onto gyro time grid for vector-by-vector dot product.
    ts_g = _ts_seconds(data['phone_gx']['raw_timestamps'], n_g)
    ts_r = _ts_seconds(data['phone_gravx']['raw_timestamps'], n_r)
    grx_i = np.interp(ts_g, ts_r, grx[:n_r])
    gry_i = np.interp(ts_g, ts_r, gry[:n_r])
    grz_i = np.interp(ts_g, ts_r, grz[:n_r])

    # Gravity unit vector per sample (in device frame)
    g_norm = np.sqrt(grx_i**2 + gry_i**2 + grz_i**2) + 1e-9
    gx_h = grx_i / g_norm
    gy_h = gry_i / g_norm
    gz_h = grz_i / g_norm

    # Yaw rate = gyro · g_hat (component of gyro vector along gravity).
    # Sign: yaw_rate is rotation around the gravity axis. If g_hat points
    # *down* (Android convention varies), this gives compass-clockwise positive.
    # We don't depend on the absolute sign — turn angles are direction-aware
    # but a flip just flips all turns; matching via |Δheading| would be
    # invariant. We keep sign as-is and let DTW handle it.
    yaw_rate = gx[:n_g] * gx_h + gy[:n_g] * gy_h + gz[:n_g] * gz_h

    # Debias: subtract median yaw rate (robust to outliers / turns)
    yaw_rate = yaw_rate - np.median(yaw_rate)

    # Integrate: cumulative heading from start.
    # gyro is in rad/s (Android default), so cumsum * dt gives radians.
    dt = (ts_g[-1] - ts_g[0]) / max(n_g - 1, 1)
    cum_heading_rad = np.cumsum(yaw_rate) * dt
    heading_deg = np.degrees(cum_heading_rad) % 360
    return heading_deg, ts_g


def estimate_walking_heading(raw, window_s=4.0, step_s=2.0,
                              motion_acc_thresh=0.4):
    """
    Estimate compass heading of MOTION (direction of travel) per sliding
    window using PCA on horizontal linear acceleration, then rotation to
    world frame via gravity + magnetometer.

    Returns
    -------
    heading_deg : array (n_windows,) compass yaw of motion in degrees [0, 360)
    win_t       : array (n_windows,) window-center time (seconds)

    Pipeline (per window):
      1. Take linear acc (gravity-removed) over the window in DEVICE frame.
      2. Project onto horizontal plane (perpendicular to instantaneous gravity).
      3. PCA → first principal axis = walking direction (sign-ambiguous) in
         DEVICE frame.
      4. Rotate that axis into world frame using a per-window-mean grav + mag
         rotation matrix. Take atan2(east, north) → compass heading.
      5. Sign disambiguation: pick the sign that gives smaller heading change
         from previous window (continuity prior).
    """
    data = raw['data']
    # All sensors at ~100 Hz on phone
    lax = np.array(data['phone_lax']['values'], dtype=float)
    lay = np.array(data['phone_lay']['values'], dtype=float)
    laz = np.array(data['phone_laz']['values'], dtype=float)
    gx  = np.array(data['phone_gravx']['values'], dtype=float)
    gy  = np.array(data['phone_gravy']['values'], dtype=float)
    gz  = np.array(data['phone_gravz']['values'], dtype=float)
    mx  = np.array(data['phone_mx']['values'], dtype=float)
    my  = np.array(data['phone_my']['values'], dtype=float)
    mz  = np.array(data['phone_mz']['values'], dtype=float)

    n = min(len(lax), len(gx), len(mx))
    if n < 100:
        return np.array([]), np.array([])

    lax, lay, laz = lax[:n], lay[:n], laz[:n]
    gx,  gy,  gz  = gx[:n],  gy[:n],  gz[:n]
    mx,  my,  mz  = mx[:n],  my[:n],  mz[:n]

    # Time axis (seconds) for windows
    ts = _ts_seconds(data['phone_lax']['raw_timestamps'], n)
    duration = ts[-1] - ts[0]
    sr = n / max(duration, 1e-3)

    win_n  = max(20, int(window_s * sr))
    step_n = max(1, int(step_s * sr))

    headings = []
    win_times = []
    prev_heading = None

    for i in range(0, n - win_n, step_n):
        sl = slice(i, i + win_n)
        lacc = np.column_stack([lax[sl], lay[sl], laz[sl]])    # (W, 3)
        grav = np.column_stack([gx[sl],  gy[sl],  gz[sl]])
        mag  = np.column_stack([mx[sl],  my[sl],  mz[sl]])

        if np.std(np.linalg.norm(lacc, axis=1)) < motion_acc_thresh:
            # Standing/very low motion → no reliable walking direction
            headings.append(np.nan); win_times.append(ts[i + win_n // 2])
            continue

        # Per-sample horizontal projection: lacc_h = lacc - (lacc·ĝ) ĝ
        g_norm = np.linalg.norm(grav, axis=1, keepdims=True)
        g_hat = grav / np.maximum(g_norm, 1e-6)
        proj = np.sum(lacc * g_hat, axis=1, keepdims=True)
        lacc_h = lacc - proj * g_hat                            # (W, 3)

        # PCA on horizontal acc: find direction of maximum variance
        # (= walking direction in device frame, sign-ambiguous)
        H = lacc_h - lacc_h.mean(axis=0)
        cov = (H.T @ H) / max(1, len(H) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        pc1_dev = eigvecs[:, -1]                                # largest eigval

        # Rotation to world frame (use mean grav + mag of the window)
        g_mean = grav.mean(axis=0)
        m_mean = mag.mean(axis=0)
        R = _device_to_world_rotation(g_mean, m_mean)

        pc1_world = R @ pc1_dev                                 # (E, N, U)
        # Heading = atan2(East, North), measured clockwise from North
        h_deg = np.degrees(np.arctan2(pc1_world[0], pc1_world[1])) % 360

        # Disambiguate ±sign: pick whichever is closer to prev heading
        if prev_heading is not None and not np.isnan(prev_heading):
            h_alt = (h_deg + 180) % 360
            d_main = abs(_circular_diff(h_deg, prev_heading))
            d_alt  = abs(_circular_diff(h_alt, prev_heading))
            if d_alt < d_main:
                h_deg = h_alt
        prev_heading = h_deg

        headings.append(h_deg)
        win_times.append(ts[i + win_n // 2])

    return np.array(headings), np.array(win_times)


def extract_recording_heading_and_elev(raw, source='gyro'):
    """
    Extract aligned heading + elevation arrays from a recording.

    Heading source (priority order):
      'gyro':   integrate phone gyro · gravity_unit (default — robust to
                phone-pocket orientation, immune to magnetic interference)
      'pca':    PCA-walking-direction (Renaudin/Kang style)
      'orient': phone_orientationx fallback (device compass yaw)

    Elevation source priority:
      1. phone_pressure → metres above start (signed)
      2. altitude       → metres above start (signed)
    """
    data = raw['data']

    # --- Heading ---
    heading, heading_ts = np.array([]), np.array([])
    if source == 'gyro' and 'phone_gx' in data and 'phone_gravx' in data:
        heading, heading_ts = gyro_integrated_heading(raw)
    if len(heading) < 50 and source != 'orient':
        # Fallback to PCA
        h_win, h_ts = estimate_walking_heading(raw)
        if len(h_win) > 5:
            valid = ~np.isnan(h_win)
            if valid.any():
                last = np.nan
                for i in range(len(h_win)):
                    if not valid[i]:
                        h_win[i] = last
                    else:
                        last = h_win[i]
                if np.isnan(h_win[0]):
                    first_valid = np.where(valid)[0][0]
                    h_win[:first_valid] = h_win[first_valid]
                heading, heading_ts = h_win, h_ts
    if len(heading) < 50:
        heading, heading_ts = _heading_from_orientationx(raw)

    # --- Elevation ---
    if 'phone_pressure' in data and len(data['phone_pressure']['values']) > 10:
        prs = np.array(data['phone_pressure']['values'], dtype=float)
        prs_ts = _ts_seconds(data['phone_pressure']['raw_timestamps'], len(prs))
        elev_native = (prs - prs[0]) * PRESSURE_TO_METRES        # m above start
        elev = np.interp(heading_ts, prs_ts, elev_native)
    elif 'altitude' in data and len(data['altitude']['values']) > 10:
        alt = np.array(data['altitude']['values'], dtype=float)
        alt_ts = _ts_seconds(data['altitude']['raw_timestamps'], len(alt))
        elev_native = alt - alt[0]
        elev = np.interp(heading_ts, alt_ts, elev_native)
    else:
        elev = np.zeros_like(heading)

    return heading, elev


def _heading_from_orientationx(raw):
    """Fallback: use Android's phone_orientationx (device yaw)."""
    data = raw['data']
    if 'phone_orientationx' in data and len(data['phone_orientationx']['values']) > 10:
        heading = np.array(data['phone_orientationx']['values'], dtype=float) % 360
        heading_ts = _ts_seconds(data['phone_orientationx']['raw_timestamps'],
                                  len(heading))
    else:
        mx = np.array(data['phone_mx']['values'], dtype=float)
        my = np.array(data['phone_my']['values'], dtype=float)
        n = min(len(mx), len(my))
        heading = np.degrees(np.arctan2(my[:n], mx[:n])) % 360
        heading_ts = _ts_seconds(data['phone_mx']['raw_timestamps'], n)
    return heading, heading_ts


# ---------------------------------------------------------------------------
# DTW for turn-angle sequences (rotation-invariant)
# ---------------------------------------------------------------------------

def _turn_angles(headings):
    """Compute signed turn angle between consecutive headings."""
    if len(headings) < 2:
        return np.array([])
    return _circular_diff(headings[1:], headings[:-1])


def _dtw_distance(seq1, seq2, max_warp=0.4):
    """
    Dynamic Time Warping distance between two 1-D sequences.

    The warping band always covers |len(seq1) - len(seq2)| (otherwise the
    goal cell becomes unreachable when sequences have very different lengths,
    e.g. 43-bin recording vs 23-bin template).
    """
    n1, n2 = len(seq1), len(seq2)
    if n1 == 0 or n2 == 0:
        return float('inf')
    # Band must be at least |n1 - n2| + slack so the goal cell is reachable
    band = max(int(max(n1, n2) * max_warp), abs(n1 - n2) + 3)
    INF  = np.float64(np.inf)
    cost = np.full((n1 + 1, n2 + 1), INF)
    cost[0, 0] = 0.0
    for i in range(1, n1 + 1):
        j_lo = max(1, i - band)
        j_hi = min(n2, i + band)
        for j in range(j_lo, j_hi + 1):
            d = abs(seq1[i - 1] - seq2[j - 1])
            cost[i, j] = d + min(cost[i - 1, j],
                                  cost[i, j - 1],
                                  cost[i - 1, j - 1])
    return float(cost[n1, n2] / max(n1, n2))    # length-normalised


# ---------------------------------------------------------------------------
# Match a recording's reconstructed trajectory to all templates
# ---------------------------------------------------------------------------

def _resample_seq(seq, n_target):
    """Linear-interp resample 1D sequence to n_target points."""
    if len(seq) < 2:
        return np.full(n_target, np.nan)
    x = np.linspace(0, 1, len(seq))
    return np.interp(np.linspace(0, 1, n_target), x, seq)


def _frechet_distance(p1, p2):
    """
    Discrete Frechet distance between two 1-D sequences.
    O(n*m) memory but with our sequences (≤100) it's fine. Length-normalised.
    """
    n, m = len(p1), len(p2)
    if n == 0 or m == 0:
        return float('inf')
    ca = np.full((n, m), -1.0)
    ca[0, 0] = abs(p1[0] - p2[0])
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], abs(p1[i] - p2[0]))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], abs(p1[0] - p2[j]))
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]),
                            abs(p1[i] - p2[j]))
    return float(ca[n - 1, m - 1])


def match_recording_to_templates(raw, templates, bin_m=ELEV_BIN_M):
    """
    Reconstruct the recording's path (heading × elevation bins) and compute
    multiple distance metrics to each template. The classifier sees all of
    them and picks the most discriminative combination.

    Per template, we emit 4 distances on the cumulative-turn (= integrated
    Δheading from start of bin sequence) representation, which is rotation-
    invariant by construction:

      traj_dtw_p{p}   : DTW distance (length-aware, allows warp)
      traj_l2_p{p}    : L2 distance after resampling both to 50 points
      traj_corr_p{p}  : Pearson correlation (shape similarity)
      traj_frechet_p{p}: Discrete Frechet distance (max-of-min, conservative)
    """
    heading, elev = extract_recording_heading_and_elev(raw)
    # Heading is at gyro sample rate ~100 Hz, so need a multi-second
    # smoothing window on elevation to suppress sample-level pressure noise.
    rec_positions, rec_bin_headings, rec_sign = reconstruct_trajectory(
        heading, elev, bin_m=bin_m, smooth_samples=500)

    keys_per_template = ['traj_dtw_p{p}', 'traj_l2_p{p}',
                          'traj_corr_p{p}', 'traj_frechet_p{p}']
    out = {}
    for p in range(N_PATHS):
        for k in keys_per_template:
            out[k.format(p=p)] = np.nan
    out['traj_n_bins']           = len(rec_bin_headings)
    out['traj_sign']             = rec_sign
    out['traj_best_match_dtw']   = -1
    out['traj_best_match_corr']  = -1
    out['traj_best_match_dist']  = np.nan

    if len(rec_bin_headings) < 5:
        return out

    # Rotation-invariant signal: cumulative turn from start (start = 0)
    rec_turns = _turn_angles(rec_bin_headings)
    rec_cum_turns = np.concatenate([[0.0], np.cumsum(rec_turns)])
    rec_cum_resampled = _resample_seq(rec_cum_turns, 50)

    for p, tmpl in templates.items():
        if tmpl['n_bins'] < 5:
            continue
        # Sign mismatch (uphill rec vs downhill tmpl etc.) → all distances big
        if rec_sign != 0 and tmpl['sign'] != 0 and rec_sign != tmpl['sign']:
            for k in keys_per_template:
                out[k.format(p=p)] = 9999.0
            continue
        tmpl_turns = _turn_angles(tmpl['bin_headings'])
        tmpl_cum   = np.concatenate([[0.0], np.cumsum(tmpl_turns)])
        tmpl_cum_resampled = _resample_seq(tmpl_cum, 50)

        # 1. DTW on raw (variable-length) cumulative turns
        d_dtw = _dtw_distance(rec_cum_turns, tmpl_cum, max_warp=0.4)
        # 2. L2 on resampled curves
        d_l2  = float(np.sqrt(np.mean((rec_cum_resampled - tmpl_cum_resampled) ** 2)))
        # 3. Pearson correlation (linear-data: cumulative turns aren't bounded)
        if (np.std(rec_cum_resampled) > 1e-6
                and np.std(tmpl_cum_resampled) > 1e-6):
            d_corr = float(np.corrcoef(rec_cum_resampled, tmpl_cum_resampled)[0, 1])
        else:
            d_corr = 0.0
        # 4. Frechet
        d_frechet = _frechet_distance(rec_cum_resampled, tmpl_cum_resampled)

        out[f'traj_dtw_p{p}']     = d_dtw
        out[f'traj_l2_p{p}']      = d_l2
        out[f'traj_corr_p{p}']    = d_corr
        out[f'traj_frechet_p{p}'] = d_frechet

    dtw_d  = {p: out[f'traj_dtw_p{p}']  for p in range(N_PATHS)
               if not np.isnan(out[f'traj_dtw_p{p}']) and out[f'traj_dtw_p{p}'] < 9000}
    corr_d = {p: out[f'traj_corr_p{p}'] for p in range(N_PATHS)
               if not np.isnan(out[f'traj_corr_p{p}']) and out[f'traj_corr_p{p}'] > -1}
    if dtw_d:
        out['traj_best_match_dtw']  = int(min(dtw_d, key=dtw_d.get))
        out['traj_best_match_dist'] = dtw_d[out['traj_best_match_dtw']]
    if corr_d:
        out['traj_best_match_corr'] = int(max(corr_d, key=corr_d.get))
    return out


PATH_RECON_FEAT_COLS = (
    [f'traj_dtw_p{p}'    for p in range(N_PATHS)] +
    [f'traj_l2_p{p}'     for p in range(N_PATHS)] +
    [f'traj_corr_p{p}'   for p in range(N_PATHS)] +
    [f'traj_frechet_p{p}' for p in range(N_PATHS)] +
    ['traj_n_bins', 'traj_sign',
     'traj_best_match_dtw', 'traj_best_match_corr', 'traj_best_match_dist']
)
