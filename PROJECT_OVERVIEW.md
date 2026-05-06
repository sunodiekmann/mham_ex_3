# MHAM Exercise 3 — Project Overview & Status

## Assignment Summary

**Course:** Mobile Health and Activity Monitoring, ETH Zürich, Spring 2026  
**Deadline:** 2026-05-11, 11:59 am  
**Points:** 60 pts (+ up to 8 bonus pts for top-2 per subtask on private leaderboard)  
**Submission:** Kaggle competition + Polybox (Jupyter notebook + models + CSV)

### Four Prediction Subtasks

| # | Task | Output | Metric |
|---|------|--------|--------|
| 1 | Step Counting | integer s ∈ ℕ₀ | MAPE / MAE |
| 2 | Watch Location | l ∈ {0=wrist, 1=belt, 2=ankle} | Balanced accuracy |
| 3 | Activity Recognition | 4 booleans: standing/walking/running/cycling | Balanced accuracy / F1 |
| 4 | Path Classification | p ∈ {0,1,2,3,4} | Balanced accuracy |

---

## Dataset

- **396 train** recordings, **280 test** recordings
- LilyGo smartwatch (acc, gyro, magnetometer ~200 Hz) + smartphone (GPS, pressure, acc, gyro, linear acc ~100 Hz)
- 3 watch locations: wrist (134), belt (123), ankle (139) — roughly balanced
- 5 paths (uphill 0-2, downhill 3-4): {0:82, 1:76, 2:74, 3:71, 4:93}
- Only **33 training recordings** have ground-truth step counts (TA-team only)
- Activity labels are **weak** (recording-level, not timestamped)

### Path Descriptions
- **Path 0:** Central → ETH via Weinbergstrasse/Leonhardstrasse/Rämistrasse — uphill, no stairs
- **Path 1:** Same + Clausiusstrasse detour — stairs early in ascent
- **Path 2:** Walchebrücke → via Walchetor stairs — stairs mid-ascent
- **Path 3:** Reverse of Path 2 — downhill, stairs early in descent
- **Path 4:** Downhill, stairs in middle of descent

---

## Pipeline Architecture

```
data/*.pkl
    ↓
extract_features.py       → features_train.csv / features_test.csv
    ↓                              ↓
classify_watch_location.py    classify_activities.py
classify_path.py              count_steps.py
    ↓
model_*.pkl / step_count_params.pkl
    ↓
predict.py                → submission.csv
```

### Module Breakdown

**`extract_features.py`** — shared feature extraction
- 5s windows, 50% overlap → percentile summaries (p25/p50/p75/p90)
- Watch: acc/gyro magnitude std, energy, dominant frequency, cross-axis correlations
- Phone: same + linear acc (gravity-removed)
- Extra: `acc_anisotropy` (max/min axis std ratio), `acc_still_frac`, `acc_walk_frac`
- Heading histogram (36 × 10° bins) from phone magnetometer
- Temperature slope, GPS altitude, barometric pressure

**`classify_watch_location.py`** — SVM-RBF (C=1.43, γ=0.026)
- Features: orientation-invariant acc/gyro magnitude stats + derived ratios
- Key derived: `watch_phone_gyro_ratio`, `acc_anisotropy`, log-scaled energies
- Sweeps done in `sweep_watch_location.py`

**`classify_activities.py`** — 4 independent binary classifiers
- VotingClassifier(SVM + GBM + RF, soft, weights=[1,2,1]) per activity
- Phone features (location-independent) + watch features + context
- Handles imbalance via `class_weight='balanced'`

**`classify_path.py`** — GradientBoosting (n_est=584, depth=5, lr=0.121)
- Primary signal: barometric pressure (GPS unreliable)
- Stair detection: sustained pressure derivative bursts (≥8s)
- Stair position features (fraction of altitude progress)
- Heading features merged from features CSV
- Sweeps done in `sweep_path.py`

**`count_steps.py`** — STFT frequency × locomotion duration (primary)
- Dual-sensor mask: acc_std > thresh AND gyro_std > gyro_thresh (excludes cycling/standing)
- FFT peak in [1.0, freq_hi] Hz → accumulate freq × window_duration
- WPD on phone linear acc (backup algorithm)
- Location-specific params tuned via grid search LOO-CV on 33 labelled recordings

---

## Current Performance (CV on train set)

| Task | Score |
|------|-------|
| Watch Location | Balanced acc ~0.92 (SVM-RBF, 5-fold CV) |
| Path Classification | Balanced acc ~see `path_results.png` (GBM, 5-fold CV) |
| Activities | Per-activity F1 / balanced acc → `activity_results.png` |
| Step Count | MAPE/MAE → `step_count_results.png` (LOO-CV, 33 samples) |

---

## Weaknesses & Improvement Opportunities

### Critical

**1. Step Count — tiny labelled dataset (33 recordings)**
- LOO-CV on 33 samples is extremely noisy; any per-location split has ~10 samples
- Opportunity: use `phone_steps` as noisy pseudo-labels for unsupervised calibration or semi-supervised learning
- Opportunity: try a physics-based approach with stricter locomotion masks and phone linear acc

**2. Step Count — location-specific params only cover watch FREQ; no ensemble**
- The WPD algorithms (watch & phone) exist in code but are not used at inference
- Opportunity: ensemble FREQ + phone-WPD, weighted by how confident the locomotion mask is
- Ankle placement should use watch directly (strong heel-strike); wrist/belt should prefer phone acc

**3. Path Classification — Paths 1 vs 2 and Paths 3 vs 4 confusion**
- Both uphill pair and both downhill pair share very similar pressure profiles and elevation gain
- Stair position features (`stair_pos_*`) are the primary discriminator; heading histogram secondary
- Opportunity: extract more granular timing features — exact pressure percentile at which stair burst peaks
- Opportunity: use GPS coordinates if not corrupted to determine start point (Central vs Walchebrücke)
- **NOTE: latitude/longitude/bearing/speed are FORBIDDEN as inputs per spec**

**4. Activity Recognition — weak labels & no temporal info**
- Labels are recording-level booleans; no timestamps for when activities occurred
- The current approach treats each recording as a bag-of-windows (permutation-invariant)
- Opportunity: explicit activity segmentation — detect transitions, compute per-segment features
- `acc_walk_frac` and `acc_still_frac` are good proxies but don't capture sequence

### Moderate

**5. Activity Recognition — standing detection**
- Standing is rare (24%) and overlaps with slow walking in acc features
- `acc_still_frac` and `acc_walk_frac` help but are threshold-sensitive
- Opportunity: explicit window-level standing detector → fraction of recording in still windows

**6. Activity Recognition — running + cycling confusion**
- Running (19%) and cycling (14%) are minority classes
- Cycling has low gyro std (key discriminator) but this signal is watch-location-dependent
- Opportunity: cross-task dependency — use predicted watch_loc to select location-specific thresholds for activity detection

**7. Path Classification — GPS corruption handling**
- GPS corrupted samples fall back to pressure-only; can lose the altitude progression shape
- Opportunity: better GPS quality metric (not just altitude range but also GPS variance / freeze detection)

**8. Error propagation: watch_loc → activity → step_count**
- Watch_loc is predicted first and used as context for both activity recognition and step counting
- A wrong watch_loc prediction cascades into both
- Opportunity: calibrate confidence; use soft watch_loc probability vector as feature instead of hard label

### Minor

**9. Feature extraction — no spectral entropy in final feature set for path/activity**
- `spec_entropy` is computed in `_windowed()` but only used for walk-specific features in extract_features.py
- Could provide additional standing/cycling discriminator

**10. Step count — frequency range fixed**
- `freq_lo=1.0` Hz cuts off slow walking cadence (~0.8 Hz in elderly)
- `freq_hi` is tuned per location but running at 3+ Hz may be under-counted
- Opportunity: detect if recording has running (from activity classifier) and widen band accordingly

**11. No cross-validation on the full pipeline**
- Each module is CV'd independently; pipeline CV would catch compounding errors
- The watch_loc prediction error is not propagated into activity CV

---

## Files Produced

| File | Content |
|------|---------|
| `features_train.csv` | 396 × 158 feature matrix with labels |
| `features_test.csv` | 280 × ~120 feature matrix |
| `model_watch_location.pkl` | Trained SVM-RBF pipeline |
| `model_activities.pkl` | 4 VotingClassifier pipelines |
| `model_path.pkl` | Trained GBM pipeline + feature list |
| `step_count_params.pkl` | Per-location FREQ algorithm parameters |
| `submission.csv` | 280 test predictions |
| `sweep_path.png` | Hyperparameter sweep plots |
| `path_results.png` | CV confusion matrix for path |
| `activity_results.png` | Per-activity CV confusion matrices |
| `watch_location_results.png` | Watch location CV confusion matrix |
| `step_count_results.png` | Step count scatter + error by location |

---

## Next Steps (Priority Order)

1. **Run `predict.py train`** to get current end-to-end scores on train set
2. **Improve path classifier**: add GPS start-point feature (allowed: first GPS fix, not lat/lon trajectory); try gradient stacking of pressure derivative
3. **Improve step counter**: test ensemble of FREQ + phone-WPD per location; tune freq bands based on activity classifier output
4. **Improve activity standing/running**: explicit window-level detector → fraction above threshold
5. **Package for Kaggle**: ensure submission runs in <10s per recording on CPU; test in clean environment
