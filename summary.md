# MHAM Ex3 — Session Summary (2026-05-10)

## Starting state

| Task        | CV (adjusted) | Weight | Contrib |
|-------------|--------------:|-------:|--------:|
| watch_loc   | 0.9506        | 0.25   | 0.2376  |
| path_idx    | 0.9125        | 0.25   | 0.2281  |
| standing    | 0.9479        | 0.0625 | 0.0592  |
| walking     | 0.9557        | 0.0625 | 0.0597  |
| running     | 0.8822        | 0.0625 | 0.0551  |
| cycling     | 0.9405        | 0.0625 | 0.0588  |
| step_count  | 0.9516        | 0.25   | 0.2379  |
| **Total CV**    |               |        | **0.9366** |
| **Public Kaggle** |             |        | **0.936** |

Goal: +0.02 composite, prioritizing path classifier and step counter (both 0.25 weight, both with apparent room).

## What we tried

### Path classifier (started at CV 0.930 / public 0.936)

| # | Idea | CV result | Public |
|---|------|----------|-------|
| 1 | Phone-IMU stair detector (11 features) — to attack cycling false positives in pressure-based stair detection | 0.930 → 0.935 (+0.005) | 0.930 → 0.923 (−0.007) |
| 2 | Stripped to 2 strongest phone-stair features | 0.930 → 0.927 (−0.003) | not tested |
| 3 | PCA-on-world-frame-acc walking-direction features (5 features) | not run — distributional analysis showed R<0.3, abandoned |
| 4 | Tilt-compensated `phone_orientationx` walking-direction features (5 features), 30s window | 0.930 → 0.911 (−0.019) | not tested |
| 5 | Drop 60 dead-weight heading features (kept 24 with `mv_imp >= 0.001`) | 0.930 → 0.913 (−0.017) | 0.936 → 0.934 (−0.002) |
| 6 | **Alt-rate features** — pressure-derived altitude rates per recording-third + early/late ratio (5 features) | 0.930 → ~0.93 (within noise) | **0.936 → 0.936 (held)** |

The only path-classifier change that survived: **alt-rate features**. They scored highest in the new feature-discrimination audit (`alt_rate_late_per_s` had the highest mut_info of any feature in the whole set, 0.81).

### Step counting (started at LOO 0.9516)

| # | Idea | LOO result |
|---|------|-----------|
| 1 | Phone-IMU per-window cycling veto via gyro std threshold | 0.9516 → 0.68 across all thresholds (catastrophic) |
| 2 | Force phone source for wrist + retune (FREQ/phone, WPD/phone) | wrist LOO 0.9226 → 0.66 (catastrophic) |

Neither worked. Step counting is at a local optimum on the 33-sample LOO data.

## What changed in the codebase

**Kept:**
- `src/train_path.py` — added `_alt_rate_features()` and `ALT_RATE_KEYS` (5 features). Wired into `extract_path_features` and `FEATURE_COLS_RAW`.
- `src/train_steps.py` — added `_phone_cycling_mask()` helper (unused but available; gyro-std discriminator failed empirically).
- `sweeps/feature_discrimination.py` (new) — per-feature univariate balanced accuracy + mutual info + multivariate importance; saves `results/feature_discrimination.csv`.
- `sweeps/compute_oof_act_proba.py` (new) — saves OOF activity *probabilities* (existing `compute_oof_predictions.py` only saved labels).
- `sweeps/sweep_phone_cycling_veto.py` (new) — sweeps the phone-cycling-gyro threshold for step-count veto. Negative result archived.
- `sweeps/sweep_wrist_phone_source.py` (new) — wrist-only step-counter source comparison. Confirms WPD/watch is best for wrist.

**Reverted:**
- Phone-IMU stair detector code (both versions).
- Phone azimuth / walking-direction features.
- Heading column pruning.
- Phone-IMU cycling veto loop in `count_steps_freq`/`count_steps_wpd`.
- Phone-source for wrist in `train_steps`.

## Final state

- CV composite ~0.9383 (with alt-rate features added)
- Public Kaggle: **0.936** (held — same as starting point)

We did not achieve the +0.02 target, but the alt-rate features are a small, clean win and the audit infrastructure is there for future iteration.

## Lessons learned (saved as memory for future sessions)

1. **CV–public gap is real and structural.** Public is ~56 samples (~1.8 pts per sample), CV is 5-fold on 396. Sub-+0.005 CV gains routinely vanish or regress on public. Treat anything below +0.005 CV gain as suspect, especially when adding features to an already-pruned candidate set.

2. **Aggregate signal ≠ classifier-usable signal.** Both the phone-stair and walking-direction features showed clear per-path means but failed in CV. Per-recording variance + 5-fold splits + per-user offsets break what looks promising in the means.

3. **Univariate balanced accuracy is the cleaner "is this feature alive" metric.** A feature with `univ_bal < 0.40` very rarely survives; one with `univ_bal > 0.50` almost always does (modulo redundancy with stronger features). `multivar_imp` from a single fit is **not** a good "is this feature dead" metric — it misses interaction-only signal (we proved this when pruning 60 mv_imp~0 heading features cost us 0.017 CV).

4. **Feature-engineering ideas that *should* work often don't, on small data.**
   - "Phone in pocket sees clean foot-strike" → wrist phone-source LOO score 0.66 (vs watch 0.92).
   - "Cycling has very low phone gyro std" → walking also has low phone gyro std.
   - "Walking direction in world frame from quaternion + linear acc PCA" → axis flips between fore-aft and lateral randomly.
   - General pattern: physically-motivated discriminators on phone-in-pocket data often have far more variance than expected.

5. **Single-fit `mv_imp` ranks features in interactions.** A feature with `mv_imp = 0.0008` can carry real information through interactions; `mv_imp = 0` does not always mean dead.

## Possible future directions (not tried)

- **Cycling-aware pressure stair detector**: gate the pressure-derivative burst detector by phone-IMU walking-detected windows. Higher implementation cost; targets the same wrist+cycling failure mode that bedeviled both classifiers. The user judged this too complex this session — but could be worth it if you want to revisit.
- **Activity classifier improvements**: running has the lowest score (0.882 adjusted) but small weight (0.0625). Lifting it to 0.95 only adds 0.004 to composite. Low EV unless it's quick.
- **Watch_loc**: already at 0.95 adjusted. Hard to push further without overfitting; little to gain.
- **Hyperparameter retune of GBMs**: not attempted. Existing hyperparams are plausibly already near-optimal (the user noted prior sweeps), but with the new alt-rate features in the candidate set, a quick re-tune *could* squeeze a fraction.

## Files changed (git status at session end)

```
M  src/train_path.py        (alt-rate features added)
M  src/train_steps.py       (cycling-mask helper added, in-loop veto reverted)
?? sweeps/feature_discrimination.py
?? sweeps/compute_oof_act_proba.py
?? sweeps/sweep_phone_cycling_veto.py
?? sweeps/sweep_wrist_phone_source.py
?? results/feature_discrimination.csv
?? results/oof_activity_proba.csv
```
