# PHYS-TRUST Lite — Mathematical Model

Notation: k = sample index, Δt = 1 s. C = room CO2-equivalent index (relative, not true ppm).
Each block cites the code that implements it.

---

## 1. Sensor signal chain (L0) — `sensor_physics.adc_to_rs`

Voltage divider, ratiometric ADC (supply cancels):

    Rs = RL · (1023 / ADC − 1)

- RL = load resistor (kΩ). Assumed 10 kΩ until measured.
- Oversampling: ADC = mean of 10 reads; σ_ADC reported for fault flags.

Fault flags: saturation (ADC ≤ 2 or ≥ 1021); stuck (5 identical samples with σ_ADC = 0).

---

## 2. Sensor physics — measurement model (L1a)

### 2.1 Power-law chemiresistor response — `log_index`

    C = a · ( Rs / (R0 · CF(T,RH)) )^b          a = 116.02, b = −2.816 (consensus)

Log form (used by filter, linear in log Rs):

    log C = log a + b · ( log Rs − log R0 − log CF )

### 2.2 Clean-air R0 calibration — `r0_from_clean`

At known clean-air index C_ref:

    R0 = Rs / CF · (a / C_ref)^(1/b)

- C_ref = 426 ppm: assumed outdoor clean-air CO2 (NOAA Mauna Loa weekly mean 425.9 ppm, week of 2026-09-13;
  typical current outdoor range 420–426 ppm, higher in cities). An error in C_ref only shifts the reference
  point of the index (a constant offset in log C), not the method.
- Provisional R0 on this node = 112.3 kΩ (RL = 10 kΩ assumed, pre-burn-in).

### 2.3 Temperature / humidity correction CF

Datasheet (GeorgK / balk77) — `cf_datasheet`:

    T < 20 °C :  CF = 0.00035 T² − 0.02718 T + 1.39538 − 0.0018 (RH − 33)
    T ≥ 20 °C :  CF = −0.003333 T − 0.001923 RH + 1.130128

Check: CF(20 °C, 33 %) ≈ 1.00.

Refit on own clean-air data (Abdullah et al. 2022 style) — `RefitCF`:

    log Rs = β0 + β1 T + β2 RH + β3 T·RH + ε
    CF_refit(T,RH) = exp( x(T,RH)ᵀβ − x(20,33)ᵀβ )

Enabled only when ΔT ≥ 3 °C and ΔRH ≥ 10 % in clean-air data (else datasheet CF).

### 2.3b Coefficient uncertainty — `coefficients.py`

With R0 anchored at C_ref, a cancels:

    log C − log C_ref = b · [ log(Rs/CF) − log(Rs_ref/CF_ref) ]

so σ(log C) = |log C − log C_ref| · σ_b / |b|.

Raw and physics channels share the same b, so their b-errors are identical (fully correlated).
Adding the term per channel would let the filter average it down as if independent, making bands too narrow.
It is therefore added once, to the fused output variance:

    σ²_log,total = σ²_EKF + ( (log C − log C_ref) · σ_b / b )²

σ_b = 0.066 is the **digitisation spread (n = 2)** of two coefficient sets read off one datasheet plot.
It is not sensor-to-sensor variability; real MQ135 unit-to-unit spread in b is likely larger.

### 2.4 Drift — observability limit and ABC

**Limitation (measured, not assumed).** On one node, slow sensor drift and a slow real change in
concentration both only scale Rs. Without a reference instrument no filter setting can separate them:
a sweep of q_ld ∈ {2·10⁻⁶, 2·10⁻⁵, 6·10⁻⁵} on an injected 0.2 %/min drift left the drift estimate at
0.96–0.97 while true drift reached 1.062; larger q_ld only widened bands and raised error
(post-drift RMSE 0.146 → 0.196). The system does **not** track drift within a day. The only in-filter
identifier is the background floor (§5): drift that pushes the index below outdoor background is
attributed to the sensor. On the real burn-in log this caught Rs rising 1.88× (drift estimate 1.85×).

**Automatic baseline calibration (ABC)** — `pipeline._abc`, as used in commercial NDIR CO2 sensors.
The lowest per-minute median over a rolling window W ≥ 24 h is assumed to be outdoor background, and
R0 (via ld) is re-anchored so that minimum maps to C_bg:

    m = min_{window W} median_60s(C_obs)
    if |log(m / C_bg)| > 0.02 :  ld ← ld − (log C_bg − log m) / b

Assumption: the room is aired to outdoor background at least once per window. With it, drift is tracked
on the day scale by re-anchoring. Manual re-zero (`PhysTrust.rezero()`, window open) is the demo version.
ABC is implemented but not yet benchmarked (needs multi-day data).

**Time-since-anchor band growth.** Unobservable drift is a random walk, so its variance grows linearly
since the last anchor (calibration, ABC event or manual re-zero), and resets there:

    σ²_total = q² · (σ²_EKF + σ²_b + σ²_sys)  +  k_d · (t − t_anchor)

σ_sys and q are calibrated on the part without this term (calibration data sits at a different time
since anchor than deployment). k_d default = (b·q_ld)², consistent with the filter's own drift model.
Honest source for k_d on a real node: the size of successive ABC re-anchor steps,
k_d ≈ var(Δ log C at re-anchor) / W.
Fit k_d only on data after burn-in (≥ 24–48 h powered): burn-in drift (e.g. Rs ×1.88 in the first
47 min on this node) would overstate the rate and give bands wider than normal operation needs.


Slow multiplicative ageing Rs → d·Rs, ld = log d. Enters measurement as

    log C_obs = log C + b · ld

---

## 3. Process physics — state model (L1b) — `process_physics.transition`

Single-zone well-mixed mass balance:

    dC/dt = S(t)/V − λ (C − C_bg)

Exact discretisation (S constant over step):

    C_{k+1} = C_bg + (C_k − C_bg) e^{−λΔt} + (S_k / (Vλ)) (1 − e^{−λΔt}) + w_k

- C_bg = 426, λ default 1 h⁻¹ (to be fitted).
- S unknown → modelled as process noise w_k ~ N(0, q_c² Δt), q_c = 1.0.

### 3.1 λ identification — ASTM E741 decay — `fit_decay`

After source stops (S = 0):

    ln(C(t) − C_a) = ln(C_0 − C_a) − λ t      → linear regression, report R²

### 3.2 Plausibility envelope — `max_rate`

Maximum physically plausible room-level rate:

    |dC/dt|_max = S_max/V + λ_max |C − C_bg| + 3 σ_rate

S_max/V = 3 index/s, λ_max = 20 h⁻¹. Observed rate from medians of 3 samples over a 5-sample window.
Rate above envelope ⇒ sensor-local interference (swab, breath on sensor), not room air.

---

## 4. Learned prior (L2) — `prior.EnsemblePrior`

Features x_k: lags of fused log C (10…60 s, relative to now), T, RH, ΔT_30s, ΔRH_30s.
Target: Δ log C_{k+1}. Ensemble of M = 5 bootstrap MLPs (16-16):

    μ_k = (1/M) Σ f_m(x_k)           σ_k = std_m f_m(x_k)
    z_prior,k+1 = log Ĉ_k + μ_k

OOD score (Mahalanobis on standardised features z):

    D²_k = zᵀ Σ⁻¹ z

---

## 5. Fusion — extended Kalman filter (L4) — `fusion.TrustEKF`

State x = [C, ld]ᵀ.

Predict:

    F = diag(e^{−λΔt}, 1),   Q = diag(q_c² Δt, q_ld² Δt),   P⁻ = F P Fᵀ + Q

Measurements (log units):

| channel | z | h(x) | H |
|---|---|---|---|
| raw | log C with CF = 1 | log C + b·ld | [1/C, b] |
| physics | log C with CF(T,RH) | log C + b·ld | [1/C, b] |
| prior | z_prior | log C | [1/C, 0] |

Trust-scaled noise (core equation):

    R_s,k = R_s,0 / max(τ_s,k, ε)      R_0 = 0.15² (sensor), max(0.05², σ_k² + σ²_fused) (prior; built on fused history, so its uncertainty is carried), ε = 10⁻³

Update (Joseph form):

    ν = z − h(x⁻),  S = H P⁻ Hᵀ + R,  K = P⁻ Hᵀ S⁻¹
    x = x⁻ + K ν,   P = (I − KH) P⁻ (I − KH)ᵀ + K R Kᵀ

Physics constraint (room cannot be below outdoor background):

    C ← max(C, 0.97 C_bg)                                    every step
    if median_60s(C_obs) < 0.97 C_bg :  ld ← ld − (log 0.97 C_bg − log median)/(60 b)

Drift moves only on a sustained 60-s median, rate-limited, so noisy single samples cannot ratchet it one way.
Initial P_ld = 0.005² (freshly calibrated sensor, d = 1).

---

## 6. Trust estimator (L3) — `trust.TrustEstimator`

Per-channel normalised innovation squared (nominal R, avoids feedback):

    NIS_s = ν_s² / (H P⁻ Hᵀ + R_s,0)
    N̄_s ← (1−α) N̄_s + α NIS_s,      α = 0.3
    p_nis,s = max(0, N̄_s / χ²₁(0.95) − 1),   χ²₁(0.95) = 3.841

Penalties:

| penalty | raw | physics | prior |
|---|---|---|---|
| innovation p_nis | ✓ | ✓ | ✓ |
| rate plausibility 2·max(0, r/r_max − 1) | ✓ | ✓ | |
| ADC fault (5) | ✓ | ✓ | |
| T/RH envelope (per 5 °C / 10 %) | | ✓ | |
| DHT11 lag max(0, |dT/dt|/0.05 − 1) + max(0, |dRH/dt|/0.2 − 1) | | ✓ | |
| CF disagreement max(0, |log CF_ds − log CF_refit|/0.15 − 1) | | ✓ | |
| OOD max(0, D²/23.2 − 1) | | | ✓ |
| ensemble spread max(0, σ/0.02 − 1) | | | ✓ |

Target and smoothing (instant drop, slow recovery):

    τ*_s = exp(−Σ penalties_s)
    τ_s ← τ_s + a (τ*_s − τ_s),   a = 1.0 if τ* < τ else 0.05

Re-anchoring: sensor plausible ≥ 30 s but N̄_phys > 3χ² ⇒ model wrong, not sensor:
reset N̄, inflate P_CC by 100².

---

## 7. Uncertainty (L5) — `conformal.Conformal`

Total σ in log units (all terms in log space, conformal applied to the total):

    σ²_total = P_CC / C²  +  ((log C − log C_ref) σ_b / b)²  +  σ²_sys

σ_sys = systematic (time-correlated) model error — correction-curve mismatch, DHT11 offset — which the
filter cannot average down because it is not white. Estimated on half A of the clean held-out data:

    σ²_sys = max(0, mean(r²) − median(σ²_EKF))

Conformal q is then fitted on half B.

σ_sys is scaled by trust, because when both sensor channels are distrusted the fused estimate rests on the
process model and prior alone:

    σ_sys,k = σ_sys / √max(max(τ_raw, τ_phys), 0.05)

On clean data τ ≈ 1, so clean bands are unchanged.
Split conformal on normalised residuals of held-out data:

    s_i = |y_i − log Ĉ_i| / σ_total,i
    q = quantile_{⌈(n+1)(1−α)⌉/n}(s),   α = 0.1

Interval:

    [ Ĉ · e^{−q σ_total},  Ĉ · e^{+q σ_total} ]

Target y: reference-free consensus (31-s rolling median of physics channel) or ground truth when available.
Real-node q: not reported until a post-burn-in run of ≥ 24 h (a 9-min pre-burn-in value is meaningless).

---

## 8. Evaluation metrics — `evaluate.py`

- RMSE in log units vs truth / consensus, split clean vs fault.
- Sensor-fault alarm: **both** sensor channels distrusted, max(τ_raw, τ_phys) < θ, held ≥ 2 samples,
  latched 10 s (raw-only drop = cross-sensitivity, physics-only drop = T/RH model issue).
  Previous rule min(τ_raw, τ_phys) < θ reported alongside on the **same** labels (spike, stuck, noise).
  θ tuned on validation seed only; reported on held-out test seeds.
- High-frequency noise test: var(Δz_phys over 30 s) / adaptive baseline > 4 → penalty on both sensor
  channels (catches variance change that the NIS mean misses, since R is dominated by slow error).
- PICP (coverage of 90 % interval) and mean log width, before vs after drift.
- Decorrelation |corr(Ĉ, T)| on clean segments.
- Ablation: raw, datasheet physics, refit physics, prior, fixed weights, adaptive.
  All modes share the same bundle: identical R, Q, σ_sys, R0, CF — only channel set / trust differ.

---

## 9. Parameter table (current)

| symbol | value | source |
|---|---|---|
| a, b | 116.02 ± 5.3, −2.816 ± 0.066 | consensus: GeorgK, MQUnifiedsensor, Kalra (coefficients.py) |
| RL | 10 kΩ | assumed — measure |
| R0 | 112.3 kΩ | provisional clean-air fit |
| C_ref, C_bg | 426 | NOAA MLO 2026-09-13 |
| λ | 1 h⁻¹ | default — fit by E741 decay |
| q_c, q_ld | 1.0, 2·10⁻⁶ | tuned on simulation |
| R_raw, R_phys, R_prior | 0.15², 0.15², 0.05² | matched to observed physics-channel error |
| σ_sys | ≈ 0.17 (synthetic) | estimated on clean held-out half |
| q (conformal) | not reported (real node) | needs ≥ 24 h post-burn-in log |
| σ_b | 0.066 | digitisation spread (n = 2), not unit-to-unit |

## 10. Assumptions / limits

- Well-mixed single zone; λ is effective.
- MQ135 non-selective → C is CO2-equivalent index.
- Raw and physics channels share one Rs → correlated errors treated as independent.
- Upward drift indistinguishable from real rise without reference.
- MQ7 not in model (constant-heater operation invalidates its curve).

## 11. Results (synthetic, non-circular)

### 11.1 Final claim set (synthetic benchmark frozen)

| claim | evidence |
|---|---|
| Accuracy: ≈ 9 % lower error than fixed-weight fusion | 0.078 vs 0.085 RMSE log, paired p = 0.0008, 5 seeds; stable across b ± σ_b |
| Calibrated uncertainty | 90 % bands: clean ≥ 90 % (1.00), local faults 1.00; conformal q ≈ 1.0 |
| Sensor-fault detection | F1 ≈ 0.79; latency spike 1 s, stuck 4 s; attribution to sensor 20 / 20 |

Documented limits:
- Drift unobservable within a day on one node; drift coverage ≈ 65 %. ABC (24 h rolling-min re-anchor)
  and time-since-anchor band growth mitigate.
- Noise-type faults poorly detected (×3 noise: 1 / 5).
- Cross-sensitivity absorbed by the T/RH correction rather than diagnosed.
- No absolute ppm accuracy (relative CO2-equivalent index, no reference instrument).

### 11.2 Accuracy — 5 independent test seeds, b_true = b, same bundle for both modes

| | fixed-weight | trust-adaptive |
|---|---|---|
| RMSE log, mean ± std | 0.085 ± 0.009 | 0.078 ± 0.008 |
| paired difference | | 0.0078 ± 0.0019, t(4) = 9.21, p = 0.0008 |

Single test seed with b_true swept over b ± σ_b: adaptive 0.091–0.094 vs fixed 0.104–0.107, gap stable.

### 11.3 Uncertainty — 90 % band coverage by segment (b_true = b, test seed 7, adaptive)

| segment | n | coverage | mean log width |
|---|---|---|---|
| clean | 5120 | 1.000 | 0.346 |
| spike (swab) | 100 | 1.000 | 1.432 |
| stuck | 60 | 1.000 | 1.442 |
| noise ×3 | 120 | 1.000 | 0.604 |
| drift 0.2 %/min | 1800 | 0.671 | 0.341 |

Drift coverage vs k_d (single test seed, b_true ∈ {b − σ, b, b + σ}):

| k_d [log²/s] | origin | drift coverage | clean coverage | local-fault coverage | mean log width |
|---|---|---|---|---|---|
| 3·10⁻¹¹ = (b·q_ld)² | consistent with filter | 0.61–0.67 | 1.00 | 1.00 | 0.35–0.38 |
| 9·10⁻⁶ | back-calculated from the injected drift (tuned to test) | 1.00 | 1.00 | 1.00 | 0.50–0.53 |

Reading: the mechanism works — with a k_d that matches the true drift rate the bands cover drift at a ≈ 40 %
width cost. But with k_d from the filter's own model, drift coverage stays ≈ 65 %. Reported as the known
cost of single-node operation, with time-since-anchor growth as the mitigation once k_d is measured from
multi-day ABC logs. Accuracy and detection are unaffected by k_d (bands only).

Clean-segment band width vs time since anchor (b_true = b, test seed 7):

| minutes since anchor | 0 | 15 | 30 | 45 | 60 | 75 | 90 |
|---|---|---|---|---|---|---|---|
| width, k_d = 3·10⁻¹¹ | 0.341 | 0.347 | 0.350 | 0.350 | 0.345 | 0.341 | 0.341 |
| width, k_d = 9·10⁻⁶ | 0.364 | 0.412 | 0.455 | 0.487 | 0.516 | 0.544 | 0.558 |

Clean coverage is 1.00 in every bin for both, so the extra width of k_d = 9·10⁻⁶ is pure over-coverage
on clean data (+53 % width by 90 min, and the drift term √(k_d t) ≈ 0.28 at 90 min already exceeds σ_sys).

Provenance and units: k_d = 9·10⁻⁶ was chosen from the test data (back-calculated from the injected
drift so drift coverage reaches ≈ 90 %) — it is tuned on test and is shown only to demonstrate the
mechanism. Units: q_ld is a standard deviation [log Rs / √s]; k_d is a variance rate [log C² / s];
consistent conversion k_d = (b·q_ld)². k_d = 9·10⁻⁶ ≡ q_ld ≈ 1.1·10⁻³ /√s, ≈ 500× the filter's q_ld
(deliberate, not a unit error).

Conformal q = 0.98–1.01. Bands widen ≈ 4× during local faults (1/√trust scaling) and cover them;
clean bands are slightly conservative (σ_sys estimated on a calibration day with larger T/RH swings than
the test day — calibration/test shift). Drift is under-covered: trust never drops under slow drift, and the
error is the unobservable bias of §2.4.

### 11.4 Diagnosis — sensor-fault alarm (5 test seeds, θ = 0.3 tuned on validation seed only)

Primary rule: alarm when either sensor channel is distrusted, min(τ_raw, τ_phys) < θ.
Variant: both channels + 2-sample confirmation + 10 s latch. Same labels (spike, stuck, noise) for both.

| | primary (either channel) | variant (both + hysteresis) |
|---|---|---|
| precision | 0.860 ± 0.118 | 0.784 ± 0.034 |
| recall | 0.729 ± 0.027 | 0.729 ± 0.013 |
| F1 | 0.785 ± 0.052 | 0.755 ± 0.009 |

Per-event latency (identical across seeds except the noise fault):

| fault | primary | variant | detected |
|---|---|---|---|
| spike (swab) | 1 s | 2 s | 5 / 5 seeds |
| stuck | 4 s | 5 s | 5 / 5 |
| noise × 3 | 41 s | 42 s | **1 / 5** |

The equal latency spread of the two rules is not an aggregation bug: the variant is exactly +1 s per event
(confirmation sample), and the spread is dominated by the single detected noise event. The noise fault is
the recall limit (missed in 4 of 5 seeds). Limitation, not tuned further: the ×3 noise is within what
the bands already absorb (σ_sys dominates the noise variance), so trust barely moves and the
high-frequency variance test rarely fires.

Channel attribution — channel state during each event (τ < θ), 25 events:

| true class \ attributed | both channels | raw only | physics only | none |
|---|---|---|---|---|
| sensor fault (spike, stuck, noise) | **20** | 0 | 0 | 0 |
| cross-sensitivity (T/RH swing) | 0 | 0 | 1 | 4 |

- Sensor-level faults are attributed to the sensor (both channels) in 20 / 20 events.
- Cross-sensitivity is **absorbed, not diagnosed**: raw-only lows occurred in 0 / 5 T/RH swings. The T/RH
  correction removes enough of the signal shift that no fault is raised (4 / 5 no flag; 1 / 5 physics-only).
  This is a positive result for the physics model, not a detection success. No claim is made that the system
  detects cross-sensitivity events; attribution is claimed only for sensor-level faults.
- Stability: variant F1 std 0.009 vs 0.052. Levene W = 1.91, p = 0.20 (not significant); F-ratio 35.2,
  p = 0.002 (assumes normality). With n = 5 this is a trend, not a result. The one seed where the variant helps
  (seed 104: primary precision 0.66) is the physics-only false alarm during the T/RH swing.

Earlier F1 0.46 → ≈ 0.78: the gain comes from the retuned noise model and the noise-variance test,
not from the alarm rule.

### 11.5 Fix history (uncertainty)
1. Drift state absorbed real room rises (bias +0.42 log): q_ld 2·10⁻⁴ → 2·10⁻⁶, P_ld,0 0.05² → 0.005².
2. Background projection ratcheted on noise: 60-s median, rate-limited.
3. Sensor R 0.03² vs observed error ≈ 0.19: R = 0.15².
4. Prior double-counted fused history: prior R += σ²_fused.
5. Time-correlated error: explicit σ_sys (q ≈ 66 → ≈ 1.0).
6. Logger restart (Uno resets, millis restarts): segment break, not a fault.

## 12. Real hardware session (2026-09-25, single node, provisional)

Node: Arduino Uno (CH340), MQ135 on A2, DHT11 on A0 (MQ7 on A1 logged, unused). Powered since 06:22
(≈ 5.5 h at test time: short burn-in). Load resistor not measured (10 kΩ assumed; cancels in the index).
Outputs: `results/session/` (trace.csv, events.csv, session.png), bundle `models/bundle.joblib`.

| step | what was done | result |
|---|---|---|
| DHT11 check | vs outdoor 33 °C | DHT11 31.5 °C indoors: no heater bias evident (rough check) |
| Clean-air reference | open-window session 11:10–11:25 | window air got **dirtier** (ADC 80 → 88) and T did not change → board likely not in outdoor air; rejected. Used cleanest 10 min of the day instead (10:15–10:25, ADC 71): ABC assumption, not a verified outdoor reference |
| Calibration | R0, σ_sys, conformal | R0 = 232 kΩ (RL 10 kΩ assumed), σ_sys = 0.151, q = 1.25; datasheet CF (T range too small to refit) |
| λ decay test | 3 people ≈ 4 min, then 6 min closed | inconclusive (λ fit negative, R² 0.61): occupancy too short, index barely rose. Default λ = 1 h⁻¹ kept |
| Perfume (60 s, 5–10 cm) | sensor-local interference | flagged at onset (≈ 0–1 s); both channels trust 0 → attributed to sensor; raw index → 2.4·10⁶ (curve meaningless there), fused held at ≈ 917 (pre-event 906); band widened to 1.69 log |
| Hot air (laptop vent, 2 min) | T/RH swing +8.5 °C, −18 %RH | raw shift −0.108 log, physics shift −0.105 log: the datasheet correction removed only ≈ 3 % of the shift. Fused shift −0.024 because trust dropped (raw 0.67, physics 0.50 median). Confounded: the room was still clearing perfume (ADC falling 120 → 93) |

Coverage of the 90 % band against the reference-free consensus (31-s median of physics channel):
undisturbed room 10:25–11:10: **0.62** (width 0.38 log). Below the 0.90 target — the index rose slowly
over the morning (burn-in / room change, unobservable on one node, §2.4) and the band was calibrated on
only ≈ 1.3 h. During the perfume event coverage is not meaningful (the consensus follows the raw spike).

Data-quality fix found on hardware: the DHT11 returned 6 corrupt reads (T ≈ 0.5–0.9 °C, RH 0) flagged as
valid; the pipeline now rejects physically implausible T/RH and holds the last good reading.

### 12.1 Real-hardware conclusions
- **Confirmed on hardware:** sensor-local interference detection and attribution; fused estimate refuses
  the raw spike; uncertainty widens while the sensor is distrusted.
- **Contradicts synthetic result:** cross-sensitivity is **not** absorbed on this node — the datasheet
  T/RH correction barely moves the reading (≈ 3 % of the shift). The synthetic simulator was generated from
  the same correction curve, which made absorption look better than it is. A refit CF on multi-day data
  with a wider T/RH range is required.
- **Not achieved:** verified clean-air reference, λ, 90 % real-node coverage, 24 h burn-in.

## 13. LSTM deep-ensemble prior (replaces MLP ensemble as default)

`prior.LSTMPrior`: M = 5 LSTMs, input 12 × 3 sequence (Δlog C relative to now, T, RH; 60 s at 5-s stride,
standardised), LSTM(3 → 32) → linear → next-step Δlog C. Adam 1e-3, batch 256, ≤ 8 epochs, early stopping on
10 % validation; bootstrap resample per member. μ = mean over members, σ_epistemic = std; OOD = Mahalanobis on
the 10-d summary vector (as before). Selected by `Config.prior_kind`.

| 5 test seeds | MLP ensemble | LSTM ensemble |
|---|---|---|
| prior one-step RMSE | 0.0735 ± 0.0042 | 0.0701 ± 0.0034 (paired p = 0.0012) |
| fused RMSE | 0.0776 ± 0.0079 | 0.0771 ± 0.0081 (p = 0.06, n.s.) |
| coverage clean / local | 1.00 / 1.00 | 1.00 / 1.00 |
| ms / step | 0.7 | 1.9 |
| real node undisturbed coverage | 0.62 | 0.70 |

## 14. PHYS-TRUST 2.0 Phase 1 (validation + intelligence)

- **Reference validation (UCI Air Quality, `tools/validate_uci.py`)**: tin-oxide MOX (PT08.S1) vs certified CO analyser,
  10-month test. Adaptive fusion RMSE log 0.493, R² 0.62, 90 % coverage 0.91 (fixed weights 0.506 / 0.60 / 0.89;
  static formula no T/RH 0.448 / 0.65). Process noise q_c tuned on validation only (12). Real-data fault detection F1 0.14.
- **Prior admission test (`prior.admission_test` + fused-level check)**: stage 1 one-step skill vs persistence > 0;
  stage 2 fused validation error with prior < without. UCI LSTM: skill 0.04 (pass), fused 0.680 vs 0.574 (reject).
- **Architectures** (synthetic, skill vs persistence): MLP −12.9 (rejected), LSTM 0.616, TCN 0.629, Transformer 0.604;
  fused RMSE identical for sequence models (0.0771).
- **Trust state machine (`states.py`)**: TRUSTED / UNCERTAIN / CALIBRATION_NEEDED / OOD / PHYSICS_MISMATCH / SENSOR_FAULT,
  each with an action; P(OOD) = 1 − Π(1 − sigmoid(4(s_i − 1))) over distance, disagreement, envelope, novelty.
  Synthetic: spike 98 % / stuck 93 % → SENSOR_FAULT; clean 15 % false OOD; per-state coverage 0.98–1.00.
- **Adaptive compute**: 1 member when TRUSTED → 67 % fewer member evaluations, 3.1 → 1.6 ms/step, same accuracy.
- **Provenance**: node_id, calib_version, quality_score, state on every row.
- **Weather validation (`reference.py`, Open-Meteo Chennai)**: 5 h so far — T bias −0.4 °C, RMSE 2.5 °C; RH bias +5.7 %.
