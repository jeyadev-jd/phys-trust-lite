const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, PageBreak, LevelFormat, TableOfContents,
  Footer, PageNumber, Header,
} = require("docx");

const P = "C:\\Users\\Jeyadev\\Documents\\phys-trust-lite";
const OUT = path.join(P, "docs", "PHYS-TRUST_Lite_Report.docx");
const W = 9026; // A4 text width in DXA with 1" margins

// ---------- helpers
const t = (s, o = {}) => new TextRun({ text: s, ...o });
const para = (runs, o = {}) => new Paragraph({ children: Array.isArray(runs) ? runs : [t(runs)],
  spacing: { after: 120, line: 300 }, alignment: AlignmentType.JUSTIFIED, ...o });
const h1 = (s) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [t(s)], spacing: { before: 360, after: 160 } });
const h2 = (s) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [t(s)], spacing: { before: 240, after: 120 } });
const eq = (s) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 120 },
  children: [t(s, { font: "Cambria Math", size: 22 })] });
const bullet = (runs) => new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 60 },
  children: Array.isArray(runs) ? runs : [t(runs)] });
const num = (runs, ref = "n") => new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 60 },
  children: Array.isArray(runs) ? runs : [t(runs)] });
const caption = (s) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [t(s, { italics: true, size: 20 })] });
const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };

function table(head, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (s, i, hdr) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: hdr ? { fill: "DCE6F1", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [t(String(s), { bold: hdr, size: 20 })] })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: head.map((s, i) => cell(s, i, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((s, i) => cell(s, i, false)) }))],
  });
}
function image(file, wPx) {
  const f = path.join(P, file);
  if (!fs.existsSync(f)) return para(`[figure missing: ${file}]`);
  const buf = fs.readFileSync(f);
  const iw = buf.readUInt32BE(16), ih = buf.readUInt32BE(20); // PNG header
  return new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({
    type: "png", data: buf, transformation: { width: wPx, height: Math.round(wPx * ih / iw) } })] });
}
const gap = () => new Paragraph({ children: [t("")] });

// ---------- content
const title = [
  new Paragraph({ spacing: { before: 2400 }, alignment: AlignmentType.CENTER,
    children: [t("PHYS-TRUST Lite", { bold: true, size: 56 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 }, children: [t(
    "Trust-Adaptive, Physics-Informed Deep-Ensemble Estimation of Air Quality on a Single Low-Cost Sensor Node",
    { size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [t("Project Report — Deep Learning", { size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 800 }, children: [t("Submitted by: [Your name, roll number]", { size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [t("Guide: [Guide name]", { size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [t("[Department, Institution]", { size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 400 }, children: [t("September 2026", { size: 24 })] }),
  new Paragraph({ children: [new PageBreak()] }),
];

const abstract = [
  h1("Abstract"),
  para("Low-cost metal-oxide gas sensors such as the MQ135 are widely deployed but unreliable: they are cross-sensitive to temperature and humidity, drift over time, respond to many gases, and are usually operated without any reference instrument. This project builds PHYS-TRUST Lite, a single-node estimator that fuses three sources of information — the raw sensor reading, a physics model (sensor power-law response with temperature/humidity correction, embedded in a single-zone mass-balance model of the room), and a learned deep-ensemble prior built from five LSTM recurrent networks — and decides online how much to trust each one. An interpretable trust estimator, driven by Kalman innovation statistics, physical-plausibility bounds, sensor-validity envelopes and out-of-distribution scores, scales each source's measurement noise inside an extended Kalman filter. Uncertainty is reported as a 90 % band combining filter, coefficient, systematic and drift terms, calibrated by split conformal prediction."),
  para("On a non-circular synthetic benchmark (true sensor exponent swept ±σ, five independent test seeds) the trust-adaptive estimator reduced error by about 9 % relative to fixed-weight fusion (RMSE 0.078 vs 0.085 in log units, paired t-test p = 0.0008), covered clean and locally faulted segments with its 90 % band, and detected sensor-level faults with F1 ≈ 0.79 and 1–4 s latency, attributing 20 of 20 events to the sensor. On real hardware (Arduino Uno, MQ135, DHT11) the system flagged sensor-local perfume interference at onset and held its estimate while the raw reading diverged by four orders of magnitude. Replacing an MLP ensemble with the LSTM ensemble reduced the prior's one-step prediction error by 4.5 % (p = 0.0012) at equal fused accuracy. Validated against a certified CO analyser on the UCI Air Quality dataset (a tin-oxide sensor over ten months), the trust-adaptive estimator reached R² = 0.62 with 90 % band coverage of 0.91, beating fixed-weight fusion, and a two-stage admission test automatically disabled a learned prior that did not help. A trust state machine converts the signals into explicit operating states and actions, and adaptive compute halves inference cost without loss of accuracy. The work also documents what one uncalibrated node cannot do: slow drift is unobservable within a day, the datasheet temperature/humidity correction proved weak on the real node, and no absolute ppm accuracy is claimed. A live dashboard streams the estimator on the running sensor for demonstration."),
  new Paragraph({ children: [new PageBreak()] }),
];

const toc = [
  new Paragraph({ children: [t("Contents", { bold: true, size: 32 })], spacing: { after: 200 } }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  para([t("(In Word: right-click the table → Update Field to fill page numbers.)", { italics: true, size: 18 })]),
  new Paragraph({ children: [new PageBreak()] }),
];

const intro = [
  h1("1. Introduction"),
  para("Air quality indoors drives comfort, cognition and health, and cheap sensors make monitoring affordable. The MQ135 is a tin-dioxide chemiresistor whose resistance falls in the presence of reducing gases. Hobby-grade modules cost very little and are used in thousands of student and IoT projects, but their readings are routinely converted to “ppm” with a single datasheet curve and presented without any statement of reliability."),
  para("Three physical facts make this unsafe. First, the sensor responds to temperature and humidity as well as gas. Second, its baseline resistance drifts with ageing and burn-in. Third, local interference — a hand-sanitiser swab, a breath directly on the sensor — produces large spikes that do not reflect room air. A useful single-node system therefore has to answer, continuously and without a reference instrument: how much should I trust my own reading right now?"),
  h2("1.1 Objectives"),
  num("Fuse the raw reading, a physics model and a learned deep-ensemble prior in one estimator."),
  num("Estimate, online and interpretably, how much each source should be trusted."),
  num("Report calibrated uncertainty, not just a point value."),
  num("Detect and attribute sensor-level faults."),
  num("Evaluate honestly — with ablations, non-circular synthetic tests and a real hardware session — and document limits."),
  num("Provide a live dashboard for demonstration."),
];

const related = [
  h1("2. Related Work"),
  para("Physics-informed and uncertainty-aware air-quality estimation is an active area, but existing work is either spatial (many stations) or relies on a reference instrument. Table 1 summarises representative work and why it does not cover the single-node, reference-free case."),
  table(["Area", "Representative work", "Gap for this project"], [
    ["PDE / physics-informed AQ forecasting", "CTENet (NeurIPS 2025); AirPhyNet (2024)", "Needs multi-station networks and meteorology"],
    ["Physics + neural fusion with uncertainty", "NeuroDDAF (2026): evidential fusion gate", "City scale; not a single cheap node"],
    ["Trust-based sensor calibration", "Mahajan & Helbing, npj Clim. Atmos. Sci. (2025)", "Trust from network consensus; PM2.5 optical sensors"],
    ["Indoor mass-balance PINNs", "Office CO2 PINN (Building & Environment, 2025)", "NDIR/reference data; no MOX cross-sensitivity or trust"],
    ["MOX T/RH correction", "Abdullah et al., Sensors (2022); Kalra et al. (2016)", "Static correction; no uncertainty or fusion"],
    ["Kalman filtering of low-cost gas sensors", "Christakis et al., Electronics (2024)", "No physics process model, no reliability gating"],
    ["Uncertainty under shift", "GeoConformal vs MC-Dropout, Atmosphere (2026)", "Network PM2.5; shows model uncertainty miscalibrates under shift"],
  ], [2400, 3313, 3313]),
  caption("Table 1. Representative related work."),
  h2("2.1 Research gap"),
  para("Existing trust-aware and physics-informed methods assume either spatially distributed sensors or access to a reference instrument. Little work addresses how a single low-cost MOX node, with no reference, can decide when to trust its raw signal versus a physics-based correction versus a learned prior, and whether its uncertainty stays calibrated under interference, cross-sensitivity and distribution shift. A targeted literature search found no paper combining these elements on one hobby-grade node; this is stated as “not found”, not as proven novelty."),
];

const system = [
  h1("3. System Overview"),
  h2("3.1 Hardware"),
  table(["Component", "Role", "Connection"], [
    ["Arduino Uno (CH340 clone)", "1 Hz sampling, 10× ADC oversampling, serial CSV at 115200 baud", "USB (COM8)"],
    ["MQ135", "Metal-oxide gas sensor (reducing gases, CO2-equivalent index)", "A2 (analog)"],
    ["DHT11", "Temperature ±2 °C, humidity ±5 %RH (correction inputs)", "A0 (used as digital)"],
    ["MQ7", "CO sensor; logged only (needs heater cycling, not used by estimator)", "A1 (analog)"],
  ], [2400, 4626, 2000]),
  caption("Table 2. Hardware of the sensor node."),
  para("Each analog read is preceded by a discarded read so the ADC multiplexer settles; without it the two MQ channels showed crosstalk. A standalone logger process writes every sample to a CSV file, reconnects automatically after a USB disconnect, and is restarted by a Windows logon task, so logging is independent of the dashboard."),
  h2("3.2 Software pipeline"),
  table(["Layer", "Function", "Module"], [
    ["L0 Signal", "ADC → sensor resistance; saturation / stuck / corrupt-DHT flags", "sensor_physics.py"],
    ["L1a Sensor physics", "Power-law response, T/RH correction, R0 calibration", "sensor_physics.py"],
    ["L1b Process physics", "Single-zone mass balance, plausibility envelope, decay fit", "process_physics.py"],
    ["L2 Learned prior", "Deep ensemble of 5 LSTMs on a 60-s sequence: prediction, epistemic spread, OOD score", "prior.py"],
    ["L3 Trust estimator", "Penalties → trust ∈ [0,1] per source", "trust.py"],
    ["L4 Fusion", "Extended Kalman filter, state [C, drift], trust-scaled noise", "fusion.py"],
    ["L5 Uncertainty", "σ_total terms + split-conformal 90 % band; ABC re-anchoring", "conformal.py, pipeline.py"],
    ["Interface", "Live dashboard, optional local-LLM (gemma3:4b) event explanations", "tools/dashboard.py"],
  ], [2000, 5026, 2000]),
  caption("Table 3. Processing layers."),
];

const method = [
  h1("4. Methodology"),
  h2("4.1 Sensor physics (measurement model)"),
  para("The divider voltage gives the sensor resistance; the ADC is ratiometric, so supply variations cancel:"),
  eq("Rs = RL · (1023 / ADC − 1)"),
  para("The datasheet power law, corrected for temperature and humidity by a factor CF(T,RH), maps resistance to a CO2-equivalent index:"),
  eq("C = a · ( Rs / (R0 · CF(T,RH)) )^b ,   a = 116.02,  b = −2.816"),
  para("a and b are the consensus of published coefficient sets (GeorgK/Gironi, MQUnifiedsensor, Kalra). R0 is fixed by a clean-air reference at C_ref = 426 ppm (NOAA Mauna Loa weekly mean). Once R0 is anchored, a cancels exactly and only b matters:"),
  eq("log C − log C_ref = b · [ log(Rs/CF) − log(Rs_ref/CF_ref) ]"),
  para("so the coefficient uncertainty σ_b = 0.066 (digitisation spread of the published sets, n = 2) propagates as σ(log C) = |log C − log C_ref| · σ_b / |b|. Because raw and physics channels share b, this term is added once to the fused variance, not per channel."),
  h2("4.2 Process physics (state model)"),
  para("The room is modelled as a single well-mixed zone:"),
  eq("dC/dt = S(t)/V − λ (C − C_bg)"),
  eq("C(k+1) = C_bg + (C(k) − C_bg)·e^(−λΔt) + (S/(Vλ))·(1 − e^(−λΔt)) + w(k)"),
  para("The unknown source S is treated as process noise. The model yields a plausibility envelope: a room cannot change faster than |dC/dt|max = S_max/V + λ_max|C − C_bg|. A rise or fall faster than this, recovering on the sensor's time constant rather than the room's, indicates sensor-local interference. λ can be identified by the ASTM E741 decay method, ln(C − C_a) = ln(C0 − C_a) − λt."),
  h2("4.3 Learned prior: LSTM deep ensemble"),
  para("The deep-learning component is a deep ensemble (Lakshminarayanan et al., 2017) of M = 5 long short-term memory (LSTM) recurrent networks. Each member reads the last 60 s of history as a sequence of 12 steps (5-s stride) with three standardised channels per step — change in fused log-index relative to now, temperature and humidity — through an LSTM layer with 32 hidden units, followed by a linear head that predicts the next-step change in log-index. Each member is trained with Adam (learning rate 10⁻³, batch 256, up to 8 epochs, early stopping on a 10 % validation split) on its own bootstrap resample of the training data, which makes the members disagree where data are scarce."),
  table(["Item", "Setting"], [
    ["Input", "12 × 3 sequence: Δlog C, T, RH (60 s at 5-s stride), standardised"],
    ["Network", "LSTM(3 → 32) → last hidden state → Linear(32 → 1)"],
    ["Target", "next-step change in fused log-index (scaled)"],
    ["Training", "Adam 1e-3, batch 256, ≤ 8 epochs, early stopping on 10 % validation, MSE loss"],
    ["Ensemble", "5 members, bootstrap resampling, different seeds"],
    ["Outputs", "mean (prediction), std across members (epistemic uncertainty)"],
    ["OOD score", "Mahalanobis distance (Ledoit-Wolf covariance) of a 6-d summary: lag level, lag slope, T, RH, ΔT, ΔRH"],
  ], [2400, 6626]),
  caption("Table 4a. LSTM deep-ensemble prior."),
  eq("μ = (1/M) Σ LSTM_m(x₁…x₁₂),     σ_epistemic = std_m LSTM_m(x₁…x₁₂),     D² = zᵀ Σ⁻¹ z"),
  para("An MLP ensemble (two hidden layers of 16 units on the 10-d summary vector) is kept as a baseline and compared in Section 6.1a."),
  para("The ensemble mean enters the Kalman filter as a third measurement; its spread and the Mahalanobis distance D² of the input from the training distribution are used to reduce its trust when inputs are unfamiliar. The prior is trained on the filter's own fused history, so its noise includes the fused variance to avoid double-counting."),
  h2("4.4 Trust estimator"),
  para("For each source s ∈ {raw, physics, prior} a set of interpretable penalties is computed and mapped to a trust score:"),
  eq("τ*_s = exp( − Σ penalties_s ),   τ_s ← τ_s + a (τ*_s − τ_s),  a = 1 (drop) or 0.05 (recover)"),
  table(["Penalty", "Raw", "Physics", "Prior"], [
    ["Normalised innovation squared (EWMA, χ² threshold)", "✓", "✓", "✓"],
    ["Rate above mass-balance envelope", "✓", "✓", ""],
    ["ADC saturation / stuck value", "✓", "✓", ""],
    ["High-frequency noise-variance ratio", "✓", "✓", ""],
    ["T/RH outside fitted envelope; DHT11 lag", "", "✓", ""],
    ["Datasheet vs refitted correction disagreement", "", "✓", ""],
    ["Out-of-distribution D²; ensemble spread", "", "", "✓"],
  ], [5026, 1333, 1334, 1333]),
  caption("Table 4. Trust penalties per source."),
  h2("4.5 Fusion: extended Kalman filter"),
  para("The state is x = [C, ld], where ld is the log of a slow drift multiplier on Rs. Raw and physics channels observe log C + b·ld; the prior observes log C. The core of the method is that each source's measurement noise is scaled by its trust, so the filter performs the weighting:"),
  eq("R_s = R_s,0 / max(τ_s, ε)"),
  para("Updates use the Joseph form for numerical stability. A physical constraint prevents the room estimate from falling below outdoor background; a sustained (60-s median) deficit is attributed to drift at a rate-limited pace."),
  h2("4.6 Uncertainty quantification"),
  eq("σ²_total = q² · ( σ²_EKF + σ²_b + σ²_sys / max(τ, 0.05) ) + k_d · (t − t_anchor)"),
  para("σ²_sys is a time-correlated model error (correction mismatch, DHT11 offset) that averaging cannot reduce, estimated on one half of clean held-out data; the conformal factor q is fitted on the other half. σ_sys is inflated when both sensor channels are distrusted, so the band widens during faults. The last term lets the band grow with time since the last re-anchoring, because slow drift is unobservable on one node."),
  h2("4.7 Automatic baseline calibration (ABC)"),
  para("As in commercial NDIR sensors, the lowest per-minute median over a rolling 24-h window is assumed to be outdoor background, and R0 is re-anchored to it. This requires the room to be aired at least once per window. A manual re-zero (window open) is also provided."),
];

const phase2method = [
  h2("4.8 Trust state machine and action policy"),
  para("Numeric trust is converted into explicit operating states, each with an action: TRUSTED (monitor), UNCERTAIN (run full ensemble), CALIBRATION_NEEDED (request re-zero), OOD (cross-check external reference), PHYSICS_MISMATCH (diagnose model or T/RH input) and SENSOR_FAULT (reject raw reading, alert a human if it persists for 60 s). A 5-s dwell provides hysteresis; sensor faults escalate immediately."),
  para("The probability that an input is out of distribution combines four signals — Mahalanobis distance of the features, ensemble disagreement, T/RH outside the calibrated envelope, and temporal novelty of the physics signal (robust z-score over 2 min) — as P(OOD) = 1 − Π(1 − p_i), with p_i = sigmoid(4(s_i − 1)). Calibration confidence decays with time since the last anchor, apparent drift and reference disagreement."),
  eq("calibration confidence = exp(−t/τ) · exp(−|log d|/0.1) · exp(−reference disagreement)"),
  para("Adaptive compute: in the TRUSTED state only one ensemble member is evaluated; in every other state all five are used, so compute rises with uncertainty. Every output row carries provenance: node identifier, calibration version, quality score and state."),
  h2("4.9 Learned-prior admission test"),
  para("A learned prior is only admitted if it helps. Stage 1: on held-out training data its one-step prediction must beat persistence (skill = 1 − MSE_model/MSE_persistence > 0). Stage 2: fusion with the prior must beat fusion without it on a held-out validation window. Otherwise the prior is disabled automatically. The test data are never used."),
];

const setup = [
  h1("5. Experimental Setup"),
  h2("5.1 Synthetic benchmark"),
  para("A simulator generates a room (occupancy sources, mass balance), the MQ135 response with a true temperature/humidity factor that deliberately differs from the datasheet, DHT11 lag, offset and quantisation, and 10-sample ADC averaging. Faults are injected: swab-like spikes, ×3 noise, stuck values and 0.2 %/min drift. To avoid circularity, the true sensor exponent is swept over b − σ_b, b and b + σ_b while the estimator always uses the consensus b. Detection thresholds are tuned on a validation seed and reported on five separate test seeds."),
  para("Ablations: raw only, datasheet physics only, refitted physics only, prior only, fixed-weight fusion of all three, and trust-adaptive fusion. All modes share the same calibration, so differences come only from the trust mechanism."),
  h2("5.2 Hardware session"),
  para("The node ran on 25 September 2026, powered from 06:22 (about 5.5 h before testing). The session comprised a clean-air reference, a room decay test, a 60-s perfume exposure at 5–10 cm (sensor-local interference) and a 2-min hot-air exposure from a laptop vent (+8.5 °C, −18 %RH)."),
  h2("5.3 Metrics"),
  bullet("Error: RMSE of log-index against ground truth (synthetic)."),
  bullet("Uncertainty: coverage of the 90 % band (PICP) and mean log-width, per segment type."),
  bullet("Diagnosis: precision, recall, F1, latency of the sensor-fault alarm; attribution accuracy."),
  bullet("Significance: paired t-test over seeds; Levene test for variance."),
];

const results = [
  h1("6. Results"),
  h2("6.1 Accuracy"),
  table(["Mode", "RMSE log (5 seeds)", "Note"], [
    ["Fixed-weight fusion", "0.085 ± 0.009", "same calibration as adaptive"],
    ["Trust-adaptive fusion", "0.078 ± 0.008", "paired difference 0.0078 ± 0.0019, t(4) = 9.21, p = 0.0008"],
  ], [2600, 2600, 3826]),
  caption("Table 5. Accuracy over five independent test seeds."),
  para("The improvement is small but consistent (≈ 9 %) and holds when the true sensor exponent is shifted by ±σ (adaptive 0.091–0.094 vs fixed 0.104–0.107 on one seed). The contribution of the method is therefore not primarily accuracy but calibrated uncertainty and fault diagnosis at similar accuracy."),
  image("results/adaptive_trace.png", 600),
  caption("Figure 1. Synthetic test day: raw and physics channels, fused estimate with 90 % band (top), trust per source (middle), temperature and humidity (bottom)."),
  h2("6.1a Learned prior: architecture comparison"),
  table(["Prior (5-member ensemble)", "Skill vs persistence", "Fused RMSE (log, 5 seeds)", "Time / step"], [
    ["MLP (summary vector)", "−12.9 → rejected by admission test", "0.0776 ± 0.0079", "0.8 ms"],
    ["LSTM (sequence)", "0.616", "0.0771 ± 0.0081", "1.8 ms"],
    ["Temporal CNN (sequence)", "0.629", "0.0771 ± 0.0081", "2.4 ms"],
    ["Transformer encoder (sequence)", "0.604", "0.0771 ± 0.0081", "1.9 ms"],
  ], [3000, 2500, 2026, 1500]),
  caption("Table 5a. Same training data and test seeds; adaptive compute off for a fair comparison."),
  para("All three sequence models reduce next-step error by about 60 % relative to persistence, while the MLP on hand-crafted lags is far worse than persistence and is now rejected automatically. The sequence models differ from each other by only about 10⁻⁵ in log units, far below the fused error, so at the fusion level the architectures tie. The LSTM is kept as default; the TCN is marginally best and could replace it."),
  h2("6.2 Uncertainty"),
  table(["Segment", "Samples", "Coverage of 90 % band", "Mean log-width"], [
    ["Clean", "5120", "1.00", "0.35"],
    ["Swab spike", "100", "1.00", "1.43"],
    ["Stuck sensor", "60", "1.00", "1.44"],
    ["×3 noise", "120", "1.00", "0.60"],
    ["Drift 0.2 %/min", "1800", "0.67", "0.34"],
  ], [2600, 1600, 2600, 2226]),
  caption("Table 6. Band coverage by segment (synthetic, conformal q ≈ 1.0)."),
  para("Bands widen about fourfold while the sensor is distrusted and cover local faults. Drift is under-covered (≈ 65 %): trust signals cannot see an unobservable bias. The time-since-anchor term reaches full drift coverage only if its rate matches the true drift, which must be measured from multi-day data."),
  h2("6.3 Fault detection and attribution"),
  table(["Metric (5 seeds)", "Value"], [
    ["Precision", "0.86 ± 0.12"], ["Recall", "0.73 ± 0.03"], ["F1", "0.785 ± 0.052"],
    ["Latency: spike / stuck", "1 s / 4 s"], ["×3 noise detected", "1 of 5 seeds"],
    ["Sensor faults attributed to the sensor", "20 / 20"],
  ], [5026, 4000]),
  caption("Table 7. Sensor-fault alarm, min(τ_raw, τ_phys) < 0.3."),
  para("Cross-sensitivity events were not flagged in simulation; the correction absorbed them. This was not reproduced on hardware (Section 6.5)."),
  h2("6.4 Drift"),
  para("A sweep of the drift noise showed that no setting can track slow drift within a day: on one node, sensor drift and a slow real change both only scale Rs. This is reported as an observability limit; ABC re-anchoring and time-growing bands are the mitigations."),
  h2("6.5 Real hardware"),
  table(["Test", "Observation"], [
    ["Perfume, 60 s at 5–10 cm", "Flagged at onset; both sensor channels trust 0 (attributed to sensor); raw index → 2.4 × 10⁶, fused held at ≈ 871 with the LSTM prior (≈ 917 with MLP); band widened"],
    ["Hot air, +8.5 °C / −18 %RH", "Raw −0.108, physics −0.105 log: datasheet correction removed only ≈ 3 % of the shift; fused moved −0.024 because trust dropped (confounded by residual perfume)"],
    ["Clean-air reference", "Window air was dirtier than the room; cleanest 10 min of the day used instead (ABC assumption)"],
    ["Decay test (λ)", "Inconclusive (occupancy too short); default λ = 1 h⁻¹ kept"],
    ["Undisturbed coverage", "70 % with LSTM prior (62 % with MLP) against a reference-free consensus (target 90 %)"],
    ["Data quality", "DHT11 produced corrupt reads (T ≈ 0.5 °C, RH 0) flagged valid; now rejected"],
  ], [2600, 6426]),
  caption("Table 8. Hardware session results (provisional: ≈ 5.5 h burn-in, load resistor not measured)."),
  image("results/session/session.png", 600),
  caption("Figure 2. Real node: index and band (top), trust (middle), temperature/humidity (bottom); dashed lines mark clean, decay, perfume and hot-air events."),
];

const validation = [
  h1("7. Validation Against References"),
  h2("7.1 Reference dataset: UCI Air Quality"),
  para("The pipeline was run on a real tin-oxide sensor (PT08.S1, same family as the MQ135) logged hourly for a year next to a certified CO analyser (De Vito et al., 2008). Days 0–30 fit the field calibration (the standard static regression used by existing low-cost networks), days 30–60 trained the prior and calibrated uncertainty, and the remaining ten months (7,578 hours) were the test. The filter's process noise was tuned on the validation window only."),
  table(["Method (10-month test)", "RMSE (log)", "R²", "MAE mg/m³", "90 % coverage"], [
    ["Static formula, no T/RH (existing practice)", "0.448", "0.65", "0.62", "–"],
    ["Static formula with T/RH", "0.547", "0.49", "0.77", "–"],
    ["Raw channel through filter", "0.475", "0.69", "0.58", "0.92"],
    ["Fixed-weight fusion", "0.506", "0.60", "0.68", "0.89"],
    ["Trust-adaptive fusion", "0.493", "0.62", "0.65", "0.91"],
  ], [3626, 1300, 1100, 1500, 1500]),
  caption("Table 9. Scored against the certified CO analyser."),
  image("results/uci/uci_validation.png", 600),
  caption("Figure 3. Three test weeks: analyser (dots), static calibration, trust-adaptive estimate with 90 % band, and trust."),
  para("On real ground truth the 90 % band is well calibrated (0.91) and trust-adaptive fusion beats fixed weights. It does not beat the best single channel: the temperature/humidity correction fitted in March extrapolates poorly into summer, repeating the hardware finding. The LSTM prior trained on only 715 hourly samples passed the one-step test (skill 0.04) but failed the fused-level test (validation RMSE 0.680 with vs 0.574 without) and was disabled; before this test existed it had degraded fused R² to 0.28. Injected faults on the real data were detected poorly (F1 0.14): with hourly sampling, natural CO swings resemble the injected faults."),
  h2("7.2 Temperature and humidity: Open-Meteo (Chennai)"),
  para("DHT11 hourly means were compared with Open-Meteo outdoor weather for Chennai. With five usable hours so far: temperature bias −0.4 °C, RMSE 2.5 °C; humidity bias +5.7 %RH, RMSE 7.3 %RH. The node is indoors, so an offset is expected; the comparison becomes meaningful with multi-day logging. The dashboard also shows the live node-minus-outdoor difference."),
  h2("7.3 Trust states and adaptive compute (synthetic, 3 seeds)"),
  table(["True condition", "TRUSTED", "UNCERTAIN", "OOD", "SENSOR_FAULT"], [
    ["Clean", "86 %", "3 %", "10 %", "1 %"], ["Swab spike", "2 %", "–", "–", "98 %"],
    ["Stuck sensor", "7 %", "–", "–", "93 %"], ["×3 noise", "30 %", "29 %", "–", "41 %"],
    ["Drift", "100 %", "–", "–", "–"],
  ], [2226, 1700, 1700, 1700, 1700]),
  caption("Table 10. Share of time in each state per true condition."),
  para("Spike and stuck faults are recognised as SENSOR_FAULT; 10 % of clean time is flagged OOD, almost all inside the scripted T/RH swing (42 % flagged there, a correct detection; 3.4 % elsewhere); drift remains TRUSTED because it is unobservable within the two-hour test. Coverage of the 90 % band was 98–100 % in every state. Adaptive compute evaluated 67 % fewer network members and halved time per step (3.1 → 1.6 ms) with identical error (0.0749) and coverage (0.984)."),
  h2("7.4 False out-of-distribution reduction"),
  para("The first OOD detector flagged 15 % of clean synthetic time. Diagnosis showed the six lag features are almost collinear, so the inverse sample covariance amplified harmless differences in the shape of a CO2 rise. Three changes fixed it: the distance now uses a reduced summary (lag level, lag slope, T, RH and their changes) with a Ledoit-Wolf shrinkage covariance; the threshold is the larger of the in-sample and held-out (time-split) 99.5th percentile of D²; and P(OOD) is smoothed with a 20-s moving average so single samples cannot switch state. Each OOD state now reports its cause (distance, envelope, disagreement or novelty) with a matching action."),
  table(["Data", "Before", "After"], [
    ["Synthetic clean, outside T/RH swing", "5.0 %", "3.4 %"],
    ["Synthetic clean, all", "15 %", "10 %"],
    ["Real node, quiet half-hours", "–", "0–2 %"],
  ], [4426, 2300, 2300]),
  caption("Table 11. Share of time flagged OOD."),
  para("On the real log the remaining OOD time falls on genuine events: sensor warm-up after power-on (78 % of the first half-hour), the perfume and hot-air tests (26 %), and warm-up after a 7.5-h power outage (51–96 %, heater still stabilising)."),
];

const discussion = [
  h1("8. Discussion"),
  para("Development was driven by diagnosing failures rather than tuning to targets. An initial conformal factor of about 66 revealed that the filter was biased (drift state absorbing real room changes) and over-confident (sensor noise under-estimated, the prior double-counting fused history). Fixing the filter brought the factor to about 1.0. A variable-name collision in the simulator and a calibration bug in the drift-growth term were found the same way. The detection gain was traced to the retuned noise model and a variance test rather than to a change of alarm rule."),
  para("The hardware session both confirmed and contradicted the simulation. Detection of sensor-local interference transferred directly. Temperature/humidity correction did not: the simulator was generated from the same datasheet curve used by the estimator, which made correction look better than it is. This is an important lesson about synthetic evaluation."),
  h2("8.1 Limitations"),
  bullet("Single node: no spatial physics; well-mixed assumption makes λ an effective parameter."),
  bullet("Drift is unobservable within a day; ABC needs the room aired at least daily."),
  bullet("Noise-type faults are poorly detected."),
  bullet("Datasheet temperature/humidity correction is weak on the real node; a refit needs multi-day data with wider T/RH range."),
  bullet("Short burn-in (≈ 5.5 h), unmeasured load resistor, no verified outdoor reference."),
  bullet("No reference instrument: the output is a relative CO2-equivalent index, not certified ppm; the MQ135 is not gas-selective."),
  bullet("The sequence priors (LSTM, TCN, Transformer) improve one-step prediction by about 60 % over persistence but not the fused accuracy; with little training data the admission test disables them."),
  bullet("On the UCI reference data, adaptive fusion beats fixed weights but not the best single channel; fault detection on real data is weak (F1 0.14); false OOD on clean data 3.4 % (synthetic) and 0–2 % in quiet real half-hours."),
  bullet("DHT11 read failures rose to 26 % in one hour on the real node (likely wiring or heat); the pipeline rejects implausible reads but the sensor needs checking."),
];

const conclusion = [
  h1("9. Conclusion and Future Work"),
  para("PHYS-TRUST Lite shows that a single cheap sensor node can estimate, online and interpretably, when to trust its own reading, its physics model and a learned deep-ensemble prior. At similar accuracy to fixed-weight fusion, and with an LSTM deep-ensemble prior that improves one-step prediction, it adds calibrated uncertainty and reliable detection and attribution of sensor-level faults, confirmed on real hardware. It is equally explicit about what one uncalibrated node cannot do."),
  h2("9.1 Future work"),
  num("Add a physics-residual loss on the mass balance to the LSTM prior (physics-regularised prior), and compare against a temporal CNN or transformer.", "n2"),
  num("Multi-day logging to benchmark ABC and measure the drift rate k_d; refit the temperature/humidity correction.", "n2"),
  num("A short co-location with an NDIR CO2 meter to report absolute error.", "n2"),
  num("Several nodes, enabling consensus-based trust and spatial physics.", "n2"),
];

const refs = [
  h1("References"),
  ...[
    "Winsen Electronics, MQ135 Semiconductor Sensor for Air Quality — Technical Manual v1.4, 2015.",
    "ASTM E741-11(2017), Standard Test Method for Determining Air Change in a Single Zone by Means of a Tracer Gas Dilution.",
    "Abdullah, A. N. et al., Correction model for metal oxide sensor drift caused by ambient temperature and humidity, Sensors 22(9):3301, 2022.",
    "Kalra, V. et al., Arduino based air quality monitoring with MQ135 and DHT11, IJARCSSE 6(4), 2016.",
    "Lakshminarayanan, B., Pritzel, A., Blundell, C., Simple and scalable predictive uncertainty estimation using deep ensembles, NeurIPS 2017.",
    "Hochreiter, S., Schmidhuber, J., Long short-term memory, Neural Computation 9(8):1735–1780, 1997.",
    "De Vito, S. et al., On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario, Sensors and Actuators B 129(2):750–757, 2008 (UCI Air Quality dataset).",
    "Open-Meteo, free weather API (forecast and historical), open-meteo.com, accessed September 2026.",
    "Angelopoulos, A. N., Bates, S., A gentle introduction to conformal prediction and distribution-free uncertainty quantification, 2021.",
    "Mahajan, S., Helbing, D., Trust-based calibration of low-cost PM2.5 sensors, npj Climate and Atmospheric Science 8, 257, 2025.",
    "Christakis, I. et al., Kalman filtering for low-cost gas sensor data, Electronics 13(1):25, 2024.",
    "Hettige, K. H. et al., AirPhyNet: harnessing physics-guided neural networks for air quality prediction, arXiv:2402.03784, 2024.",
    "NOAA Global Monitoring Laboratory, Mauna Loa CO2 weekly mean, accessed September 2026.",
  ].map((s, i) => new Paragraph({ spacing: { after: 80 }, children: [t(`[${i + 1}] ${s}`, { size: 20 })] })),
  para([t("Note: verify every reference against the original source before submission.", { italics: true, size: 18 })]),
];

const appendix = [
  h1("Appendix A. Running the System"),
  para([t("Logger (runs independently, keeps the COM port):", { bold: true })]),
  para([t("pythonw tools\\logger.py --port COM8 --out data\\burnin.csv", { font: "Consolas", size: 20 })]),
  para([t("Live dashboard:", { bold: true })]),
  para([t("python -m streamlit run tools/dashboard.py", { font: "Consolas", size: 20 })]),
  para("In the sidebar choose Live sensor (follows the logger in real time), CSV replay, Simulator (no hardware needed; includes injected faults) or Serial, then switch Run on. The Model & method and Results tabs summarise the approach and results."),
  para([t("Synthetic benchmark and hardware analysis:", { bold: true })]),
  para([t("python -m phystrust.evaluate --seeds 5", { font: "Consolas", size: 20 })]),
  para([t("python tools/compare_priors.py", { font: "Consolas", size: 20 })]),
  para([t("Validation: python tools/validate_uci.py · python tools/validate_weather.py · python tools/evaluate_states.py", { font: "Consolas", size: 18 })]),
  para([t("python tools/session_report.py --clean HH:MM HH:MM --swab HH:MM --hot HH:MM HH:MM", { font: "Consolas", size: 20 })]),
];

const doc = new Document({
  creator: "PHYS-TRUST Lite",
  title: "PHYS-TRUST Lite Project Report",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: "1F3864" }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: "2E75B6" }, paragraph: { outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "n", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "n2", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
  ] },
  features: { updateFields: true },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 },
      margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [t("PHYS-TRUST Lite — Project Report", { size: 18, color: "808080" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [t("Page ", { size: 18 }), new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children: [...title, ...abstract, ...toc, ...intro, ...related, ...system, ...method, ...phase2method, ...setup,
      ...results, ...validation, ...discussion, ...conclusion, ...refs, ...appendix],
  }],
});

Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("wrote", OUT, b.length, "bytes"); });
