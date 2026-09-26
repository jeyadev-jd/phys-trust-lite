"""Trust state machine + action policy (PHYS-TRUST 2.0, single node).

Turns the numeric trust/uncertainty signals into explicit operating states and actions.

States (priority order):
  SENSOR_FAULT        both sensor channels distrusted, or ADC fault
  PHYSICS_MISMATCH    sensor plausible but the physics channel persistently disagrees with the state
  OOD                 inputs unlike the training distribution (multi-signal P(OOD))
  CALIBRATION_NEEDED  calibration confidence low (time since anchor, drift, reference disagreement)
  UNCERTAIN           any source trust reduced or band wide
  TRUSTED             otherwise

P(OOD) combines: feature-space distance (Mahalanobis), ensemble disagreement, T/RH outside the fitted
envelope, and temporal novelty of the physics signal. Each signal s_i (1 = at threshold) maps to
p_i = sigmoid(4 (s_i - 1)); P(OOD) = 1 - prod(1 - p_i).

Compute tier drives adaptive compute: tier 0 = cheap (1 prior member), tier >= 1 = full 5-member ensemble.
"""
from collections import deque

import numpy as np

STATES = ("TRUSTED", "UNCERTAIN", "CALIBRATION_NEEDED", "OOD", "PHYSICS_MISMATCH", "SENSOR_FAULT")
ACTIONS = {
    "TRUSTED": ("monitor", 0, "Normal monitoring; cheap inference (1 ensemble member)."),
    "UNCERTAIN": ("full_ensemble", 1, "Run full 5-member ensemble; keep widened band."),
    "CALIBRATION_NEEDED": ("request_calibration", 1, "Request clean-air re-zero / ABC re-anchor."),
    "OOD": ("cross_check", 2, "Full ensemble + physics residual; cross-check external reference."),
    "PHYSICS_MISMATCH": ("diagnose_physics", 2, "Run physics-residual analysis; check T/RH input and model."),
    "SENSOR_FAULT": ("reject_raw", 2, "Reject raw reading, hold estimate; alert human if persistent."),
}


def _sig(x):
    return 1.0 / (1.0 + np.exp(-4.0 * (x - 1.0)))


class TrustStateMachine:
    def __init__(self, cfg, dwell_s=5.0, novelty_win=120, cal_tau_s=7 * 86400, alert_after_s=60.0):
        self.cfg = cfg
        self.dwell_s, self.cal_tau_s, self.alert_after_s = dwell_s, cal_tau_s, alert_after_s
        self.state, self.in_state_s, self.pending, self.pending_s = "TRUSTED", 0.0, None, 0.0
        self.z_hist = deque(maxlen=novelty_win)
        self.mismatch_s = 0.0
        self.p_ood_s = 0.0          # EWMA of P(OOD): state needs persistent evidence, not one sample
        self.ood_alpha = 1.0 - np.exp(-1.0 / 20.0)  # ~20 s time constant

    def p_ood(self, f):
        cfg = self.cfg
        sig = dict(
            distance=f.get("ood_d2", 0.0) / cfg.ood_thr,
            disagreement=f.get("spread", 0.0) / cfg.spread_thr,
            envelope=1.0 + f.get("env_pen", 0.0) if f.get("env_pen", 0.0) > 0 else 0.0,
            novelty=f.get("novelty", 0.0) / 6.0,
        )
        p = {k: float(_sig(v)) if v > 0 else 0.0 for k, v in sig.items()}
        return 1.0 - float(np.prod([1 - v for v in p.values()])), p

    def calib_confidence(self, t_since_anchor, drift, ref_disagree=0.0):
        """1 = freshly calibrated; decays with time, apparent drift and reference disagreement."""
        return float(np.exp(-t_since_anchor / self.cal_tau_s) * np.exp(-abs(np.log(drift)) / 0.1)
                     * np.exp(-max(ref_disagree, 0.0)))

    def update(self, row, f, dt=1.0):
        """row: pipeline output; f: extra features (ood_d2, spread, env_pen, nis_phys_ewma, z_phys, rate_ok)."""
        z = f.get("z_phys")
        novelty = 0.0
        if z is not None and np.isfinite(z):
            if len(self.z_hist) >= 30:
                h = np.array(self.z_hist)
                novelty = abs(z - np.median(h)) / (1.4826 * np.median(np.abs(h - np.median(h))) + 1e-3)
            self.z_hist.append(z)
        f = dict(f, novelty=novelty)
        p_ood, parts = self.p_ood(f)
        self.p_ood_s += self.ood_alpha * dt * (p_ood - self.p_ood_s)
        cause = max(parts, key=parts.get) if p_ood > 0 else "none"
        # drift measured from the last anchor: a deliberate re-zero shift is not evidence of bad calibration
        cal = self.calib_confidence(row["t_since_anchor"], row.get("drift_anchor", row["drift"]), f.get("ref_disagree", 0.0))

        t_raw, t_phys = row["trust_raw"], row["trust_phys"]
        mismatch = f.get("rate_ok", True) and f.get("nis_phys_ewma", 1.0) > 3 * 3.841 and t_raw >= 0.3
        self.mismatch_s = self.mismatch_s + dt if mismatch else 0.0
        half = (np.log(row["hi"]) - np.log(row["lo"])) / 2

        # both channels come from one MQ135: a sensor fault drives BOTH down. Raw alone low = uncorrected T/RH
        # cross-sensitivity (humidity away from calibration), which is handled below as UNCERTAIN.
        if max(t_raw, t_phys) < 0.3 or row["adc_fault"]:
            cand = "SENSOR_FAULT"
        elif self.mismatch_s >= 30 or (t_phys < 0.3 <= t_raw):
            cand = "PHYSICS_MISMATCH"
        elif self.p_ood_s > 0.5:
            cand = "OOD"
        elif cal < 0.3:
            cand = "CALIBRATION_NEEDED"
        elif min(t_raw, t_phys, row["trust_prior"] if np.isfinite(row["trust_prior"]) else 1.0) < 0.7 or half > 0.35:
            cand = "UNCERTAIN"
        else:
            cand = "TRUSTED"

        # hysteresis: escalate to SENSOR_FAULT immediately, otherwise require dwell before switching
        if cand == self.state:
            self.pending, self.pending_s = None, 0.0
        elif cand == "SENSOR_FAULT":
            self.state, self.in_state_s, self.pending = cand, 0.0, None
        else:
            self.pending_s = self.pending_s + dt if cand == self.pending else dt
            self.pending = cand
            if self.pending_s >= self.dwell_s:
                self.state, self.in_state_s, self.pending, self.pending_s = cand, 0.0, None, 0.0
        self.in_state_s += dt

        action, tier, why = ACTIONS[self.state]
        if self.state == "OOD":
            why = {"envelope": "T/RH outside the calibrated range: corrected channel less reliable; cross-check reference.",
                   "distance": "Signal pattern unlike training data: full ensemble + cross-check reference.",
                   "disagreement": "Ensemble members disagree: prediction uncertain; widen band, cross-check.",
                   "novelty": "Sudden change in the physics signal: check for local source or sensor issue."}.get(cause, why)
        alert = (self.state == "SENSOR_FAULT" and self.in_state_s >= self.alert_after_s) or \
                self.state == "CALIBRATION_NEEDED"
        return dict(state=self.state, action=action, compute_tier=tier, action_note=why, alert_human=alert,
                    p_ood=p_ood, p_ood_smooth=self.p_ood_s, ood_cause=cause, calib_confidence=cal, novelty=novelty,
                    **{f"p_ood_{k}": v for k, v in parts.items()})
