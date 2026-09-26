"""Full per-sample pipeline L0..L5 + ablation modes."""
from collections import deque

import joblib
import numpy as np
import pandas as pd

from .config import Config
from .coefficients import coef_var_log
from .conformal import Conformal
from .fusion import TrustEKF
from .process_physics import max_rate
from .sensor_physics import RefitCF, adc_to_rs, cf_datasheet, log_index
from .states import TrustStateMachine
from .trust import CHANNELS, TrustEstimator

# mode -> (enabled channels, adaptive trust?, CF source)
MODES = {
    "raw":        ({"raw"}, False, "refit"),
    "ds_phys":    ({"phys"}, False, "datasheet"),
    "refit_phys": ({"phys"}, False, "refit"),
    "prior":      ({"prior"}, False, "refit"),
    "fixed":      ({"raw", "phys", "prior"}, False, "refit"),
    "adaptive":   ({"raw", "phys", "prior"}, True, "refit"),
}


class Bundle:
    """Calibration artefacts: R0, refit CF, prior ensemble, conformal q, config."""

    def __init__(self, cfg=None, r0=None, refit=None, prior=None, conformal=None, sigma_sys=0.0,
                 version=None):
        self.cfg = cfg or Config()
        self.version = version or pd.Timestamp.now().strftime("cal-%Y%m%d-%H%M")  # provenance
        self.sigma_sys = sigma_sys  # time-correlated model error (log), does not average down
        self.r0 = r0 if r0 is not None else self.cfg.r0_kohm
        self.refit = refit
        self.prior = prior
        self.conformal = conformal or Conformal(self.cfg.alpha)

    def save(self, path):
        joblib.dump(dict(cfg=self.cfg.to_dict(), r0=self.r0,
                         refit=self.refit.to_dict() if self.refit else None,
                         prior=self.prior, q=self.conformal.q, sigma_sys=self.sigma_sys,
                         version=self.version), path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        cfg = Config.from_dict(d["cfg"])
        return cls(cfg, d["r0"], RefitCF.from_dict(d["refit"]) if d["refit"] else None,
                   d["prior"], Conformal(cfg.alpha, d["q"]), d.get("sigma_sys", 0.0),
                   d.get("version", "unversioned"))


class PhysTrust:
    def __init__(self, bundle=None, mode="adaptive"):
        self.b = bundle or Bundle()
        self.cfg = self.b.cfg
        self.channels, self.adaptive, cf_src = MODES[mode]
        self.use_refit = cf_src == "refit" and self.b.refit is not None
        if self.b.prior is None:
            self.channels = self.channels - {"prior"}
        self.mode = mode
        self.ekf = None
        self.trust = TrustEstimator(self.cfg)
        n_hist = max(self.cfg.lag_win, self.b.prior.need if self.b.prior else 0) + 1
        self.h_t, self.h_rh = deque(maxlen=n_hist), deque(maxlen=n_hist)
        self.h_logc = deque(maxlen=n_hist)
        self.h_cobs = deque(maxlen=self.cfg.rate_win + 3)
        self.h_adc = deque(maxlen=5)
        self.h_cobs_long = deque(maxlen=60)
        self.h_z = deque(maxlen=31)   # physics-channel log index, for high-frequency noise test
        self.dz_base = None           # baseline variance of first differences
        self.segment = 0              # increments on logger restart / time gap
        self.abc_min = deque(maxlen=max(1, self.cfg.abc_window_s // 60))  # per-minute medians
        self.abc_acc, self.abc_t = [], 0.0
        self.abc_events = 0
        self.t_since_anchor = 0.0     # s since last ABC / manual re-zero / calibration
        self.anchor_ld = 0.0          # drift state right after the last anchor (re-zero moves it on purpose)
        self.last_ms, self.last_t, self.last_rh = None, 25.0, 50.0
        self.prior_next = None  # (z, var, d2, spread) predicted for this step
        self.r_nom = {"raw": self.cfg.r_raw, "phys": self.cfg.r_phys, "prior": self.cfg.r_prior}
        self.sm = TrustStateMachine(self.cfg)
        self.tier, self.last_sd, self.prior_evals = 2, 0.0, 0   # adaptive compute bookkeeping

    # ---------------- L0
    def _preprocess(self, s):
        cfg = self.cfg
        adc = float(s["adc_mean"])
        t_ok = np.isfinite(s["T"]) and np.isfinite(s["RH"])
        # DHT11 occasionally returns corrupt reads (T~0.5 C, RH 0) flagged as valid: reject
        # physically implausible values and hold the last good reading
        t_ok = t_ok and 1.0 <= s["RH"] <= 100.0 and -10.0 <= s["T"] <= 60.0 and s["T"] >= 1.0
        if int(s.get("dht_ok", 1)) and t_ok:
            self.last_t, self.last_rh = float(s["T"]), float(s["RH"])
        self.h_adc.append(adc)
        sat = adc <= 2 or adc >= cfg.adc_max - 2
        stuck = (len(self.h_adc) == self.h_adc.maxlen and len(set(self.h_adc)) == 1
                 and float(s.get("adc_std", 1.0)) == 0.0)
        return adc, sat or stuck

    def _dt(self, ms):
        gap = self.last_ms is not None and (ms < self.last_ms or ms - self.last_ms > 5000)
        dt = 1.0 if self.last_ms is None or gap else (ms - self.last_ms) / 1000.0
        self.last_ms = ms
        if gap:
            self._segment_break()
        return float(np.clip(dt, 0.2, 5.0))

    def _segment_break(self):
        """Logger restart (Uno resets, millis restarts) or time gap: do not treat the jump
        as one continuous step. Clear short histories, widen state uncertainty."""
        for h in (self.h_cobs, self.h_cobs_long, self.h_adc, self.h_z, self.h_logc):
            h.clear()
        self.prior_next = None
        if self.ekf is not None:
            self.ekf.reanchor()
        self.segment += 1

    def _shift_baseline(self, c_from):
        """Re-anchor: make index c_from map to background by moving drift state."""
        cfg = self.cfg
        dl = (np.log(cfg.c_bg) - np.log(c_from)) / cfg.b
        self.ekf.x[1] -= dl
        f = cfg.c_bg / c_from
        for q in (self.abc_min, self.h_cobs, self.h_cobs_long):
            for i in range(len(q)):
                q[i] *= f
        for i in range(len(self.h_logc)):  # prior input history in log units: shift too, else a fake jump -> OOD
            self.h_logc[i] += np.log(f)
        self.ekf.x[0] *= f
        self.t_since_anchor = 0.0
        self.anchor_ld = float(self.ekf.x[1])

    def _abc(self, c_obs, dt, adc_fault):
        """Automatic baseline calibration (as in NDIR CO2 sensors): lowest per-minute median
        over the window is assumed to be outdoor background. Needs the room aired once per window."""
        if not self.cfg.abc_enabled or adc_fault:
            return
        self.abc_acc.append(c_obs)
        self.abc_t += dt
        if self.abc_t < 60.0:
            return
        self.abc_min.append(float(np.median(self.abc_acc)))
        self.abc_acc, self.abc_t = [], 0.0
        if len(self.abc_min) == self.abc_min.maxlen:
            m = min(self.abc_min)
            if abs(np.log(m / self.cfg.c_bg)) > 0.02:
                self._shift_baseline(m)
                self.abc_events += 1

    def rezero(self):
        """Manual re-zero (window open, button press): current 60-s median := background."""
        if len(self.h_cobs_long):
            self._shift_baseline(float(np.median(self.h_cobs_long)))

    def step(self, s):
        cfg = self.cfg
        dt = self._dt(float(s["millis"]))
        adc, adc_fault = self._preprocess(s)
        t, rh = self.last_t, self.last_rh
        self.h_t.append(t)
        self.h_rh.append(rh)

        # ---------------- L1a sensor physics
        rs = float(adc_to_rs(adc, cfg.rl_kohm, cfg.adc_max))
        cf_ds = float(cf_datasheet(t, rh))
        cf_rf = self.b.refit(t, rh) if self.b.refit is not None else cf_ds
        cf = cf_rf if self.use_refit else cf_ds
        z_raw = float(log_index(rs, self.b.r0, cfg.a, cfg.b, 1.0))
        z_phys = float(log_index(rs, self.b.r0, cfg.a, cfg.b, cf))

        if self.ekf is None:
            self.ekf = TrustEKF(cfg, float(np.exp(z_phys)))

        # high-frequency noise test: variance of first differences vs adaptive baseline.
        # Removes slow systematic error, so a variance change is visible even when R is large.
        noise_ratio = 1.0
        if not adc_fault:
            self.h_z.append(z_phys)
        if len(self.h_z) == self.h_z.maxlen:
            v = float(np.var(np.diff(self.h_z)))
            if self.dz_base is None:
                self.dz_base = v
            noise_ratio = v / max(self.dz_base, 1e-10)
            if noise_ratio < 2.0:
                self.dz_base += (v - self.dz_base) / 600.0
        # ---------------- L1b process physics (prediction)
        self.ekf.predict(dt)

        z = {}
        if not adc_fault:
            if "raw" in self.channels:
                z["raw"] = z_raw
            if "phys" in self.channels:
                z["phys"] = z_phys
        prior_ok = "prior" in self.channels and self.prior_next is not None
        prior_used = self.prior_next if prior_ok else None
        r_nom = dict(self.r_nom)
        if prior_ok:
            z["prior"] = self.prior_next[0]
            r_nom["prior"] = max(cfg.r_prior, self.prior_next[1])
        innov = self.ekf.innovations(z, r_nom)

        # ---------------- L3 trust features
        c_obs = float(np.exp(z_phys - cfg.b * self.ekf.ld))  # drift-corrected
        self.h_cobs.append(c_obs)
        self.h_cobs_long.append(c_obs)
        rate_ratio = 0.0
        if len(self.h_cobs) >= cfg.rate_win + 3:
            h = np.array(self.h_cobs)
            rate = abs(np.median(h[-3:]) - np.median(h[:3])) / (cfg.rate_win * dt)
            noise_rate = self.ekf.c * np.sqrt(cfg.r_phys) * np.sqrt(2 / 3) / (cfg.rate_win * dt)
            rate_ratio = rate / max_rate(self.ekf.c, cfg, noise_rate)

        lag_pen = 0.0
        if len(self.h_t) > cfg.lag_win:
            w = cfg.lag_win * dt
            dTdt = abs(self.h_t[-1] - self.h_t[-1 - cfg.lag_win]) / w
            dRHdt = abs(self.h_rh[-1] - self.h_rh[-1 - cfg.lag_win]) / w
            lag_pen = max(0.0, dTdt / cfg.dtdt_thr - 1) + max(0.0, dRHdt / cfg.drhdt_thr - 1)
        env_pen = self.b.refit.envelope_penalty(t, rh) if self.b.refit is not None else 0.0
        cf_dis_pen = max(0.0, abs(np.log(cf_ds) - np.log(cf_rf)) / cfg.cf_dis_thr - 1)
        ood_pen = spread_pen = 0.0
        if prior_ok:
            ood_pen = max(0.0, self.prior_next[2] / getattr(self.b.prior, "ood_thr", cfg.ood_thr) - 1)
            spread_pen = max(0.0, self.prior_next[3] / cfg.spread_thr - 1)

        feats = dict(rate_ratio=rate_ratio, adc_fault=adc_fault, env_pen=env_pen,
                     noise_pen=max(0.0, noise_ratio / cfg.noise_ratio_thr - 1),
                     lag_pen=lag_pen, cf_dis_pen=cf_dis_pen, ood_pen=ood_pen,
                     spread_pen=spread_pen, prior_ok=prior_ok)
        trust, reanchor = self.trust.update({c: v[1] for c, v in innov.items()}, feats, dt)
        if not self.adaptive:
            trust = {c: 1.0 for c in CHANNELS}
            reanchor = False

        # ---------------- L4 fusion
        R = {c: r_nom[c] / max(trust[c], cfg.trust_eps) for c in z}
        if reanchor:
            self.ekf.reanchor()
        self.ekf.update(z, R)
        med = (float(np.median(self.h_cobs_long))
               if len(self.h_cobs_long) == self.h_cobs_long.maxlen and not adc_fault else None)
        self.ekf.project_background(med)

        self.t_since_anchor += dt
        self._abc(c_obs, dt, adc_fault)

        # ---------------- L5 uncertainty
        logc = float(np.log(self.ekf.c))
        # b error is shared by raw and physics channels (same b): add once to fused
        # variance, not per channel (filter would average it down as if independent)
        cv = coef_var_log(logc, np.log(cfg.c_ref), cfg.b, cfg.sigma_b)
        # systematic error grows when both sensor channels are distrusted (fused estimate then
        # rests on process model + prior); trust = 1 on clean data, so clean bands unchanged
        t_best = max(trust["raw"], trust["phys"]) if self.adaptive else 1.0
        s_sys = self.b.sigma_sys / np.sqrt(max(t_best, cfg.sys_trust_floor))
        # unobservable drift is a random walk: band grows linearly in variance since last anchor
        k_d = cfg.drift_var_rate if cfg.drift_var_rate >= 0 else (cfg.b * cfg.q_ld) ** 2
        sig0 = float(np.sqrt(self.ekf.sigma_log ** 2 + cv + s_sys ** 2))  # calibrated part
        sig = float(np.sqrt(sig0 ** 2 + k_d * self.t_since_anchor))
        lo, hi = self.b.conformal.interval(logc, sig)

        # ---------------- L2 prior prediction for next step (fed on fused history)
        self.h_logc.append(logc)
        self.prior_next = None
        pr = self.b.prior
        if pr is not None and "prior" in self.channels and len(self.h_logc) >= pr.need:
            n = pr.need
            x = pr.features(list(self.h_logc)[-n:], list(self.h_t)[-n:], list(self.h_rh)[-n:])
            cheap = cfg.adaptive_compute and self.adaptive and self.tier == 0
            k = 1 if cheap else None
            mu, sd, d2 = pr.predict(x, k)
            self.prior_evals += 1 if cheap else len(pr.members)
            if cheap:
                sd = self.last_sd          # 1 member has no spread: carry last full-ensemble spread
            else:
                self.last_sd = sd
            # prior is built on fused history: carry its uncertainty (not independent info)
            self.prior_next = (logc + mu, sd ** 2 + self.ekf.sigma_log ** 2, d2, sd)

        row = dict(
            millis=s["millis"], T=t, RH=rh, adc=adc, rs=rs, cf_ds=cf_ds, cf_refit=cf_rf,
            idx_raw=float(np.exp(z_raw)), idx_phys=float(np.exp(z_phys)),
            idx_prior=float(np.exp(z["prior"])) if "prior" in z else np.nan,
            fused=self.ekf.c, lo=lo, hi=hi, sigma_log=sig, sigma_log_nodrift=sig0, drift=float(np.exp(self.ekf.ld)),
            drift_anchor=float(np.exp(self.ekf.ld - self.anchor_ld)),
            trust_raw=trust["raw"], trust_phys=trust["phys"], trust_prior=trust["prior"],
            rate_ratio=rate_ratio, adc_fault=adc_fault, reanchor=reanchor,
            noise_ratio=noise_ratio, segment=self.segment, abc_events=self.abc_events,
            t_since_anchor=self.t_since_anchor,
            # variance terms of sigma_total before conformal q (log units^2), for display only
            var_ekf=self.ekf.sigma_log ** 2, var_b=cv, var_sys=s_sys ** 2,
            var_drift=k_d * self.t_since_anchor, q_conf=self.b.conformal.q,
            node_id=cfg.node_id, calib_version=self.b.version, prior_evals=self.prior_evals,
            nis_raw=innov.get("raw", (0, np.nan))[1], nis_phys=innov.get("phys", (0, np.nan))[1],
        )
        used = prior_used if prior_used is not None else (np.nan, 0.0, 0.0, 0.0)
        thr = getattr(self.b.prior, "ood_thr", cfg.ood_thr)          # empirical if calibrated
        sm = self.sm.update(row, dict(ood_d2=used[2] * cfg.ood_thr / thr, spread=used[3], env_pen=env_pen, z_phys=z_phys,
                                      nis_phys_ewma=self.trust.nis["phys"], rate_ok=rate_ratio <= 1.0), dt)
        self.tier = sm["compute_tier"]
        row.update(sm, quality_score=min(trust["raw"], trust["phys"]))
        return row


def run_frame(df, bundle=None, mode="adaptive"):
    pt = PhysTrust(bundle, mode)
    rows = [pt.step(r) for r in df.to_dict("records")]
    return pd.DataFrame(rows, index=df.index)
