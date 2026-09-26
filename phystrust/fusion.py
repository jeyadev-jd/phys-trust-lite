"""L4: EKF, state x = [C, ld]  (C = room index, ld = log Rs drift multiplier).

Measurements (log-index units):
  raw / phys : z = log C + b * ld        H = [1/C, b]
  prior      : z = log C                 H = [1/C, 0]
Trust enters as R_s = R0_s / max(trust_s, eps).
"""
import numpy as np

SENSOR = ("raw", "phys")


class TrustEKF:
    def __init__(self, cfg, c0):
        self.cfg = cfg
        self.x = np.array([max(c0, 1.0), 0.0])
        self.P = np.diag([50.0 ** 2, 0.005 ** 2])  # freshly calibrated: d = 1 known tightly

    def predict(self, dt):
        cfg = self.cfg
        e = np.exp(-cfg.lam * dt)
        self.x[0] = cfg.c_bg + (self.x[0] - cfg.c_bg) * e
        F = np.diag([e, 1.0])
        Q = np.diag([cfg.q_c ** 2 * dt, cfg.q_ld ** 2 * dt])
        self.P = F @ self.P @ F.T + Q

    def _hH(self, ch):
        c, ld = self.x
        b = self.cfg.b
        if ch in SENSOR:
            return np.log(c) + b * ld, np.array([1.0 / c, b])
        return np.log(c), np.array([1.0 / c, 0.0])

    def innovations(self, z, r_nominal):
        """Per-channel innovation + NIS against nominal (untrusted-scaled) R."""
        out = {}
        for ch, zv in z.items():
            h, H = self._hH(ch)
            nu = zv - h
            s = H @ self.P @ H + r_nominal[ch]
            out[ch] = (nu, nu * nu / s)
        return out

    def update(self, z, R):
        if not z:
            return
        chs = list(z)
        hs, Hs = zip(*(self._hH(c) for c in chs))
        H = np.vstack(Hs)
        nu = np.array([z[c] for c in chs]) - np.array(hs)
        Rm = np.diag([R[c] for c in chs])
        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        I_KH = np.eye(2) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T  # Joseph form
        self.x[0] = max(self.x[0], 1.0)

    def project_background(self, c_obs_median=None, margin=0.97):
        """Physics constraint: room index cannot sit below outdoor background.
        C is clamped every step. Drift absorbs the deficit only when the 60-s median
        observation is below background (sustained), so single noisy samples cannot
        ratchet the drift state one way."""
        c_bg, b = self.cfg.c_bg, self.cfg.b
        floor = margin * c_bg
        if c_obs_median is not None and c_obs_median < floor:
            # rate-limited: window holds stale samples, so move 1/60 of deficit per step
            self.x[1] -= (np.log(floor) - np.log(c_obs_median)) / b / 60.0
        if self.x[0] < floor:
            self.x[0] = floor

    def reanchor(self, var=None):
        """Open the level state so the filter can re-acquire after a persistent step. Relative (50 % of the
        level): a fixed 100-unit sd could not reach a 15-20 % step once the index sits in the thousands."""
        if var is None:
            var = (0.5 * max(float(self.x[0]), 1.0)) ** 2
        self.P[0, 0] += var

    @property
    def c(self):
        return float(self.x[0])

    @property
    def ld(self):
        return float(self.x[1])

    @property
    def sigma_log(self):
        return float(np.sqrt(max(self.P[0, 0], 1e-12)) / self.x[0])
