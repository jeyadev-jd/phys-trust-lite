"""L3: interpretable trust estimator. Penalties -> target trust = exp(-sum penalties),
asymmetric smoothing (fast drop, slow recovery)."""
import numpy as np

CHI2_1_95 = 3.841
CHANNELS = ("raw", "phys", "prior")


class TrustEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.trust = {c: 1.0 for c in CHANNELS}
        self.nis = {c: 1.0 for c in CHANNELS}
        self.pen = {c: {} for c in CHANNELS}
        self.plaus_ok_s = 0.0

    def update(self, nis, f, dt):
        """nis: {channel: NIS}; f: feature dict from pipeline. Returns (trust, reanchor)."""
        cfg = self.cfg
        for c, v in nis.items():
            self.nis[c] = (1 - cfg.nis_alpha) * self.nis[c] + cfg.nis_alpha * min(v, 100.0)

        rate_pen = max(0.0, f["rate_ratio"] - 1.0)
        self.plaus_ok_s = self.plaus_ok_s + dt if rate_pen == 0 else 0.0

        # Sensor plausible for a while but state disagrees: the model, not the sensor, is wrong.
        reanchor = False
        if self.plaus_ok_s >= cfg.recover_s and self.nis["phys"] > 3 * CHI2_1_95:
            self.nis["phys"] = self.nis["raw"] = 1.0
            self.plaus_ok_s = 0.0
            reanchor = True

        p_nis = {c: max(0.0, self.nis[c] / CHI2_1_95 - 1.0) for c in CHANNELS}
        hard = 5.0 if f["adc_fault"] else 0.0

        pen = {
            "raw": {"nis": p_nis["raw"], "rate": cfg.k_rate * rate_pen, "adc": hard,
                    "noise": f.get("noise_pen", 0.0)},
            "phys": {"nis": p_nis["phys"], "rate": cfg.k_rate * rate_pen, "adc": hard,
                     "noise": f.get("noise_pen", 0.0),
                     "env": f["env_pen"], "lag": f["lag_pen"], "cf_dis": f["cf_dis_pen"]},
            "prior": ({"nis": p_nis["prior"], "ood": f["ood_pen"], "spread": f["spread_pen"]}
                      if f["prior_ok"] else {"unavailable": np.inf}),
        }
        for c in CHANNELS:
            tgt = float(np.exp(-sum(pen[c].values())))
            tr = self.trust[c]
            a = cfg.trust_down if tgt < tr else cfg.trust_up
            self.trust[c] = tr + a * (tgt - tr)
        self.pen = pen
        return dict(self.trust), reanchor
