"""All tunables in one place. Units: kOhm, seconds, index = CO2-equivalent ppm (relative)."""
import json
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # --- hardware / sensor physics (MQ135 datasheet prior, GeorgK coefficients)
    rl_kohm: float = 10.0          # measure A0-GND on your module! cheap boards ship 1 kOhm
    r0_kohm: float = 76.63         # replaced by clean-air calibration
    a: float = 116.024             # consensus of published sets (coefficients.py)
    b: float = -2.8155
    sigma_b: float = 0.0657        # b uncertainty; a cancels once R0 anchored
    c_ref: float = 426.0           # assumed clean-air index for R0 calibration
    adc_max: int = 1023

    # --- process physics (single-zone mass balance)
    lam: float = 1.0 / 3600        # air exchange rate [1/s]; fit with ASTM E741 decay
    lam_max: float = 20.0 / 3600   # fastest plausible room decay (window open)
    c_bg: float = 426.0            # outdoor background index
    rise_max: float = 3.0          # max plausible room-level rise S_max/V [index/s]
    rate_win: int = 5              # samples for rate plausibility check

    # --- filter noise
    q_c: float = 1.0               # process noise on C [index/sqrt(s)]
    q_ld: float = 2e-6             # drift random walk [log/sqrt(s)]
    r_raw: float = 0.15 ** 2       # measurement var, log-index units
    r_phys: float = 0.15 ** 2
    r_prior: float = 0.05 ** 2

    # --- trust estimator
    nis_alpha: float = 0.3         # EWMA on NIS
    trust_down: float = 1.0        # fast drop
    trust_up: float = 0.05         # slow recovery
    trust_eps: float = 1e-3
    k_rate: float = 2.0            # weight of rate-plausibility penalty
    recover_s: float = 30.0        # plausible seconds before re-anchoring
    lag_win: int = 30              # samples for dT/dt, dRH/dt
    dtdt_thr: float = 0.05         # degC/s  (DHT11 lag regime)
    drhdt_thr: float = 0.2         # %RH/s
    drift_var_rate: float = -1.0   # k_d [log-index^2/s] for band growth since last anchor;
                                   # < 0 means (b * q_ld)^2, i.e. consistent with the filter's drift model
    abc_window_s: int = 86400      # automatic baseline calibration: rolling-min window (>= 24 h)
    abc_enabled: bool = True       # assumes room aired to outdoor background once per window
    noise_ratio_thr: float = 4.0   # diff-variance ratio above which noise penalty starts
    sys_trust_floor: float = 0.05  # sigma_sys scaled by 1/sqrt(max(trust, floor))
    cf_dis_thr: float = 0.15       # |log CF_ds - log CF_refit|
    spread_thr: float = 0.02       # ensemble std (log units)
    ood_thr: float = 23.2          # chi2(10, 0.99)

    # --- learned prior (deep ensemble)
    prior_kind: str = "lstm"       # "lstm" | "tcn" | "transformer" (sequence) or "mlp" (summary)

    # --- operation (PHYS-TRUST 2.0)
    node_id: str = "node-A"
    adaptive_compute: bool = True  # 1 prior member when TRUSTED, 5 otherwise

    # --- uncertainty
    alpha: float = 0.1             # 90% intervals

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))
