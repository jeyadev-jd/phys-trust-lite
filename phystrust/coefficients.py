"""Published MQ135 CO2 power-law coefficient sets -> consensus prior with uncertainty.

All sets are digitised from the same datasheet plot, so the spread is digitising
error, not sensor-to-sensor variation. Treat as a physics prior, not ground truth.

With R0 anchored at C_ref, a cancels exactly:
    log C - log C_ref = b * [log(Rs/CF) - log(Rs_ref/CF_ref)]
so coefficient uncertainty in log C is |log C - log C_ref| * sigma_b / |b|.
"""
import numpy as np

# (source, a, b); b=None where only a is recoverable from accessible copy
SETS = [
    ("GeorgK / Gironi 2014", 116.6020682, -2.769034857),
    ("MQUnifiedsensor (miguel5612)", 110.47, -2.862),
    ("Kalra et al. 2016 (partial)", 121.0, None),
]

SIGMA_A_FLOOR = 5.0
SIGMA_B_FLOOR = 0.05


def consensus():
    a = np.array([s[1] for s in SETS])
    b = np.array([s[2] for s in SETS if s[2] is not None])
    return dict(a=float(a.mean()), sigma_a=float(max(a.std(ddof=1), SIGMA_A_FLOOR)),
                b=float(b.mean()), sigma_b=float(max(b.std(ddof=1), SIGMA_B_FLOOR)))


def coef_var_log(logc, log_cref, b, sigma_b):
    """Variance added to log-index measurement by b uncertainty (R0 anchored at C_ref)."""
    return ((logc - log_cref) * sigma_b / b) ** 2


if __name__ == "__main__":
    for s in SETS:
        print(s)
    print(consensus())
