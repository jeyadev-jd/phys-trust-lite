"""Ablation benchmark: fault injection, RMSE, detection P/R/F1/latency, PICP/width, decorrelation.

Synthetic run is non-circular: the true sensor exponent b_true is swept over
{b - sigma_b, b, b + sigma_b} while the estimator keeps the consensus prior b.
Detection threshold is tuned on a validation seed, reported on a separate test seed.

python -m phystrust.evaluate                    # fully synthetic
python -m phystrust.evaluate --bundle models/bundle.joblib --log data/clean.csv   # real log + injected faults
"""
import argparse
import os

import numpy as np
import pandas as pd

from . import faults
from .calibrate import build_bundle
from .config import Config
from .pipeline import MODES, Bundle, run_frame
from .simulate import test_day, training_day

THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
VAL_SEED, TEST_SEED = 11, 7


def inject_suite(df, rl):
    n = len(df)
    df = faults.noise(df, int(0.25 * n), int(0.25 * n) + 120, rl=rl)
    df = faults.stuck(df, int(0.45 * n), int(0.45 * n) + 60)
    df = faults.drift(df, int(0.75 * n), 0.002, rl=rl)
    return df


def alarm_old(out, thr):
    """Previous rule (for comparison on the same labels): either sensor channel distrusted."""
    return np.minimum(out["trust_raw"], out["trust_phys"]).values < thr


def alarm(out, thr, min_on=2, hold=10):
    """Sensor-fault alarm: BOTH sensor channels distrusted (raw-only drop = cross-sensitivity,
    physics-only drop = T/RH model issue; neither is a sensor fault). Hysteresis: condition must
    hold min_on samples; alarm latched for `hold` samples after it clears."""
    cond = np.maximum(out["trust_raw"], out["trust_phys"]).values < thr
    a = np.zeros(len(cond), bool)
    run = latch = 0
    for i, c in enumerate(cond):
        run = run + 1 if c else 0
        if run >= min_on:
            latch = hold
        a[i] = run >= min_on or latch > 0
        if not c and latch > 0:
            latch -= 1
    return a


def detection(out, df, thr, rule=alarm):
    kind = df["fault_kind"].values
    local = np.isin(kind, ["spike", "stuck", "noise"])  # same labels for both rules
    det = rule(out, thr)
    tp, fp, fn = (det & local).sum(), (det & ~local).sum(), (~det & local).sum()
    lat = []
    for s in np.flatnonzero(np.diff(np.r_[0, local.astype(int)]) == 1):
        hit = np.flatnonzero(det[s:s + 60])
        lat.append(hit[0] if len(hit) else np.nan)
    lat = np.array(lat, float)
    return dict(det_precision=tp / max(tp + fp, 1), det_recall=tp / max(tp + fn, 1),
                det_f1=2 * tp / max(2 * tp + fp + fn, 1),
                det_latency_s=np.nanmean(lat) if np.isfinite(lat).any() else np.nan,
                events_missed=int(np.isnan(lat).sum()))


def metrics(out, truth, df, thr=0.5, thr_old=0.5):
    lt, lf = np.log(truth), np.log(out["fused"].values)
    clean = df["fault_kind"].values == ""
    pre = np.arange(len(df)) < int(0.75 * len(df))
    inside = (truth >= out["lo"].values) & (truth <= out["hi"].values)
    return dict(
        rmse_log=np.sqrt(np.mean((lt - lf) ** 2)),
        rmse_log_clean=np.sqrt(np.mean((lt - lf)[clean] ** 2)),
        rmse_log_fault=np.sqrt(np.mean((lt - lf)[~clean] ** 2)) if (~clean).any() else np.nan,
        picp_clean=inside[clean].mean(), picp_fault=inside[~clean].mean() if (~clean).any() else np.nan,
        picp_pre_drift=inside[pre].mean(), picp_post_drift=inside[~pre].mean(),
        picp_local_fault=inside[np.isin(df["fault_kind"].values, ["spike", "stuck", "noise"])].mean(),
        picp_drift=(inside[df["fault_kind"].values == "drift"].mean()
                    if (df["fault_kind"].values == "drift").any() else np.nan),
        width_log=np.mean(np.log(out["hi"] / out["lo"])),
        **detection(out, df, thr),
        **{f"old_{k}": v for k, v in detection(out, df, thr_old, alarm_old).items()},
        corr_T=abs(np.corrcoef(lf[clean], df["T"].values[clean])[0, 1]),
        rmse_log_post_drift=np.sqrt(np.mean((lt - lf)[~pre] ** 2)),
        drift_true_end=1.002 ** ((df["fault_kind"].values == "drift").sum() / 60.0),
        drift_est_end=float(out["drift"].iloc[-1]),
    )


def _sim_case(b, b_true):
    df = inject_suite(test_day(seed=TEST_SEED, b_true=b_true), b.cfg.rl_kohm)
    truth = df["c_local"].where(df["fault"] == 0, df["c_true"]).values
    val = inject_suite(test_day(seed=VAL_SEED, b_true=b_true), b.cfg.rl_kohm)
    val_out = run_frame(val, b, "adaptive")
    thr = max(THRESHOLDS, key=lambda t: detection(val_out, val, t)["det_f1"])
    thr_old = max(THRESHOLDS, key=lambda t: detection(val_out, val, t, alarm_old)["det_f1"])
    return df, truth, (thr, thr_old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle")
    ap.add_argument("--log", help="real clean log; truth = consensus of refit-physics")
    ap.add_argument("--out", default="results")
    ap.add_argument("--q-ld", type=float, help="override drift random-walk noise")
    ap.add_argument("--k-d", type=float, help="override drift band growth rate k_d [log^2/s]")
    ap.add_argument("--seeds", type=int, default=5, help="test seeds for paired fixed-vs-adaptive")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cases = []  # (label, bundle, df, truth, thr)
    if a.log:
        b = Bundle.load(a.bundle)
        clean = pd.read_csv(a.log)
        clean["fault"], clean["fault_kind"] = 0, ""
        ref = run_frame(clean, b, "refit_phys")
        truth = ref["fused"].rolling(61, center=True, min_periods=1).median().values
        n = len(clean)
        df = faults.spike(clean, int(0.1 * n), rl=b.cfg.rl_kohm)
        df = faults.spike(df, int(0.55 * n), rl=b.cfg.rl_kohm)
        cases.append(("real", b, inject_suite(df, b.cfg.rl_kohm), truth, (0.5, 0.5)))
    else:
        cfg = Config()
        if a.q_ld is not None:
            cfg.q_ld = a.q_ld
        if a.k_d is not None:
            cfg.drift_var_rate = a.k_d
        for label, bt in (("b-sigma", cfg.b - cfg.sigma_b), ("b", cfg.b),
                          ("b+sigma", cfg.b + cfg.sigma_b)):
            b, info = build_bundle(training_day(b_true=bt), cfg, truth_col="c_true")
            print(f"[{label}] b_true={bt:.4f} q={info['q']:.2f} sigma_sys={info['sigma_sys']:.3f} "
                  f"unscaled_cov90={info['unscaled_cov90']:.2f}")
            cases.append((label, b, *_sim_case(b, bt)))

    rows, trace = [], None
    for label, b, df, truth, thr in cases:
        for mode in MODES:
            out = run_frame(df, b, mode)
            rows.append(dict(b_true=label, mode=mode, det_thr=thr[0], det_thr_old=thr[1],
                             **metrics(out, truth, df, *thr)))
            if mode == "adaptive" and label in ("b", "real"):
                trace = (out, truth, df)
    tab = pd.DataFrame(rows).set_index(["b_true", "mode"]).round(4)
    pd.set_option("display.width", 250)
    print(tab.to_string())
    tab.to_csv(f"{a.out}/ablation.csv")
    with open(f"{a.out}/ablation.md", "w", encoding="utf-8") as f:
        f.write(tab.to_markdown())
    if not a.log and a.seeds:
        paired(cases[1][1], cfg.b, a.seeds, a.out, cases[1][4])
    out, truth, df = trace
    out.assign(truth=truth, fault_kind=df["fault_kind"].values).to_csv(
        f"{a.out}/adaptive_trace.csv", index=False)
    _plot(out, truth, df, f"{a.out}/adaptive_trace.png")


def paired(b, b_true, n, outdir, thr=(0.5, 0.5)):
    """Fixed vs adaptive over n independent test seeds: mean +/- std and paired t."""
    rows = []
    for seed in range(100, 100 + n):
        df = inject_suite(test_day(seed=seed, b_true=b_true), b.cfg.rl_kohm)
        truth = df["c_local"].where(df["fault"] == 0, df["c_true"]).values
        r = {"seed": seed}
        for mode in ("fixed", "adaptive"):
            m = metrics(run_frame(df, b, mode), truth, df, *thr)
            r[f"{mode}_rmse"], r[f"{mode}_picp_fault"] = m["rmse_log"], m["picp_fault"]
            r[f"{mode}_picp_clean"] = m["picp_clean"]
            if mode == "adaptive":
                for k in ("det_precision", "det_recall", "det_f1", "det_latency_s",
                          "old_det_f1", "old_det_latency_s"):
                    r[k] = m[k]
        rows.append(r)
    t = pd.DataFrame(rows)
    d = t["fixed_rmse"] - t["adaptive_rmse"]
    tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    from scipy import stats
    pval = float(stats.ttest_rel(t["fixed_rmse"], t["adaptive_rmse"]).pvalue)
    t.to_csv(f"{outdir}/paired_seeds.csv", index=False)
    print(t.round(4).to_string(index=False))
    print(f"fixed {t.fixed_rmse.mean():.4f}+/-{t.fixed_rmse.std(ddof=1):.4f}  "
          f"adaptive {t.adaptive_rmse.mean():.4f}+/-{t.adaptive_rmse.std(ddof=1):.4f}  "
          f"paired diff {d.mean():.4f}+/-{d.std(ddof=1):.4f}  t({len(d) - 1})={tstat:.2f}  p={pval:.4f}")
    print(t.drop(columns="seed").agg(["mean", "std"]).round(3).to_string())


def _plot(out, truth, df, path):
    import matplotlib.pyplot as plt
    t = np.arange(len(out)) / 60
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(t, out["idx_raw"], lw=0.6, alpha=0.6, label="raw")
    ax[0].plot(t, out["idx_phys"], lw=0.6, alpha=0.6, label="physics-corrected")
    ax[0].fill_between(t, out["lo"], out["hi"], alpha=0.3, label="90% conformal")
    ax[0].plot(t, out["fused"], lw=1.2, label="fused")
    ax[0].plot(t, truth, "k--", lw=0.8, label="truth")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("CO2-eq index")
    ax[0].legend(fontsize=8)
    for ch in ("raw", "phys", "prior"):
        ax[1].plot(t, out[f"trust_{ch}"], label=ch)
    ax[1].set_ylabel("trust")
    ax[1].legend(fontsize=8)
    f = df["fault_kind"].values != ""
    ax[1].fill_between(t, 0, 1, where=f, color="red", alpha=0.1)
    ax[2].plot(t, df["T"], label="T")
    ax[2].plot(t, df["RH"], label="RH")
    ax[2].set_xlabel("min")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
