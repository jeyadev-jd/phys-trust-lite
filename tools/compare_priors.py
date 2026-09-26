"""MLP-ensemble vs LSTM-ensemble learned prior, same synthetic data and test seeds.

python tools/compare_priors.py            -> results/prior_compare.csv, prints paired stats
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phystrust.calibrate import build_bundle  # noqa: E402
from phystrust.config import Config  # noqa: E402
from phystrust.evaluate import inject_suite, metrics  # noqa: E402
from phystrust.pipeline import run_frame  # noqa: E402
from phystrust.simulate import test_day, training_day  # noqa: E402

SEEDS = range(100, 105)


def main():
    rows, bundles = [], {}
    for kind in ("mlp", "lstm", "tcn", "transformer"):
        cfg = Config(prior_kind=kind, adaptive_compute=False)  # full ensemble: fair comparison
        t0 = time.time()
        bundles[kind], info = build_bundle(training_day(b_true=cfg.b), cfg, truth_col="c_true")
        print(f"[{kind}] calibration {time.time() - t0:.0f} s, q={info['q']:.2f}, sigma_sys={info['sigma_sys']:.3f}")
    for seed in SEEDS:
        df = inject_suite(test_day(seed=seed, b_true=Config().b), 10.0)
        truth = df["c_local"].where(df["fault"] == 0, df["c_true"]).values
        clean = df["fault_kind"].values == ""
        for kind, b in bundles.items():
            t0 = time.time()
            out = run_frame(df, b, "adaptive")
            ms = (time.time() - t0) / len(df) * 1000
            m = metrics(out, truth, df, 0.3, 0.3)
            p = np.log(out["idx_prior"].values)          # prior's one-step-ahead prediction
            ok = np.isfinite(p) & clean
            rows.append(dict(seed=seed, prior=kind, rmse_log=m["rmse_log"], picp_clean=m["picp_clean"],
                             picp_local_fault=m["picp_local_fault"], det_f1=m["old_det_f1"],
                             prior_1step_rmse=np.sqrt(np.mean((p[ok] - np.log(truth[ok])) ** 2)),
                             trust_prior_median=float(np.median(out["trust_prior"])), ms_per_step=ms))
        print(f"seed {seed} done")
    t = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    t.to_csv("results/prior_compare.csv", index=False)
    print(t.groupby("prior").agg(["mean", "std"]).drop(columns="seed").round(4).T.to_string())
    base = t[t.prior == "lstm"].set_index("seed")
    for other in ("mlp", "tcn", "transformer"):
        o = t[t.prior == other].set_index("seed")
        for col in ("rmse_log", "prior_1step_rmse"):
            d = o[col] - base[col]
            print(f"{col}: {other} - lstm = {d.mean():+.4f} +/- {d.std(ddof=1):.4f}, "
                  f"paired p = {stats.ttest_rel(o[col], base[col]).pvalue:.4f}")


if __name__ == "__main__":
    main()
