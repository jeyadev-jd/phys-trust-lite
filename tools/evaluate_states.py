"""Evaluate the trust state machine, per-state coverage and adaptive compute (synthetic, 3 test seeds).

python tools/evaluate_states.py  -> results/states/*
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phystrust.calibrate import build_bundle  # noqa: E402
from phystrust.config import Config  # noqa: E402
from phystrust.evaluate import inject_suite  # noqa: E402
from phystrust.pipeline import run_frame  # noqa: E402
from phystrust.simulate import test_day, training_day  # noqa: E402

OUT = "results/states"


def main():
    os.makedirs(OUT, exist_ok=True)
    b, _ = build_bundle(training_day(b_true=Config().b), Config(), truth_col="c_true")
    frames, comp = [], []
    for seed in (100, 101, 102):
        df = inject_suite(test_day(seed=seed, b_true=Config().b), 10.0)
        truth = df["c_local"].where(df["fault"] == 0, df["c_true"]).values
        for ac in (False, True):
            b.cfg.adaptive_compute = ac
            t0 = time.time()
            o = run_frame(df, b, "adaptive")
            ms = (time.time() - t0) / len(df) * 1000
            e = np.log(o["fused"].values) - np.log(truth)
            ins = (truth >= o["lo"].values) & (truth <= o["hi"].values)
            comp.append(dict(seed=seed, adaptive_compute=ac, rmse_log=np.sqrt(np.mean(e ** 2)),
                             picp90=ins.mean(), prior_evals=int(o["prior_evals"].iloc[-1]),
                             full_evals=5 * len(df), ms_per_step=ms))
            if ac:
                frames.append(o.assign(seed=seed, fault_kind=df["fault_kind"].replace("", "clean").values,
                                       inside=ins))
    comp = pd.DataFrame(comp)
    comp.to_csv(f"{OUT}/adaptive_compute.csv", index=False)
    print(comp.groupby("adaptive_compute").mean(numeric_only=True).round(4).to_string())
    a = pd.concat(frames, ignore_index=True)
    conf = pd.crosstab(a["fault_kind"], a["state"], normalize="index").round(3)
    conf.to_csv(f"{OUT}/state_vs_fault.csv")
    print("\nstate share per true condition:\n", conf.to_string())
    cov = a.groupby("state").agg(samples=("inside", "size"), coverage90=("inside", "mean"),
                                 mean_p_ood=("p_ood", "mean")).round(3)
    cov.to_csv(f"{OUT}/coverage_by_state.csv")
    print("\ncoverage by state:\n", cov.to_string())


if __name__ == "__main__":
    main()
