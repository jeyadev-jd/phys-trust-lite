"""Overnight burn-in health watch.  pythonw tools/overnight_watch.py --hours 7

Every 10 min summarises the last 10 min of data/burnin.csv into results/overnight/health.csv:
ADC mean/std, within-burst noise, drift per hour, T/RH, gaps, and a verdict. At the end, picks the
cleanest (lowest, most stable) 10-min window of the night as a candidate clean-air reference
and writes results/overnight/summary.txt. With --calibrate, recalibrates from that window only if the
quality gates pass (burst noise < 3 counts, window stable, settled); the old bundle is backed up first.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
LOG, OUT = "data/burnin.csv", "results/overnight"


def check(d):
    if len(d) < 60:
        return dict(n=len(d), verdict="NO DATA (board unplugged or logger stopped)")
    a = d["adc_mean"]
    slope = np.polyfit((d["unix"] - d["unix"].iloc[0]) / 3600, a, 1)[0]
    burst = float(d["adc_std"].median())
    v = []
    if burst > 3:
        v.append(f"NOISY burst std {burst:.1f} (wiring/ground)")
    if a.median() > 600:
        v.append("ADC very high (gas source or fault)")
    if abs(slope) > 0.05 * a.median():
        v.append("still settling")
    return dict(n=len(d), adc=round(float(a.median()), 1), adc_sd=round(float(a.std()), 2), burst_std=round(burst, 2),
                slope_per_h=round(float(slope), 2), T=round(float(d["T"].median()), 1),
                RH=round(float(d["RH"].median()), 1), verdict="; ".join(v) or "OK stable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=7)
    ap.add_argument("--every", type=float, default=600)
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()
    t_end = t_start + a.hours * 3600
    rows = []
    while time.time() < t_end:
        time.sleep(a.every)
        try:
            d = pd.read_csv(LOG)
            d = d[d["unix"] > time.time() - a.every]
            r = dict(time=time.strftime("%Y-%m-%d %H:%M"), **check(d))
        except Exception as e:  # file mid-write etc.
            r = dict(time=time.strftime("%Y-%m-%d %H:%M"), verdict=f"read error {e}")
        rows.append(r)
        pd.DataFrame(rows).to_csv(f"{OUT}/health.csv", index=False)

    d = pd.read_csv(LOG)
    d = d[d["unix"] >= t_start].reset_index(drop=True)
    lines = [f"Overnight watch {time.strftime('%Y-%m-%d %H:%M', time.localtime(t_start))} for {a.hours} h, {len(d)} rows"]
    if len(d) > 1200:
        ok = (d["adc_std"] < 3) & (d["unix"] >= d["unix"].iloc[0] + 2 * 3600)  # search only after warm-up
        roll = d["adc_mean"].where(ok).rolling(600, min_periods=540)
        score = roll.median() + 3 * roll.std()
        k = int(score.idxmin()) if score.notna().any() else None
        if k is not None:
            w = d.iloc[k - 599:k + 1]
            t0 = time.strftime("%H:%M", time.localtime(w["unix"].iloc[0]))
            t1 = time.strftime("%H:%M", time.localtime(w["unix"].iloc[-1]))
            lines.append(f"cleanest 10 min: {t0}-{t1}, ADC {w['adc_mean'].median():.1f} +- {w['adc_mean'].std():.2f}, "
                         f"T {w['T'].median():.1f}, RH {w['RH'].median():.0f}")
            gates = dict(burst_ok=bool(w["adc_std"].median() < 3), stable=bool(w["adc_mean"].std() < 0.02 * w["adc_mean"].median()),
                         warmed=bool(w["unix"].iloc[0] - d["unix"].iloc[0] > 2 * 3600))
            lines.append(f"gates: {gates}")
            if a.calibrate and all(gates.values()):
                shutil.copy("models/bundle.joblib", "models/bundle_pre_overnight.joblib")
                date = time.strftime("%Y-%m-%d", time.localtime(w["unix"].iloc[0]))
                hist = min(6.0, (w["unix"].iloc[0] - d["unix"].iloc[0]) / 3600)
                res = subprocess.run([sys.executable.replace("pythonw", "python"), "tools/session_report.py", "--clean", t0, t1,
                                      "--date", date, "--history-h", f"{hist:.1f}"], capture_output=True, text=True)
                lines.append("CALIBRATED (backup models/bundle_pre_overnight.joblib)" if res.returncode == 0 else "calibration FAILED")
                lines.append(res.stdout[-1500:] + res.stderr[-800:])
            elif a.calibrate:
                lines.append("NOT calibrated: gates failed; old bundle kept")
        else:
            lines.append("no quiet window with low burst noise: fix wiring before calibrating")
        last = d.tail(1800)
        lines.append(f"last 30 min: ADC {last['adc_mean'].median():.1f}, burst std {last['adc_std'].median():.2f}, "
                     f"T {last['T'].median():.1f}")
    open(f"{OUT}/summary.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
