"""Past-data validation of the DHT11 against Open-Meteo outdoor weather (Chennai).

python tools/validate_weather.py   -> results/weather/{hourly.csv, stats.csv, weather.png}
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phystrust.reference import CHENNAI, compare, weather_hourly  # noqa: E402

OUT = "results/weather"


def node_hourly(path="data/burnin.csv"):
    d = pd.read_csv(path)
    d = d[(d.dht_ok == 1) & (d.RH >= 1) & (d["T"] >= 1)]
    d["time"] = pd.to_datetime(d.unix, unit="s", utc=True).dt.tz_convert(CHENNAI["tz"]).dt.tz_localize(None)
    h = d.set_index("time")[["T", "RH"]].resample("1h").mean()
    n = d.set_index("time")["T"].resample("1h").count()
    return h[n >= 600].reset_index()           # hours with >= 10 min of data


def main():
    os.makedirs(OUT, exist_ok=True)
    node = node_hourly()
    ref = weather_hourly(past_days=7)
    stats, m = compare(node, ref)
    m.to_csv(f"{OUT}/hourly.csv", index=False)
    pd.DataFrame([stats]).to_csv(f"{OUT}/stats.csv", index=False)
    for k, v in stats.items():
        print(f"{k:16s} {v:.3f}" if isinstance(v, float) else f"{k:16s} {v}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(m.time, m["T"], "o-", label="DHT11 (indoor node)")
    ax[0].plot(m.time, m.T_ref, "s-", label="Open-Meteo outdoor, Chennai")
    ax[0].set_ylabel("°C")
    ax[0].legend(fontsize=8)
    ax[1].plot(m.time, m["RH"], "o-", label="DHT11")
    ax[1].plot(m.time, m.RH_ref, "s-", label="Open-Meteo")
    ax[1].set_ylabel("%RH")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/weather.png", dpi=120)
    print(f"plot: {OUT}/weather.png")


if __name__ == "__main__":
    main()
