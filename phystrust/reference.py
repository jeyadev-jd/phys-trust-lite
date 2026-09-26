"""External references: Open-Meteo weather (free, no key) for past and real-time T/RH validation.

Only the configured city coordinates are sent. Outdoor weather is not the same air as an indoor node:
compare trends (correlation, lag) and report the indoor-outdoor offset, not equality.
"""
import json
import time
import urllib.parse
import urllib.request

import pandas as pd

CHENNAI = dict(name="Chennai", lat=13.0827, lon=80.2707, tz="Asia/Kolkata")
FORECAST = "https://api.open-meteo.com/v1/forecast"


def _get(url, params, timeout=20):
    q = url + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(q, timeout=timeout) as r:
        return json.loads(r.read())


def weather_hourly(past_days=7, loc=CHENNAI):
    """Hourly outdoor T/RH for the last `past_days` days (local time, naive)."""
    j = _get(FORECAST, dict(latitude=loc["lat"], longitude=loc["lon"], timezone=loc["tz"],
                            hourly="temperature_2m,relative_humidity_2m", past_days=past_days, forecast_days=1))
    h = j["hourly"]
    return pd.DataFrame(dict(time=pd.to_datetime(h["time"]), T_ref=h["temperature_2m"],
                             RH_ref=h["relative_humidity_2m"]))


_cache = {"t": 0.0, "v": None}


def weather_now(loc=CHENNAI, max_age_s=900):
    """Current outdoor T/RH, cached 15 min. Returns dict or None on network error."""
    if _cache["v"] is not None and time.time() - _cache["t"] < max_age_s:
        return _cache["v"]
    try:
        j = _get(FORECAST, dict(latitude=loc["lat"], longitude=loc["lon"], timezone=loc["tz"],
                                current="temperature_2m,relative_humidity_2m"))
        c = j["current"]
        v = dict(time=pd.Timestamp(c["time"]), T_ref=c["temperature_2m"], RH_ref=c["relative_humidity_2m"])
    except Exception:
        return _cache["v"]
    _cache.update(t=time.time(), v=v)
    return v


def compare(node_hourly, ref, cols=(("T", "T_ref"), ("RH", "RH_ref")), max_lag_h=6):
    """Agreement stats between node hourly means and reference: bias, RMSE, r, best lag."""
    m = node_hourly.merge(ref, on="time", how="inner")
    out = {"hours": len(m)}
    for a, b in cols:
        x, y = m[a].values, m[b].values
        out[f"{a}_bias"] = float((x - y).mean())
        out[f"{a}_rmse"] = float(((x - y) ** 2).mean() ** 0.5)
        out[f"{a}_r"] = float(pd.Series(x).corr(pd.Series(y)))
        lags = {}
        for L in range(-max_lag_h, max_lag_h + 1):
            ys = pd.Series(y).shift(L)
            if ys.notna().sum() >= 6:                    # need enough overlap for a meaningful r
                lags[L] = pd.Series(x).corr(ys)
        if not lags:
            out[f"{a}_best_lag_h"] = out[f"{a}_r_at_lag"] = float("nan")
            continue
        best = max(lags, key=lambda k: -1 if pd.isna(lags[k]) else lags[k])
        out[f"{a}_best_lag_h"], out[f"{a}_r_at_lag"] = int(best), float(lags[best])
    return out, m
