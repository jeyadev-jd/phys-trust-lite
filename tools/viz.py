"""Plotly components for the PHYS-TRUST Lite dashboard.

Always plain Python lists (Streamlit's bundled plotly.js cannot decode plotly>=6 binary arrays).
One colour per source everywhere.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COL = dict(raw="#d62728", phys="#ff7f0e", prior="#2ca02c", fused="#1f77b4",
           band="rgba(31,119,180,0.18)", ok="#2e7d32", warn="#ef6c00", bad="#c62828")
NAME = dict(raw="raw sensor", phys="physics-corrected", prior="learned prior (LSTM ensemble)",
            fused="fused estimate")
ALARM_THR, PHYS_LOW = 0.3, 0.7
NOBAR = {"displayModeBar": False}


def L(v):
    """plain list, NaN -> None so plotly draws gaps"""
    return [None if (isinstance(a, float) and not np.isfinite(a)) else a for a in list(v)]


def states(d):
    """0 = OK, 1 = physics trust low (T/RH swing), 2 = sensor-fault alarm."""
    alarm = np.minimum(d["trust_raw"], d["trust_phys"]).values < ALARM_THR
    phys_low = d["trust_phys"].values < PHYS_LOW
    return np.where(alarm, 2, np.where(phys_low, 1, 0))


def spans(x, mask):
    """contiguous True runs of mask -> list of (x_start, x_end)"""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i == len(mask) - 1) and start is not None:
            end = i if m else i - 1
            out.append((x[start], x[end]))
            start = None
    return out


def hero(x, d, xb, band, events=()):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=L(xb), y=L(band["hi"]), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=L(xb), y=L(band["lo"]), fill="tonexty", fillcolor=COL["band"],
                             line=dict(width=0), name="90 % band"))
    for ch in ("raw", "phys"):
        fig.add_trace(go.Scatter(x=L(x), y=L(d[f"idx_{ch}"]), name=NAME[ch], opacity=0.55,
                                 line=dict(width=1, color=COL[ch])))
    fig.add_trace(go.Scatter(x=L(x), y=L(d["fused"]), name=NAME["fused"], line=dict(width=3, color=COL["fused"])))
    st = states(d)
    for a, b in spans(list(x), st == 2):
        fig.add_vrect(x0=a, x1=b, fillcolor=COL["bad"], opacity=0.15, line_width=0)
    for ex, label in events:
        fig.add_vline(x=ex, line_dash="dot", line_color="#555", line_width=1)
        fig.add_annotation(x=ex, y=1, yref="paper", text=label, showarrow=False, textangle=-90,
                           xanchor="right", yanchor="top", font=dict(size=10, color="#555"))
    lo = max(50.0, float(np.nanmin(band["lo"])) * 0.8)
    hi = float(np.nanmax(band["hi"])) * 1.6
    fig.update_layout(height=400, yaxis_type="log", yaxis_range=[np.log10(lo), np.log10(hi)],
                      yaxis_title="CO₂-eq index (log)", margin=dict(t=10, b=10, l=10, r=10),
                      legend=dict(orientation="h", y=-0.15), hovermode="x unified")
    return fig


def gauges(row):
    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "indicator"}] * 3])
    for i, ch in enumerate(("raw", "phys", "prior")):
        v = float(row[f"trust_{ch}"]) * 100
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=v, number=dict(suffix="%", font=dict(size=18)),
            title=dict(text=NAME[ch].split(" (")[0], font=dict(size=14, color=COL[ch])),
            gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=COL[ch], thickness=0.3),
                       steps=[dict(range=[0, 30], color="#ffcdd2"), dict(range=[30, 70], color="#ffe0b2"),
                              dict(range=[70, 100], color="#c8e6c9")],
                       threshold=dict(line=dict(color="black", width=3), thickness=0.8, value=30))),
            row=1, col=i + 1)
    fig.update_layout(height=230, margin=dict(t=50, b=10, l=30, r=30))
    return fig


def uncertainty_bar(row):
    parts = [("EKF state", row["var_ekf"], "#90caf9"), ("coefficient b", row["var_b"], "#ce93d8"),
             ("systematic", row["var_sys"], "#ffcc80"), ("drift since anchor", row["var_drift"], "#a5d6a7")]
    tot = sum(max(v, 0.0) for _, v, _ in parts) or 1.0
    fig = go.Figure()
    for name, v, c in parts:
        share = 100 * max(v, 0.0) / tot
        fig.add_trace(go.Bar(y=["σ²_total"], x=[share], name=name, orientation="h", marker_color=c,
                             text=[f"{share:.1f}%"], textposition="inside",
                             hovertemplate=f"{name}: %{{x:.1f}}% (var {v:.2e})<extra></extra>"))
    fig.update_layout(barmode="stack", height=230, xaxis=dict(range=[0, 100], ticksuffix="%"),
                      yaxis=dict(showticklabels=False), margin=dict(t=10, b=10, l=10, r=10),
                      legend=dict(orientation="h", y=-0.35, x=0))
    return fig


def alarm_strip(x, d):
    st = states(d)
    fig = go.Figure(go.Heatmap(
        x=L(x), y=["status"], z=[L(st.astype(float))], zmin=0, zmax=2, showscale=False,
        colorscale=[[0, COL["ok"]], [0.33, COL["ok"]], [0.34, COL["warn"]], [0.66, COL["warn"]],
                    [0.67, COL["bad"]], [1, COL["bad"]]],
        hovertemplate="%{x}<extra></extra>"))
    fig.update_layout(height=110, margin=dict(t=0, b=35, l=10, r=10), yaxis=dict(showticklabels=False),
                      xaxis=dict(showticklabels=True, nticks=8))
    return fig


def trust_area(x, d):
    tr = d[["trust_raw", "trust_phys", "trust_prior"]].clip(lower=0).values
    tot = tr.sum(1, keepdims=True)
    share = np.where(tot > 0, tr / np.where(tot > 0, tot, 1), np.nan) * 100
    fig = go.Figure()
    for i, ch in enumerate(("raw", "phys", "prior")):
        fig.add_trace(go.Scatter(x=L(x), y=L(share[:, i]), name=NAME[ch], stackgroup="one",
                                 line=dict(width=0.5, color=COL[ch])))
    fig.update_layout(height=300, yaxis=dict(range=[0, 100], title="normalised trust share %"),
                      margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h", y=-0.2))
    return fig


def cross_sensitivity(d):
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("vs temperature", "vs humidity"))
    ok = np.isfinite(d["idx_raw"]) & (d["idx_raw"] > 0) & (d["idx_phys"] > 0)
    dd = d[ok]
    for col, xs in ((1, "T"), (2, "RH")):
        for ch in ("raw", "phys"):
            fig.add_trace(go.Scattergl(x=L(dd[xs]), y=L(np.log(dd[f"idx_{ch}"])), mode="markers",
                                       marker=dict(size=3, color=COL[ch], opacity=0.35),
                                       name=NAME[ch], showlegend=col == 1), row=1, col=col)
    fig.update_xaxes(title_text="T (°C)", row=1, col=1)
    fig.update_xaxes(title_text="RH (%)", row=1, col=2)
    fig.update_yaxes(title_text="log index", row=1, col=1)
    fig.update_layout(height=320, margin=dict(t=30, b=10, l=10, r=10), legend=dict(orientation="h", y=-0.25))
    return fig


def event_table(x, d):
    """Episodes where status != OK: start, duration, attributed channel, peak raw, fused change."""
    st = states(d)
    raw_low = d["trust_raw"].values < ALARM_THR
    phys_low = d["trust_phys"].values < PHYS_LOW
    rows, i, n = [], 0, len(st)
    while i < n:
        if st[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and st[j + 1] != 0:
            j += 1
        seg = slice(i, j + 1)
        both = (raw_low[seg] & (d["trust_phys"].values[seg] < ALARM_THR)).mean()
        if both > 0.5:
            who = "sensor (both channels)"
        elif raw_low[seg].mean() > phys_low[seg].mean():
            who = "raw channel"
        else:
            who = "physics channel (T/RH)"
        x0, x1 = x[i], x[j]
        dur = (x1 - x0).total_seconds() if isinstance(x0, pd.Timestamp) else float(x1 - x0)
        rows.append({"start": x0.strftime("%H:%M:%S") if isinstance(x0, pd.Timestamp) else f"{x0:.0f} s",
                     "duration s": round(dur + 1),
                     "type": "sensor-fault alarm" if (st[seg] == 2).any() else "physics trust low",
                     "attributed to": who,
                     "peak raw index": f"{np.nanmax(d['idx_raw'].values[seg]):.3g}",
                     "fused change": f"{d['fused'].values[j] - d['fused'].values[i]:+.0f}"})
        i = j + 1
    return pd.DataFrame(rows[::-1])
