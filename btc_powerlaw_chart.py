#!/usr/bin/env python3
"""BTC power-law + cycle chart (TechDev/TradingView dark style) for the EQM dashboard.

Model: log10(P) = c0 + c1*log10(t - t0) + a*sin(2pi t/T) + b*cos(2pi t/T)
with ENDOGENOUS start point t0 and period T found via grid search
(for fixed t0/T the model is linear -> exact lstsq).

Data: Coin Metrics community API (PriceUSD daily, since 2010, no key required).
Outputs (next to this script):
  btc_powerlaw.png        — the chart
  btc_powerlaw_info.json  — fit stats consumed by eqm_dashboard.py
"""
import os
import json
import urllib.request
import numpy as np
import pandas as pd
try:
    import matplotlib
except ImportError:  # CI installs base deps only; pull matplotlib on demand
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DIR = os.path.dirname(os.path.abspath(__file__))
GENESIS = pd.Timestamp("2009-01-03")
BG = "#0c0e12"; AX = "#9598a1"
C_PRICE = "#f8f9fa"; C_TREND = "#d9b54a"; C_MODEL = "#4a90d9"
C_TOP = "#f23645"; C_LOW = "#2ebd85"


def fetch_coinmetrics():
    url = ("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
           "?assets=btc&metrics=PriceUSD&frequency=1d&page_size=10000")
    rows = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            j = json.loads(r.read())
        rows += j["data"]
        url = j.get("next_page_url")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    df["price"] = pd.to_numeric(df["PriceUSD"], errors="coerce")
    return df[["date", "price"]].dropna().sort_values("date").reset_index(drop=True)


df = fetch_coinmetrics()
df = df[df["price"] > 0.01].reset_index(drop=True)
df["t"] = (df["date"] - GENESIS).dt.days.astype(float)
df["logp"] = np.log10(df["price"])
df["smooth"] = 10 ** df["logp"].rolling(7, center=True, min_periods=1).mean()
print(f"Data: {df['date'].min().date()} .. {df['date'].max().date()}  n={len(df)}  "
      f"last ${df['price'].iloc[-1]:,.0f}")


def fit(t0, T):
    t = df["t"].values
    w = 2 * np.pi * t / T
    X = np.column_stack([np.ones(len(df)), np.log10(t - t0), np.sin(w), np.cos(w)])
    coef, *_ = np.linalg.lstsq(X, df["logp"].values, rcond=None)
    return coef, np.sum((df["logp"].values - X @ coef) ** 2)


best = None
for t0 in np.arange(-5475, 521, 15.0):
    for T in np.arange(600, 2401, 20.0):
        coef, sse = fit(t0, T)
        if best is None or sse < best[0]:
            best = (sse, t0, T, coef)
sse, t0, T, c = best
r2 = 1 - sse / np.sum((df["logp"].values - df["logp"].values.mean()) ** 2)
print(f"Fit: t0={(GENESIS + pd.Timedelta(days=float(t0))).date()}  "
      f"T={T / 365.25:.2f}y  exp={c[1]:.2f}  R2={r2:.4f}")


def m_pl(t):
    return c[0] + c[1] * np.log10(t - t0)


def m_full(t):
    w = 2 * np.pi * t / T
    return m_pl(t) + c[2] * np.sin(w) + c[3] * np.cos(w)


# future horizon: at least one full cycle so both next top and trough exist
horizon_days = max(int(365 * 3.4), int(T) + 60)
fut = pd.date_range(df["date"].iloc[-1] + pd.Timedelta(days=1),
                    df["date"].iloc[-1] + pd.Timedelta(days=horizon_days), freq="D")
all_dates = pd.DatetimeIndex(df["date"]).append(fut)
t_all = (all_dates - GENESIS).days.astype(float)

phi = np.arctan2(c[3], c[2])


def extrema(target):
    res, k = [], np.floor((2 * np.pi * t_all[0] / T + phi - target) / (2 * np.pi))
    while True:
        t_e = (target - phi + 2 * np.pi * k) * T / (2 * np.pi)
        if t_e > t_all[-1]:
            break
        if t_e >= t_all[0]:
            res.append(t_e)
        k += 1
    return np.array(res)


tops, troughs = extrema(np.pi / 2), extrema(3 * np.pi / 2)
today = df["date"].iloc[-1]
t_today = df["t"].iloc[-1]

fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
ax.set_facecolor(BG)
ax.plot(all_dates, 10 ** m_pl(t_all), color=C_TREND, lw=1.3, ls=(0, (2, 3)), zorder=3)
ax.plot(all_dates, 10 ** m_full(t_all), color=C_MODEL, lw=1.5, ls=(0, (5, 4)),
        alpha=0.9, zorder=3)
ax.plot(df["date"], df["smooth"], color=C_PRICE, lw=1.5, zorder=5)

for t_es, col in [(tops, C_TOP), (troughs, C_LOW)]:
    for t_e in t_es:
        d_e = GENESIS + pd.Timedelta(days=float(t_e))
        v = 10 ** m_full(np.array([t_e]))[0]
        ax.scatter([d_e], [v], color=col, s=90, zorder=7, edgecolors=BG, linewidths=1.5)

# only the two future extrema get a label (no other text on the canvas)
info_extrema = {}
for kind, t_es, col, dy, va in [("trough", troughs, C_LOW, -16, "top"),
                                ("top", tops, C_TOP, 16, "bottom")]:
    future = t_es[t_es > t_today]
    if len(future) == 0:
        continue
    t_e = future[0]
    d_e = GENESIS + pd.Timedelta(days=float(t_e))
    v = 10 ** m_full(np.array([t_e]))[0]
    ax.annotate(f"{d_e.strftime('%b %Y')}", xy=(d_e, v), xytext=(0, dy),
                textcoords="offset points", ha="center", va=va, fontsize=13,
                color=col, fontweight="bold")
    info_extrema[kind] = {"date": str(d_e.date()), "price": round(float(v))}

ax.text(0.42, 0.18, f"Power-Law Trend  +  {T / 365.25:.1f}-Year Cycle",
        transform=ax.transAxes, fontsize=17, color=C_MODEL, fontweight="bold", ha="center")

ax.set_yscale("log")
ax.set_xlim(pd.Timestamp("2010-06-01"), today + pd.Timedelta(days=horizon_days + 90))
ax.set_ylim(0.04, 3e6)
yt = [0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]
ax.set_yticks(yt)
ax.set_yticklabels(["0.10", "1.00", "10", "100", "1,000", "10,000", "100,000", "1,000,000"],
                   fontsize=10, color=AX)
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.get_xticklabels(), fontsize=10, color=AX)
ax.tick_params(colors=AX, length=0)
ax.minorticks_off()
for s in ax.spines.values():
    s.set_visible(False)
ax.grid(False)
plt.tight_layout(pad=1.5)
out = os.path.join(DIR, "btc_powerlaw.png")
plt.savefig(out, dpi=120, facecolor=BG, bbox_inches="tight")
print("saved", out)

fair = 10 ** m_full(np.array([t_today]))[0]

# banana zone: distance from the pure power-law trend + percentile of that
# residual in full history (banana zone = top decile of all-time deviation)
resid = df["logp"].values - m_pl(df["t"].values)
resid_pct = float((resid < resid[-1]).mean() * 100)
trend_val = 10 ** m_pl(np.array([t_today]))[0]

info = {
    "data_date": str(today.date()),
    "price": round(float(df["price"].iloc[-1])),
    "fair_value": round(float(fair)),
    "deviation": round(float(df["price"].iloc[-1] / fair - 1), 4),
    "trend_value": round(float(trend_val)),
    "trend_deviation": round(float(df["price"].iloc[-1] / trend_val - 1), 4),
    "resid_pct": round(resid_pct, 1),
    "banana_zone": bool(resid_pct >= 90),
    "t0": str((GENESIS + pd.Timedelta(days=float(t0))).date()),
    "period_years": round(float(T / 365.25), 2),
    "exponent": round(float(c[1]), 2),
    "amplitude": round(float(10 ** np.hypot(c[2], c[3])), 2),
    "r2": round(float(r2), 4),
    "next_trough": info_extrema.get("trough"),
    "next_top": info_extrema.get("top"),
}
with open(os.path.join(DIR, "btc_powerlaw_info.json"), "w") as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))
