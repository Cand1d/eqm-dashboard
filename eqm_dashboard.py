"""EQM Dashboard — Interactive ECharts visualization for BTC, SOL, SUI."""
import pandas as pd
import numpy as np
import requests
import os
import json
import warnings
warnings.filterwarnings('ignore')

DIR = os.path.dirname(os.path.abspath(__file__))

# ─── ASSET CONFIG ────────────────────────────────────────────────────────────

ASSETS = {
    'BTC': {
        'symbol': 'BTC-USD', 'genesis': pd.Timestamp('2009-01-03'),
        'min_window': 365, 'rolling_window': 730,
        'display_start': '2014-01-01', 'cc_early': True, 'cc_to_ts': 1410912000,
    },
    'SOL': {
        'symbol': 'SOL-USD', 'genesis': pd.Timestamp('2020-03-16'),
        'min_window': 180, 'rolling_window': 365,
        'display_start': '2021-01-01', 'cc_early': False,
    },
    'SUI': {
        'symbol': 'SUI20947-USD', 'genesis': pd.Timestamp('2023-05-03'),
        'min_window': 90, 'rolling_window': 180,
        'display_start': '2023-09-01', 'cc_early': False,
    },
}

# ─── DATA FETCHING ───────────────────────────────────────────────────────────

def fetch_prices(name, cfg):
    cache = os.path.join(DIR, f"{name.lower()}_prices.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"  {name}: {len(df)} cached prices")
        return df
    import yfinance as yf
    frames = []
    if cfg.get('cc_early'):
        print(f"  {name}: fetching early data from CryptoCompare...")
        r = requests.get('https://min-api.cryptocompare.com/data/v2/histoday',
                         params={'fsym': name, 'tsym': 'USD', 'limit': 2000, 'toTs': cfg['cc_to_ts']}, timeout=30)
        early = pd.DataFrame([{'date': pd.Timestamp.utcfromtimestamp(p['time']), 'price': p['close']}
                              for p in r.json()['Data']['Data'] if p['close'] > 0]).set_index('date')
        early.index = early.index.tz_localize(None)
        frames.append(early)
    print(f"  {name}: fetching from yfinance ({cfg['symbol']})...")
    btc = yf.download(cfg['symbol'], start="2010-01-01", progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    recent = pd.DataFrame({"price": btc["Close"].values}, index=btc.index)
    recent.index.name = "date"
    recent.index = recent.index.tz_localize(None)
    if frames:
        df = pd.concat([frames[0][frames[0].index < recent.index[0]], recent]).dropna()
    else:
        df = recent.dropna()
    df.to_csv(cache)
    print(f"  {name}: {len(df)} prices")
    return df

# ─── MODEL ───────────────────────────────────────────────────────────────────

def expectile_regression(X, y, tau, max_iter=200, tol=1e-6):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(max_iter):
        r = y - X @ beta
        w = np.where(r >= 0, tau, 1 - tau)
        Xw = X * w[:, None]  # element-wise weighting, avoids O(n^2) diag matrix
        beta_new = np.linalg.solve(Xw.T @ X, Xw.T @ y)
        if np.max(np.abs(beta_new - beta)) < tol:
            break
        beta = beta_new
    return beta

def compute_calibration(X_all, y_all, min_window, df_index, rolling_window=730, tau=0.9999):
    """Track expectile regression coefficient stability. Returns (calibration_date, smoothed_drift_series).
    Uses rolling_window-day p95 of daily beta changes. Calibrated when < 0.1% sustained."""
    prev_beta = None
    changes = []
    dates = []
    for i in range(min_window, len(y_all)):
        beta = expectile_regression(X_all[:i+1], y_all[:i+1], tau)
        if prev_beta is not None:
            rel_change = np.max(np.abs(beta - prev_beta) / (np.abs(prev_beta) + 1e-10))
            changes.append(rel_change)
            dates.append(df_index[i])
        prev_beta = beta.copy()
    if not changes:
        return None, pd.Series(dtype=float)
    s = pd.Series(changes, index=dates)
    # Smooth: rolling p95 over rolling_window/2 days — smooth but responsive
    win = min(rolling_window // 2, len(s) // 3)
    win = max(win, 30)
    smooth = s.rolling(win, min_periods=30).quantile(0.95)
    # Calibrated: smooth drift < 0.05% sustained for 90 days
    cal_date = None
    threshold = 0.0005
    for i in range(len(smooth)):
        if pd.notna(smooth.iloc[i]) and smooth.iloc[i] < threshold:
            remaining = smooth.iloc[i:i+90]
            if len(remaining) >= 60 and remaining.max() < threshold * 1.5:
                cal_date = smooth.index[i]
                break
    return cal_date, smooth

def compute_eqm(df, genesis_date, min_window=365, rolling_window=730, signal_start=None):
    log_prices = np.log(df["price"])
    eqm = {q: pd.Series(index=df.index, dtype=float) for q in ['001', '50', '999']}
    for i in range(min_window, len(df)):
        s = max(0, i + 1 - rolling_window)
        w = log_prices.iloc[s:i+1]
        eqm['001'].iloc[i] = np.quantile(w, 0.001)
        eqm['50'].iloc[i] = np.quantile(w, 0.5)
        eqm['999'].iloc[i] = np.quantile(w, 0.999)
    for k in eqm:
        eqm[k] = np.exp(eqm[k])

    valid_idx = df.index[min_window:]
    days = (df.index - genesis_date).days.values.astype(float) + 1
    log_d = np.log(days)
    X_all = np.column_stack([np.ones(len(df)), log_d, log_d**2])
    X_v = X_all[min_window:]

    # Find calibration date (when coefficients stabilize)
    cal_date, beta_drift = compute_calibration(X_all, log_prices.values, min_window, df.index, rolling_window)
    if signal_start is not None:
        sig_start_ts = pd.Timestamp(signal_start)
        if cal_date is not None:
            sig_start_ts = max(sig_start_ts, cal_date)
    else:
        sig_start_ts = cal_date if cal_date is not None else df.index[-1]
    print(f"  Calibrated: {cal_date.date() if cal_date else 'NOT YET'} | Signals from: {sig_start_ts.date()}")

    er = {}
    for label, tau in [('lower', 0.0001), ('median', 0.5), ('upper', 0.9999)]:
        predictions = np.empty(len(valid_idx))
        for i in range(min_window, len(df)):
            beta = expectile_regression(X_all[:i+1], log_prices.values[:i+1], tau)
            predictions[i - min_window] = X_all[i] @ beta
        er[label] = pd.Series(np.exp(predictions), index=valid_idx)

    score = pd.Series(index=df.index, dtype=float)
    for i in range(min_window, len(df)):
        w = log_prices.iloc[:i+1]
        score.iloc[i] = (w < log_prices.iloc[i]).sum() / len(w)

    risk = pd.Series(index=valid_idx, dtype=float)
    for i, date in enumerate(valid_idx):
        p = log_prices.loc[date]
        lo, hi = np.log(er['lower'].loc[date]), np.log(er['upper'].loc[date])
        if hi > lo:
            risk.loc[date] = np.clip((p - lo) / (hi - lo), 0, 1)

    phase = pd.Series("neutral", index=valid_idx)
    pvm = df["price"].loc[valid_idx] / eqm['50'].loc[valid_idx]
    for date in valid_idx:
        try:
            p, r = float(pvm.at[date]), float(risk.at[date])
        except (KeyError, ValueError):
            continue
        if pd.isna(p) or pd.isna(r):
            continue
        if r > 0.7:
            phase.loc[date] = "bull"
        elif r < 0.3 and p < 0.8:
            phase.loc[date] = "bear"

    # Smooth score and risk with 14-day EMA
    score_smooth = score.dropna().ewm(span=14, adjust=False).mean()
    score.update(score_smooth)
    risk_smooth = risk.dropna().ewm(span=14, adjust=False).mean()
    risk.update(risk_smooth)

    # ─── RISK MACD ─────────────────────────────────────────────────────────
    risk_clean = risk.dropna()
    macd_fast = risk_clean.ewm(span=50, adjust=False).mean()
    macd_slow = risk_clean.ewm(span=150, adjust=False).mean()
    macd_line = macd_fast - macd_slow
    macd_signal = macd_line.ewm(span=40, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    # ─── BUY/SELL SIGNALS from absolute Risk thresholds ────────────────────
    # Buy:  Risk drops to <= 10% (extreme undervaluation)
    # Sell: Risk rises to >= 90% (extreme overvaluation)
    # Simple, robust, minimal parameters — no overfitting risk
    signals = []
    last_signal_type = None
    for date in risk_clean.index:
        if date < sig_start_ts:
            continue
        r = float(risk_clean.loc[date])
        price_val = float(df["price"].loc[date])
        if r <= 0.10 and last_signal_type != 'buy':
            signals.append({'date': date, 'type': 'buy', 'price': price_val, 'risk': r})
            last_signal_type = 'buy'
        elif r >= 0.90 and last_signal_type != 'sell':
            signals.append({'date': date, 'type': 'sell', 'price': price_val, 'risk': r})
            last_signal_type = 'sell'

    return {'prices': df["price"], 'eqm': eqm, 'er': er, 'score': score, 'risk': risk,
            'phase': phase, 'signals': signals,
            'macd_line': macd_line, 'macd_signal': macd_signal, 'macd_hist': macd_hist,
            'beta_drift': beta_drift, 'cal_date': cal_date}

# ─── HTML GENERATION ─────────────────────────────────────────────────────────

def to_js_data(series, start):
    """Convert pandas Series to [[timestamp_ms, value], ...] for ECharts."""
    s = series.loc[start:].dropna()
    return [[int(d.timestamp() * 1000), round(float(v), 6)] for d, v in zip(s.index, s.values)]

def build_html(all_results):
    # Prepare data per asset
    assets_data = {}
    for name, r in all_results.items():
        cfg = ASSETS[name]
        start = cfg['display_start']
        p = r['prices']
        latest = p.iloc[-1]
        risk_val = r['risk'].dropna().iloc[-1] if len(r['risk'].dropna()) > 0 else 0
        score_val = r['score'].dropna().iloc[-1] if len(r['score'].dropna()) > 0 else 0

        # Phase blocks
        phase = r['phase'].loc[start:]
        blocks = []
        cur, bs = None, None
        for date, ph in phase.items():
            if ph != cur:
                if cur in ('bull', 'bear') and bs is not None:
                    blocks.append([int(bs.timestamp()*1000), int(date.timestamp()*1000), cur])
                cur, bs = ph, date
        if cur in ('bull', 'bear') and bs is not None:
            blocks.append([int(bs.timestamp()*1000), int(phase.index[-1].timestamp()*1000), cur])

        # Format signals for JS
        start_ts = pd.Timestamp(start)
        buy_signals_price = [[int(s['date'].timestamp()*1000), s['price']] for s in r['signals'] if s['type'] == 'buy' and s['date'] >= start_ts]
        sell_signals_price = [[int(s['date'].timestamp()*1000), s['price']] for s in r['signals'] if s['type'] == 'sell' and s['date'] >= start_ts]
        buy_signals_risk = [[int(s['date'].timestamp()*1000), s['risk']] for s in r['signals'] if s['type'] == 'buy' and s['date'] >= start_ts]
        sell_signals_risk = [[int(s['date'].timestamp()*1000), s['risk']] for s in r['signals'] if s['type'] == 'sell' and s['date'] >= start_ts]

        # MACD histogram
        macd_hist_data = to_js_data(r['macd_hist'], start)
        # Beta drift (convergence)
        beta_drift_data = to_js_data(r['beta_drift'], start)
        cal_date_ts = int(r['cal_date'].timestamp() * 1000) if r['cal_date'] else None

        assets_data[name] = {
            'price': to_js_data(p, start),
            'eqm_001': to_js_data(r['eqm']['001'], start),
            'eqm_50': to_js_data(r['eqm']['50'], start),
            'eqm_999': to_js_data(r['eqm']['999'], start),
            'er_lower': to_js_data(r['er']['lower'], start),
            'er_median': to_js_data(r['er']['median'], start),
            'er_upper': to_js_data(r['er']['upper'], start),
            'score': to_js_data(r['score'], start),
            'risk': to_js_data(r['risk'], start),
            'macd_line': to_js_data(r['macd_line'], start),
            'macd_signal': to_js_data(r['macd_signal'], start),
            'macd_hist': macd_hist_data,
            'beta_drift': beta_drift_data,
            'cal_date': cal_date_ts,
            'phases': blocks,
            'buy_price': buy_signals_price,
            'sell_price': sell_signals_price,
            'buy_risk': buy_signals_risk,
            'sell_risk': sell_signals_risk,
            'info': {
                'price': round(float(latest), 2),
                'er_lower': round(float(r['er']['lower'].iloc[-1]), 2),
                'er_median': round(float(r['er']['median'].iloc[-1]), 2),
                'er_upper': round(float(r['er']['upper'].iloc[-1]), 2),
                'risk': round(float(risk_val), 4),
                'score': round(float(score_val), 4),
                'date': str(p.index[-1].date()),
                'last_signal': r['signals'][-1]['type'] if r['signals'] else None,
                'last_signal_date': str(r['signals'][-1]['date'].date()) if r['signals'] else None,
                'last_signal_price': round(r['signals'][-1]['price'], 2) if r['signals'] else None,
                'total_buys': sum(1 for s in r['signals'] if s['type'] == 'buy'),
                'total_sells': sum(1 for s in r['signals'] if s['type'] == 'sell'),
            }
        }

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EQM Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0a0a0f; color: #e0e0e0; font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; }}
  .header {{ display: flex; align-items: center; justify-content: space-between; padding: 16px 24px 8px; }}
  .title {{ font-size: 18px; font-weight: 700; color: #fff; letter-spacing: 1px; }}
  .subtitle {{ font-size: 11px; color: #666; margin-top: 2px; }}
  .controls {{ display: flex; gap: 16px; align-items: center; }}
  .tabs {{ display: flex; gap: 4px; }}
  .tab, .range-btn {{ padding: 8px 20px; border: 1px solid #333; border-radius: 6px; background: #141420;
          color: #888; cursor: pointer; font-size: 13px; font-weight: 600; transition: all 0.2s; }}
  .tab:hover, .range-btn:hover {{ border-color: #555; color: #ccc; }}
  .tab.active {{ background: #1a1a2e; border-color: #4a9eff; color: #4a9eff; }}
  .range-btn.active {{ background: #1a1a2e; border-color: #666; color: #fff; }}
  .range-btns {{ display: flex; gap: 4px; }}
  .info-bar {{ display: flex; gap: 24px; padding: 8px 24px 4px; flex-wrap: wrap; }}
  .info-item {{ font-size: 12px; }}
  .info-label {{ color: #666; }}
  .info-value {{ color: #fff; font-weight: 600; }}
  .info-value.green {{ color: #00c853; }}
  .info-value.red {{ color: #ff1744; }}
  .info-value.orange {{ color: #ff9100; }}
  .info-value.yellow {{ color: #ffd600; }}
  #chart {{ width: 100%; height: calc(100vh - 100px); min-height: 600px; }}
  .footer {{ text-align: center; padding: 8px; font-size: 10px; color: #444; }}
</style>
</head>
<body>
<div class="header">
  <div>
    <div class="title">EQM Dashboard</div>
    <div class="subtitle">Empirical Quantile Model — Expectile Regression</div>
  </div>
  <div class="controls">
    <div class="tabs" id="tabs"></div>
    <div class="range-btns" id="range-btns"></div>
  </div>
</div>
<div class="info-bar" id="info-bar"></div>
<div id="chart"></div>
<div class="footer">Data: yfinance + CryptoCompare | Model: Expectile Regression (IRLS) | Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</div>

<script>
const DATA = {json.dumps(assets_data)};
const ASSETS = {json.dumps(list(assets_data.keys()))};
let currentAsset = ASSETS[0];
const chart = echarts.init(document.getElementById('chart'), null, {{renderer: 'canvas'}});

function fmt(v, dec) {{
  if (v >= 1e6) return '$' + (v/1e6).toFixed(1) + 'M';
  if (v >= 1e3) return '$' + (v/1e3).toFixed(dec > 0 ? 0 : 0) + 'K';
  if (v >= 1) return '$' + v.toFixed(2);
  return '$' + v.toFixed(4);
}}

function riskColor(r) {{
  if (r < 0.3) return '#00c853';
  if (r < 0.5) return '#aeea00';
  if (r < 0.7) return '#ff9100';
  return '#ff1744';
}}

function buildTabs() {{
  const el = document.getElementById('tabs');
  el.innerHTML = '';
  ASSETS.forEach(a => {{
    const tab = document.createElement('div');
    tab.className = 'tab' + (a === currentAsset ? ' active' : '');
    tab.textContent = a;
    tab.onclick = () => {{ currentAsset = a; buildTabs(); updateChart(); }};
    el.appendChild(tab);
  }});
}}

function updateInfoBar() {{
  const info = DATA[currentAsset].info;
  const rc = riskColor(info.risk);
  const el = document.getElementById('info-bar');
  el.innerHTML = `
    <div class="info-item"><span class="info-label">Price </span><span class="info-value">${{fmt(info.price)}}</span></div>
    <div class="info-item"><span class="info-label">ER Lower </span><span class="info-value green">${{fmt(info.er_lower)}}</span></div>
    <div class="info-item"><span class="info-label">ER Median </span><span class="info-value orange">${{fmt(info.er_median)}}</span></div>
    <div class="info-item"><span class="info-label">ER Upper </span><span class="info-value red">${{fmt(info.er_upper)}}</span></div>
    <div class="info-item"><span class="info-label">Risk </span><span class="info-value" style="color:${{rc}}">${{(info.risk*100).toFixed(1)}}%</span></div>
    <div class="info-item"><span class="info-label">Score </span><span class="info-value">${{(info.score*100).toFixed(1)}}%</span></div>
    <div class="info-item"><span class="info-label">Upside </span><span class="info-value green">+${{((info.er_upper/info.price - 1)*100).toFixed(0)}}%</span></div>
    <div class="info-item"><span class="info-label">Last Signal </span><span class="info-value" style="color:${{info.last_signal === 'buy' ? '#00e676' : '#ff1744'}}">${{info.last_signal ? info.last_signal.toUpperCase() + ' @ ' + fmt(info.last_signal_price) + ' (' + info.last_signal_date + ')' : '—'}}</span></div>
    <div class="info-item"><span class="info-label">${{info.date}}</span></div>
  `;
}}

function updateChart() {{
  updateInfoBar();
  const d = DATA[currentAsset];

  // Phase markAreas
  const phases = d.phases.map(p => ({{
    itemStyle: {{ color: p[2] === 'bull' ? 'rgba(0,200,83,0.06)' : 'rgba(255,23,68,0.06)' }},
    xAxis: p[0], yAxis: 0
  }})).reduce((acc, p, i, arr) => {{
    if (i % 1 === 0) {{
      const orig = d.phases[i];
      acc.push([
        {{ itemStyle: {{ color: orig[2] === 'bull' ? 'rgba(0,200,83,0.06)' : 'rgba(255,23,68,0.06)' }}, xAxis: orig[0] }},
        {{ xAxis: orig[1] }}
      ]);
    }}
    return acc;
  }}, []);

  // Risk colored segments
  const riskData = d.risk;
  const riskGreen = riskData.map(p => [p[0], p[1] <= 0.3 ? p[1] : null]);
  const riskYellow = riskData.map(p => [p[0], p[1] > 0.3 && p[1] <= 0.5 ? p[1] : null]);
  const riskOrange = riskData.map(p => [p[0], p[1] > 0.5 && p[1] <= 0.7 ? p[1] : null]);
  const riskRed = riskData.map(p => [p[0], p[1] > 0.7 ? p[1] : null]);

  const option = {{
    backgroundColor: '#0a0a0f',
    animation: true,
    animationDuration: 600,
    title: [
      {{ text: 'EQM Price Bands', subtext: 'Empirical quantiles (solid) & expectile regression bands (dashed) on log scale',
         left: 60, top: 4, textStyle: {{ color: '#ccc', fontSize: 13, fontWeight: 600 }},
         subtextStyle: {{ color: '#555', fontSize: 10 }} }},
      {{ text: 'EQM Risk', subtext: 'Position between lower/upper expectile bands (0–100%)',
         left: 60, top: '52%', textStyle: {{ color: '#ccc', fontSize: 12, fontWeight: 600 }},
         subtextStyle: {{ color: '#555', fontSize: 10 }} }},
      {{ text: 'Model Convergence', subtext: 'Rolling p95 coefficient drift — calibrated when < 0.05% (green line)',
         left: 60, top: '72%', textStyle: {{ color: '#ccc', fontSize: 12, fontWeight: 600 }},
         subtextStyle: {{ color: '#555', fontSize: 10 }} }},
      {{ text: 'EQM Score', subtext: 'Percentile rank in expanding historical distribution',
         left: 60, top: '88%', textStyle: {{ color: '#ccc', fontSize: 12, fontWeight: 600 }},
         subtextStyle: {{ color: '#555', fontSize: 10 }} }}
    ],
    tooltip: {{
      trigger: 'axis',
      backgroundColor: 'rgba(20,20,32,0.95)',
      borderColor: '#333',
      textStyle: {{ color: '#e0e0e0', fontFamily: 'monospace', fontSize: 11 }},
      axisPointer: {{ type: 'cross', lineStyle: {{ color: '#555' }} }},
      formatter: function(params) {{
        if (!params.length) return '';
        const date = new Date(params[0].value[0]).toISOString().slice(0,10);
        let lines = ['<b>' + date + '</b>'];
        params.forEach(p => {{
          if (p.value[1] !== null && p.value[1] !== undefined) {{
            const v = p.value[1];
            const formatted = p.componentIndex < 7 ? fmt(v) : (v*100).toFixed(1) + '%';
            lines.push(p.marker + ' ' + p.seriesName + ': <b>' + formatted + '</b>');
          }}
        }});
        return lines.join('<br>');
      }}
    }},
    axisPointer: {{ link: [{{ xAxisIndex: 'all' }}] }},
    grid: [
      {{ left: 60, right: 20, top: 50, height: '38%' }},
      {{ left: 60, right: 20, top: '54%', height: '12%' }},
      {{ left: 60, right: 20, top: '74%', height: '10%' }},
      {{ left: 60, right: 20, top: '90%', height: '6%' }}
    ],
    xAxis: [
      {{ type: 'time', gridIndex: 0, axisLabel: {{ show: false }}, axisLine: {{ lineStyle: {{ color: '#333' }} }}, splitLine: {{ show: false }} }},
      {{ type: 'time', gridIndex: 1, axisLabel: {{ show: false }}, axisLine: {{ lineStyle: {{ color: '#333' }} }}, splitLine: {{ show: false }} }},
      {{ type: 'time', gridIndex: 2, axisLabel: {{ show: false }}, axisLine: {{ lineStyle: {{ color: '#333' }} }}, splitLine: {{ show: false }} }},
      {{ type: 'time', gridIndex: 3, axisLine: {{ lineStyle: {{ color: '#333' }} }}, axisLabel: {{ color: '#666', fontSize: 10 }}, splitLine: {{ show: false }} }}
    ],
    yAxis: [
      {{ type: 'log', gridIndex: 0, axisLabel: {{ color: '#888', fontSize: 10, formatter: v => fmt(v) }},
         splitLine: {{ lineStyle: {{ color: '#1a1a2e' }} }}, axisLine: {{ lineStyle: {{ color: '#333' }} }} }},
      {{ type: 'value', gridIndex: 1, min: 0, max: 1, axisLabel: {{ color: '#888', fontSize: 10, formatter: v => (v*100)+'%' }},
         splitLine: {{ lineStyle: {{ color: '#1a1a2e' }} }}, axisLine: {{ lineStyle: {{ color: '#333' }} }} }},
      {{ type: 'log', gridIndex: 2, axisLabel: {{ color: '#888', fontSize: 10, formatter: v => (v*100).toFixed(v>=0.01?0:1)+'%' }},
         splitLine: {{ lineStyle: {{ color: '#1a1a2e' }} }}, axisLine: {{ lineStyle: {{ color: '#333' }} }} }},
      {{ type: 'value', gridIndex: 3, min: 0, max: 1, axisLabel: {{ color: '#888', fontSize: 10, formatter: v => (v*100)+'%' }},
         splitLine: {{ lineStyle: {{ color: '#1a1a2e' }} }}, axisLine: {{ lineStyle: {{ color: '#333' }} }} }}
    ],
    dataZoom: [
      {{ type: 'inside', xAxisIndex: [0,1,2,3], start: 0, end: 100 }},
      {{ type: 'slider', xAxisIndex: [0,1,2,3], bottom: 5, height: 20, borderColor: '#333',
         backgroundColor: '#141420', fillerColor: 'rgba(74,158,255,0.1)',
         handleStyle: {{ color: '#4a9eff' }}, textStyle: {{ color: '#888', fontSize: 10 }} }}
    ],
    series: [
      // Panel 1: Price bands
      {{ name: 'Price', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: d.price,
         lineStyle: {{ color: '#fff', width: 2 }}, symbol: 'none', z: 10,
         markArea: {{ silent: true, data: phases }} }},
      // Empirical quantiles (subtle, thin)
      {{ name: 'EQM 0.1%', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: d.eqm_001,
         lineStyle: {{ color: '#00c853', width: 0.8, opacity: 0.5 }}, symbol: 'none', z: 2 }},
      {{ name: 'EQM 50%', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: d.eqm_50,
         lineStyle: {{ color: '#ff9100', width: 0.8, opacity: 0.5 }}, symbol: 'none', z: 2 }},
      {{ name: 'EQM 99.9%', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: d.eqm_999,
         lineStyle: {{ color: '#ff1744', width: 0.8, opacity: 0.5 }}, symbol: 'none', z: 2 }},
      // ER bands (prominent, with area fill between lower-median and median-upper)
      {{ name: 'ER 0.1%', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: d.er_lower,
         lineStyle: {{ color: '#00c853', width: 1.5, type: 'dashed' }}, symbol: 'none', z: 5,
         areaStyle: {{ color: 'rgba(0,200,83,0.08)' }} }},
      {{ name: 'ER 50%', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: d.er_median,
         lineStyle: {{ color: '#ff9100', width: 1.5, type: 'dashed' }}, symbol: 'none', z: 5,
         areaStyle: {{ color: 'rgba(255,145,0,0.06)' }} }},
      {{ name: 'ER 99.9%', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: d.er_upper,
         lineStyle: {{ color: '#ff1744', width: 1.5, type: 'dashed' }}, symbol: 'none', z: 5,
         areaStyle: {{ color: {{ type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
           colorStops: [{{ offset: 0, color: 'rgba(255,23,68,0.08)' }}, {{ offset: 1, color: 'rgba(255,23,68,0)' }}] }} }} }},
      // Buy/Sell signals on price chart
      {{ name: 'Buy', type: 'scatter', xAxisIndex: 0, yAxisIndex: 0, data: d.buy_price,
         symbol: 'triangle', symbolSize: 14, z: 20,
         itemStyle: {{ color: '#00e676', borderColor: '#fff', borderWidth: 1 }},
         label: {{ show: true, position: 'bottom', formatter: 'B', fontSize: 10, fontWeight: 700, color: '#00e676' }} }},
      {{ name: 'Sell', type: 'scatter', xAxisIndex: 0, yAxisIndex: 0, data: d.sell_price,
         symbol: 'triangle', symbolSize: 14, symbolRotate: 180, z: 20,
         itemStyle: {{ color: '#ff1744', borderColor: '#fff', borderWidth: 1 }},
         label: {{ show: true, position: 'top', formatter: 'S', fontSize: 10, fontWeight: 700, color: '#ff1744' }} }},
      // Panel 2: Risk (colored) with absolute threshold zones
      {{ name: 'Risk', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: riskGreen,
         lineStyle: {{ color: '#00c853', width: 1.2 }}, symbol: 'none', connectNulls: false,
         markLine: {{ silent: true, lineStyle: {{ color: '#555', type: 'dashed', width: 1 }},
           data: [{{ yAxis: 0.1, label: {{ show: true, position: 'insideEndTop', formatter: 'BUY 10%', fontSize: 9, color: '#00c853' }} }},
                  {{ yAxis: 0.9, label: {{ show: true, position: 'insideEndTop', formatter: 'SELL 90%', fontSize: 9, color: '#ff1744' }} }}] }},
         markArea: {{ silent: true, data: [
           [{{ yAxis: 0, itemStyle: {{ color: 'rgba(0,200,83,0.08)' }} }}, {{ yAxis: 0.1 }}],
           [{{ yAxis: 0.9, itemStyle: {{ color: 'rgba(255,23,68,0.08)' }} }}, {{ yAxis: 1.0 }}]
         ] }} }},
      {{ name: 'Risk', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: riskYellow,
         lineStyle: {{ color: '#aeea00', width: 1.2 }}, symbol: 'none', connectNulls: false }},
      {{ name: 'Risk', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: riskOrange,
         lineStyle: {{ color: '#ff9100', width: 1.2 }}, symbol: 'none', connectNulls: false }},
      {{ name: 'Risk', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: riskRed,
         lineStyle: {{ color: '#ff1744', width: 1.2 }}, symbol: 'none', connectNulls: false }},
      // Buy/Sell signals on risk chart
      {{ name: 'Buy', type: 'scatter', xAxisIndex: 1, yAxisIndex: 1, data: d.buy_risk,
         symbol: 'triangle', symbolSize: 12, z: 20,
         itemStyle: {{ color: '#00e676', borderColor: '#fff', borderWidth: 1 }} }},
      {{ name: 'Sell', type: 'scatter', xAxisIndex: 1, yAxisIndex: 1, data: d.sell_risk,
         symbol: 'triangle', symbolSize: 12, symbolRotate: 180, z: 20,
         itemStyle: {{ color: '#ff1744', borderColor: '#fff', borderWidth: 1 }} }},
      // Panel 3: Model Convergence (beta drift)
      {{ name: 'Beta Drift', type: 'line', xAxisIndex: 2, yAxisIndex: 2, data: d.beta_drift,
         lineStyle: {{ width: 1.2 }}, symbol: 'none',
         itemStyle: {{ color: function(params) {{ return params.value[1] <= 0.01 ? '#00c853' : '#ff9100'; }} }},
         markLine: {{ silent: true, lineStyle: {{ color: '#00c853', type: 'dashed', width: 1 }},
           data: [{{ yAxis: 0.0005, label: {{ show: true, position: 'insideEndTop', formatter: '0.05% threshold', fontSize: 9, color: '#00c853' }} }}] }},
         markPoint: d.cal_date ? {{ data: [{{ coord: [d.cal_date, 0.0005], symbol: 'pin', symbolSize: 30,
           itemStyle: {{ color: '#00c853' }}, label: {{ show: true, formatter: 'Calibrated', fontSize: 9, color: '#fff' }} }}] }} : {{}} }},
      // Panel 4: Score
      {{ name: 'Score', type: 'line', xAxisIndex: 3, yAxisIndex: 3, data: d.score,
         lineStyle: {{ color: '#4a9eff', width: 1 }}, symbol: 'none',
         markLine: {{ silent: true, lineStyle: {{ color: '#333' }}, data: [{{ yAxis: 0.5 }}], label: {{ show: false }} }} }}
    ]
  }};
  chart.setOption(option, true);
}}

let currentRange = 'ALL';
const RANGES = ['6M', '1Y', '2Y', '5Y', 'ALL'];

function buildRangeBtns() {{
  const el = document.getElementById('range-btns');
  el.innerHTML = '';
  RANGES.forEach(r => {{
    const btn = document.createElement('div');
    btn.className = 'range-btn' + (r === currentRange ? ' active' : '');
    btn.textContent = r;
    btn.onclick = () => {{ currentRange = r; buildRangeBtns(); applyRange(); }};
    el.appendChild(btn);
  }});
}}

function applyRange() {{
  const d = DATA[currentAsset];
  const lastTs = d.price[d.price.length - 1][0];
  const lastDate = new Date(lastTs);
  let startDate;
  if (currentRange === 'ALL') {{
    chart.dispatchAction({{ type: 'dataZoom', start: 0, end: 100 }});
    return;
  }}
  const months = {{ '6M': 6, '1Y': 12, '2Y': 24, '5Y': 60 }}[currentRange];
  startDate = new Date(lastDate);
  startDate.setMonth(startDate.getMonth() - months);
  const startTs = startDate.getTime();
  const firstTs = d.price[0][0];
  const totalRange = lastTs - firstTs;
  const startPct = Math.max(0, ((startTs - firstTs) / totalRange) * 100);
  chart.dispatchAction({{ type: 'dataZoom', start: startPct, end: 100 }});
}}

buildTabs();
buildRangeBtns();
updateChart();
window.addEventListener('resize', () => chart.resize());
</script>
</body>
</html>"""
    return html

# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("EQM Dashboard Generator")
    print("=" * 50)
    all_results = {}
    for name, cfg in ASSETS.items():
        print(f"\n[{name}]")
        df = fetch_prices(name, cfg)
        if len(df) < cfg['min_window'] + 30:
            print(f"  SKIP: not enough data")
            continue
        result = compute_eqm(df, cfg['genesis'], cfg['min_window'], cfg['rolling_window'])
        all_results[name] = result
        p = result['prices'].iloc[-1]
        risk = result['risk'].dropna()
        score = result['score'].dropna()
        print(f"  Price: ${p:,.2f} | ER: ${result['er']['lower'].iloc[-1]:,.0f} / ${result['er']['median'].iloc[-1]:,.0f} / ${result['er']['upper'].iloc[-1]:,.0f}")
        if len(risk) > 0: print(f"  Risk: {risk.iloc[-1]:.1%} | Score: {score.iloc[-1]:.1%}")
        sigs = result['signals']
        if sigs:
            print(f"  Signals: {sum(1 for s in sigs if s['type']=='buy')} buys, {sum(1 for s in sigs if s['type']=='sell')} sells")
            for s in sigs:
                arrow = "BUY " if s['type'] == 'buy' else "SELL"
                print(f"    {arrow}  {s['date'].date()}  ${s['price']:>10,.2f}  Risk: {s['risk']:.1%}")
            print(f"  Last: {sigs[-1]['type'].upper()} @ ${sigs[-1]['price']:,.2f} ({sigs[-1]['date'].date()})")

    print(f"\nBuilding dashboard...")
    html = build_html(all_results)
    out = os.path.join(DIR, "eqm_dashboard.html")
    with open(out, 'w') as f:
        f.write(html)
    print(f"Saved to {out}")
