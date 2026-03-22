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
        W = np.diag(w)
        XtW = X.T @ W
        beta_new = np.linalg.solve(XtW @ X, XtW @ y)
        if np.max(np.abs(beta_new - beta)) < tol:
            break
        beta = beta_new
    return beta

def compute_eqm(df, genesis_date, min_window=365, rolling_window=730):
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

    er = {}
    for label, tau in [('lower', 0.0001), ('median', 0.5), ('upper', 0.9999)]:
        beta = expectile_regression(X_all, log_prices.values, tau)
        er[label] = pd.Series(np.exp(X_v @ beta), index=valid_idx)

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

    return {'prices': df["price"], 'eqm': eqm, 'er': er, 'score': score, 'risk': risk, 'phase': phase}

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
            'phases': blocks,
            'info': {
                'price': round(float(latest), 2),
                'er_lower': round(float(r['er']['lower'].iloc[-1]), 2),
                'er_median': round(float(r['er']['median'].iloc[-1]), 2),
                'er_upper': round(float(r['er']['upper'].iloc[-1]), 2),
                'risk': round(float(risk_val), 4),
                'score': round(float(score_val), 4),
                'date': str(p.index[-1].date()),
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
  .tabs {{ display: flex; gap: 4px; }}
  .tab {{ padding: 8px 20px; border: 1px solid #333; border-radius: 6px; background: #141420;
          color: #888; cursor: pointer; font-size: 13px; font-weight: 600; transition: all 0.2s; }}
  .tab:hover {{ border-color: #555; color: #ccc; }}
  .tab.active {{ background: #1a1a2e; border-color: #4a9eff; color: #4a9eff; }}
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
  <div class="tabs" id="tabs"></div>
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
      {{ left: 60, right: 20, top: 30, height: '50%' }},
      {{ left: 60, right: 20, top: '63%', height: '14%' }},
      {{ left: 60, right: 20, top: '82%', height: '14%' }}
    ],
    xAxis: [
      {{ type: 'time', gridIndex: 0, axisLabel: {{ show: false }}, axisLine: {{ lineStyle: {{ color: '#333' }} }}, splitLine: {{ show: false }} }},
      {{ type: 'time', gridIndex: 1, axisLabel: {{ show: false }}, axisLine: {{ lineStyle: {{ color: '#333' }} }}, splitLine: {{ show: false }} }},
      {{ type: 'time', gridIndex: 2, axisLine: {{ lineStyle: {{ color: '#333' }} }}, axisLabel: {{ color: '#666', fontSize: 10 }}, splitLine: {{ show: false }} }}
    ],
    yAxis: [
      {{ type: 'log', gridIndex: 0, axisLabel: {{ color: '#888', fontSize: 10, formatter: v => fmt(v) }},
         splitLine: {{ lineStyle: {{ color: '#1a1a2e' }} }}, axisLine: {{ lineStyle: {{ color: '#333' }} }} }},
      {{ type: 'value', gridIndex: 1, min: 0, max: 1, axisLabel: {{ color: '#888', fontSize: 10, formatter: v => (v*100)+'%' }},
         splitLine: {{ lineStyle: {{ color: '#1a1a2e' }} }}, axisLine: {{ lineStyle: {{ color: '#333' }} }} }},
      {{ type: 'value', gridIndex: 2, min: 0, max: 1, axisLabel: {{ color: '#888', fontSize: 10, formatter: v => (v*100)+'%' }},
         splitLine: {{ lineStyle: {{ color: '#1a1a2e' }} }}, axisLine: {{ lineStyle: {{ color: '#333' }} }} }}
    ],
    dataZoom: [
      {{ type: 'inside', xAxisIndex: [0,1,2], start: 0, end: 100 }},
      {{ type: 'slider', xAxisIndex: [0,1,2], bottom: 5, height: 20, borderColor: '#333',
         backgroundColor: '#141420', fillerColor: 'rgba(74,158,255,0.1)',
         handleStyle: {{ color: '#4a9eff' }}, textStyle: {{ color: '#888', fontSize: 10 }} }}
    ],
    series: [
      // Panel 1: Price bands
      {{ name: 'Price', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: d.price,
         lineStyle: {{ color: '#fff', width: 1.5 }}, symbol: 'none', z: 10,
         markArea: {{ silent: true, data: phases }} }},
      {{ name: 'EQM 0.1%', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: d.eqm_001,
         lineStyle: {{ color: '#00c853', width: 1.2 }}, symbol: 'none' }},
      {{ name: 'EQM 50%', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: d.eqm_50,
         lineStyle: {{ color: '#ff9100', width: 1.2 }}, symbol: 'none' }},
      {{ name: 'EQM 99.9%', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: d.eqm_999,
         lineStyle: {{ color: '#ff1744', width: 1.2 }}, symbol: 'none' }},
      {{ name: 'ER 0.1%', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: d.er_lower,
         lineStyle: {{ color: '#00c853', width: 1, type: 'dashed' }}, symbol: 'none' }},
      {{ name: 'ER 50%', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: d.er_median,
         lineStyle: {{ color: '#ff9100', width: 1, type: 'dashed' }}, symbol: 'none' }},
      {{ name: 'ER 99.9%', type: 'line', xAxisIndex: 0, yAxisIndex: 0, data: d.er_upper,
         lineStyle: {{ color: '#ff1744', width: 1, type: 'dashed' }}, symbol: 'none' }},
      // Panel 2: Score
      {{ name: 'Score', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: d.score,
         lineStyle: {{ color: '#4a9eff', width: 1 }}, symbol: 'none',
         markLine: {{ silent: true, lineStyle: {{ color: '#333' }}, data: [{{ yAxis: 0.5 }}], label: {{ show: false }} }} }},
      // Panel 3: Risk (colored)
      {{ name: 'Risk', type: 'line', xAxisIndex: 2, yAxisIndex: 2, data: riskGreen,
         lineStyle: {{ color: '#00c853', width: 1.2 }}, symbol: 'none', connectNulls: false,
         markLine: {{ silent: true, lineStyle: {{ color: '#333' }}, data: [{{ yAxis: 0.3 }}, {{ yAxis: 0.7 }}], label: {{ show: false }} }} }},
      {{ name: 'Risk', type: 'line', xAxisIndex: 2, yAxisIndex: 2, data: riskYellow,
         lineStyle: {{ color: '#aeea00', width: 1.2 }}, symbol: 'none', connectNulls: false }},
      {{ name: 'Risk', type: 'line', xAxisIndex: 2, yAxisIndex: 2, data: riskOrange,
         lineStyle: {{ color: '#ff9100', width: 1.2 }}, symbol: 'none', connectNulls: false }},
      {{ name: 'Risk', type: 'line', xAxisIndex: 2, yAxisIndex: 2, data: riskRed,
         lineStyle: {{ color: '#ff1744', width: 1.2 }}, symbol: 'none', connectNulls: false }}
    ]
  }};
  chart.setOption(option, true);
}}

buildTabs();
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

    print(f"\nBuilding dashboard...")
    html = build_html(all_results)
    out = os.path.join(DIR, "eqm_dashboard.html")
    with open(out, 'w') as f:
        f.write(html)
    print(f"Saved to {out}")
