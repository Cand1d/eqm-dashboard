"""EQM Dashboard — Interactive Plotly visualization for BTC, SOL, SUI."""
import pandas as pd
import numpy as np
import requests
import os
import warnings
warnings.filterwarnings('ignore')

DIR = os.path.dirname(os.path.abspath(__file__))

# ─── ASSET CONFIG ────────────────────────────────────────────────────────────

ASSETS = {
    'BTC': {
        'symbol': 'BTC-USD',
        'genesis': pd.Timestamp('2009-01-03'),
        'min_window': 365,
        'rolling_window': 730,
        'display_start': '2014-01-01',
        'cc_early': True,
        'cc_to_ts': 1410912000,  # 2014-09-17
    },
    'SOL': {
        'symbol': 'SOL-USD',
        'genesis': pd.Timestamp('2020-03-16'),
        'min_window': 180,
        'rolling_window': 365,
        'display_start': '2021-01-01',
        'cc_early': False,
    },
    'SUI': {
        'symbol': 'SUI20947-USD',
        'genesis': pd.Timestamp('2023-05-03'),
        'min_window': 90,
        'rolling_window': 180,
        'display_start': '2023-09-01',
        'cc_early': False,
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

    # Early data from CryptoCompare if needed
    if cfg.get('cc_early'):
        print(f"  {name}: fetching early data from CryptoCompare...")
        cc_url = 'https://min-api.cryptocompare.com/data/v2/histoday'
        cc_params = {'fsym': name, 'tsym': 'USD', 'limit': 2000, 'toTs': cfg['cc_to_ts']}
        cc_resp = requests.get(cc_url, params=cc_params, timeout=30)
        cc_data = cc_resp.json()['Data']['Data']
        early = pd.DataFrame([
            {'date': pd.Timestamp.utcfromtimestamp(p['time']), 'price': p['close']}
            for p in cc_data if p['close'] > 0
        ]).set_index('date')
        early.index = early.index.tz_localize(None)
        frames.append(early)

    # yfinance data
    print(f"  {name}: fetching from yfinance ({cfg['symbol']})...")
    btc = yf.download(cfg['symbol'], start="2010-01-01", progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    recent = pd.DataFrame({"price": btc["Close"].values}, index=btc.index)
    recent.index.name = "date"
    recent.index = recent.index.tz_localize(None)

    if frames:
        early = frames[0]
        df = pd.concat([early[early.index < recent.index[0]], recent]).dropna()
    else:
        df = recent.dropna()

    df.to_csv(cache)
    print(f"  {name}: {len(df)} prices from {df.index[0].date()} to {df.index[-1].date()}")
    return df

# ─── MODEL COMPUTATION ──────────────────────────────────────────────────────

def expectile_regression(X, y, tau, max_iter=200, tol=1e-6):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(max_iter):
        residuals = y - X @ beta
        weights = np.where(residuals >= 0, tau, 1 - tau)
        W = np.diag(weights)
        XtW = X.T @ W
        beta_new = np.linalg.solve(XtW @ X, XtW @ y)
        if np.max(np.abs(beta_new - beta)) < tol:
            break
        beta = beta_new
    return beta


def compute_eqm(df, genesis_date, min_window=365, rolling_window=730):
    log_prices = np.log(df["price"])

    # Empirical quantiles (rolling window)
    eqm_001 = pd.Series(index=df.index, dtype=float)
    eqm_50 = pd.Series(index=df.index, dtype=float)
    eqm_999 = pd.Series(index=df.index, dtype=float)
    for i in range(min_window, len(df)):
        start = max(0, i + 1 - rolling_window)
        window = log_prices.iloc[start:i+1]
        eqm_001.iloc[i] = np.quantile(window, 0.001)
        eqm_50.iloc[i] = np.quantile(window, 0.5)
        eqm_999.iloc[i] = np.quantile(window, 0.999)
    eqm_001 = np.exp(eqm_001)
    eqm_50 = np.exp(eqm_50)
    eqm_999 = np.exp(eqm_999)

    # Expectile regression (IRLS, log-log quadratic, genesis block axis)
    valid_idx = df.index[min_window:]
    days_since_genesis = (df.index - genesis_date).days.values.astype(float) + 1  # +1 avoids log(0)
    log_days = np.log(days_since_genesis)
    y_all = log_prices.values
    X_all = np.column_stack([np.ones(len(df)), log_days, log_days**2])
    X_valid = X_all[min_window:]

    taus = {'lower': 0.0001, 'median': 0.5, 'upper': 0.9999}
    er = {}
    for label, tau in taus.items():
        beta = expectile_regression(X_all, y_all, tau)
        er[label] = pd.Series(np.exp(X_valid @ beta), index=valid_idx)

    # EQM Score (expanding window percentile rank)
    eqm_score = pd.Series(index=df.index, dtype=float)
    for i in range(min_window, len(df)):
        window = log_prices.iloc[:i+1]
        eqm_score.iloc[i] = (window < log_prices.iloc[i]).sum() / len(window)

    # EQM Risk
    eqm_risk = pd.Series(index=valid_idx, dtype=float)
    for i, date in enumerate(valid_idx):
        price = log_prices.loc[date]
        lower = np.log(er['lower'].loc[date])
        upper = np.log(er['upper'].loc[date])
        if upper > lower:
            eqm_risk.loc[date] = np.clip((price - lower) / (upper - lower), 0, 1)

    # Phases
    phase = pd.Series("neutral", index=valid_idx)
    price_vs_median = df["price"].loc[valid_idx] / eqm_50.loc[valid_idx]
    for date in valid_idx:
        try:
            pvm = float(price_vs_median.at[date])
            r = float(eqm_risk.at[date])
        except (KeyError, ValueError):
            continue
        if pd.isna(pvm) or pd.isna(r):
            continue
        if r > 0.7:
            phase.loc[date] = "bull"
        elif r < 0.3 and pvm < 0.8:
            phase.loc[date] = "bear"

    return {
        'prices': df["price"],
        'eqm_001': eqm_001, 'eqm_50': eqm_50, 'eqm_999': eqm_999,
        'er_lower': er['lower'], 'er_median': er['median'], 'er_upper': er['upper'],
        'score': eqm_score, 'risk': eqm_risk, 'phase': phase,
    }

# ─── PLOTLY DASHBOARD ────────────────────────────────────────────────────────

def build_dashboard(all_results):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.04,
        subplot_titles=['EQM Price Bands', 'EQM Score', 'EQM Risk']
    )

    asset_names = list(all_results.keys())
    trace_groups = {}
    trace_idx = 0

    for asset_name in asset_names:
        r = all_results[asset_name]
        cfg = ASSETS[asset_name]
        start = cfg['display_start']
        visible = (asset_name == asset_names[0])
        group_start = trace_idx

        def filt(s):
            return s.loc[start:].dropna()

        prices = filt(r['prices'])

        # Row 1: Price + Bands
        fig.add_trace(go.Scatter(
            x=prices.index, y=prices.values, name='Price',
            line=dict(color='black', width=1.2), visible=visible,
            hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>Price</extra>'
        ), row=1, col=1)
        trace_idx += 1

        # Empirical bands
        for series, color, name_lbl in [
            (r['eqm_001'], 'green', 'EQM 0.1%'),
            (r['eqm_50'], 'orange', 'EQM 50%'),
            (r['eqm_999'], 'red', 'EQM 99.9%'),
        ]:
            s = filt(series)
            fig.add_trace(go.Scatter(
                x=s.index, y=s.values, name=name_lbl,
                line=dict(color=color, width=1.5), visible=visible,
                hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>' + name_lbl + '</extra>'
            ), row=1, col=1)
            trace_idx += 1

        # Expectile bands (dashed)
        for series, color, name_lbl in [
            (r['er_lower'], 'green', 'ER 0.1%'),
            (r['er_median'], 'orange', 'ER 50%'),
            (r['er_upper'], 'red', 'ER 99.9%'),
        ]:
            s = filt(series)
            fig.add_trace(go.Scatter(
                x=s.index, y=s.values, name=name_lbl,
                line=dict(color=color, width=1, dash='dash'), visible=visible,
                hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>' + name_lbl + '</extra>'
            ), row=1, col=1)
            trace_idx += 1

        # Phase shading (merge consecutive days into blocks)
        phase = r['phase'].loc[start:]
        phase_blocks = []
        current_phase = None
        block_start = None
        for date, p in phase.items():
            if p != current_phase:
                if current_phase in ('bull', 'bear') and block_start is not None:
                    phase_blocks.append((block_start, date, current_phase))
                current_phase = p
                block_start = date
        if current_phase in ('bull', 'bear') and block_start is not None:
            phase_blocks.append((block_start, phase.index[-1], current_phase))

        for bstart, bend, bphase in phase_blocks:
            fig.add_vrect(
                x0=bstart, x1=bend, row=1, col=1,
                fillcolor='green' if bphase == 'bull' else 'red',
                opacity=0.07, line_width=0,
                visible=visible
            )

        # Row 2: Score
        score = filt(r['score'])
        fig.add_trace(go.Scatter(
            x=score.index, y=score.values, name='Score',
            line=dict(color='black', width=0.8), visible=visible,
            hovertemplate='%{x|%Y-%m-%d}<br>Score: %{y:.1%}<extra></extra>'
        ), row=2, col=1)
        trace_idx += 1

        # Row 3: Risk (colored segments)
        risk = filt(r['risk'])
        color_map = [
            (0.3, 'green'), (0.5, 'yellowgreen'), (0.7, 'orange'), (1.01, 'red')
        ]
        for threshold, color in color_map:
            mask = risk <= threshold
            if threshold > 0.3:
                prev = color_map[color_map.index((threshold, color)) - 1][0]
                mask = (risk > prev) & (risk <= threshold)
            seg = risk.where(mask)
            fig.add_trace(go.Scatter(
                x=seg.index, y=seg.values, name=f'Risk',
                line=dict(color=color, width=1), visible=visible,
                showlegend=False, connectgaps=False,
                hovertemplate='%{x|%Y-%m-%d}<br>Risk: %{y:.1%}<extra></extra>'
            ), row=3, col=1)
            trace_idx += 1

        trace_groups[asset_name] = (group_start, trace_idx)

    # Build tab buttons
    total_traces = trace_idx
    # Count shapes per asset for vrect visibility
    shape_counts = {}
    shape_offset = 0
    for asset_name in asset_names:
        phase = all_results[asset_name]['phase'].loc[ASSETS[asset_name]['display_start']:]
        n_shapes = 0
        current_phase = None
        for _, p in phase.items():
            if p != current_phase:
                if current_phase in ('bull', 'bear'):
                    n_shapes += 1
                current_phase = p
        if current_phase in ('bull', 'bear'):
            n_shapes += 1
        shape_counts[asset_name] = (shape_offset, shape_offset + n_shapes)
        shape_offset += n_shapes

    buttons = []
    for asset_name in asset_names:
        gstart, gend = trace_groups[asset_name]
        visible = [False] * total_traces
        for i in range(gstart, gend):
            visible[i] = True

        # Info text
        r = all_results[asset_name]
        latest_price = r['prices'].iloc[-1]
        risk_val = r['risk'].dropna().iloc[-1] if len(r['risk'].dropna()) > 0 else 0
        score_val = r['score'].dropna().iloc[-1] if len(r['score'].dropna()) > 0 else 0

        cfg = ASSETS[asset_name]

        buttons.append(dict(
            label=asset_name,
            method='update',
            args=[
                {'visible': visible},
                {
                    'xaxis.range': [cfg['display_start'], str(r['prices'].index[-1].date())],
                    'xaxis2.range': [cfg['display_start'], str(r['prices'].index[-1].date())],
                    'xaxis3.range': [cfg['display_start'], str(r['prices'].index[-1].date())],
                    'annotations': [{
                        'x': 0.99, 'y': 0.01, 'xref': 'paper', 'yref': 'y',
                        'text': (
                            f"<b>{asset_name}</b> ({r['prices'].index[-1].strftime('%Y-%m-%d')})<br>"
                            f"Price: ${latest_price:,.0f}<br>"
                            f"ER Lower: ${r['er_lower'].iloc[-1]:,.0f}<br>"
                            f"ER Median: ${r['er_median'].iloc[-1]:,.0f}<br>"
                            f"ER Upper: ${r['er_upper'].iloc[-1]:,.0f}<br>"
                            f"Risk: {risk_val:.1%} | Score: {score_val:.1%}"
                        ),
                        'showarrow': False, 'font': dict(size=11, family='monospace'),
                        'bgcolor': 'rgba(255,255,255,0.9)', 'bordercolor': 'gray',
                        'xanchor': 'right', 'yanchor': 'bottom',
                    }]
                }
            ]
        ))

    fig.update_layout(
        updatemenus=[dict(
            type='buttons', direction='right',
            x=0.0, y=1.06, xanchor='left',
            buttons=buttons,
            bgcolor='white', bordercolor='gray',
            font=dict(size=13),
        )],
        title=dict(text='Empirical Quantile Model (EQM) Dashboard', font=dict(size=18)),
        height=900, template='plotly_white',
        showlegend=True,
        legend=dict(orientation='h', y=1.02, x=0.3, font=dict(size=9)),
        hovermode='x unified',
    )

    # Axis formatting
    fig.update_yaxes(type='log', row=1, col=1, tickprefix='$',
                     tickformat=',.0f', title='Price (USD)')
    fig.update_yaxes(range=[-0.05, 1.05], row=2, col=1, title='Score',
                     tickformat='.0%')
    fig.update_yaxes(range=[-0.05, 1.05], row=3, col=1, title='Risk',
                     tickformat='.0%')

    # Reference lines
    fig.add_hline(y=0.5, row=2, col=1, line_dash='dash', line_color='gray', opacity=0.4)
    fig.add_hline(y=0.3, row=3, col=1, line_dash='dash', line_color='green', opacity=0.3)
    fig.add_hline(y=0.7, row=3, col=1, line_dash='dash', line_color='red', opacity=0.3)

    # Initial annotation (first asset)
    first = asset_names[0]
    r = all_results[first]
    latest_price = r['prices'].iloc[-1]
    risk_val = r['risk'].dropna().iloc[-1] if len(r['risk'].dropna()) > 0 else 0
    score_val = r['score'].dropna().iloc[-1] if len(r['score'].dropna()) > 0 else 0
    fig.add_annotation(
        x=0.99, y=0.01, xref='paper', yref='y',
        text=(
            f"<b>{first}</b> ({r['prices'].index[-1].strftime('%Y-%m-%d')})<br>"
            f"Price: ${latest_price:,.0f}<br>"
            f"ER Lower: ${r['er_lower'].iloc[-1]:,.0f}<br>"
            f"ER Median: ${r['er_median'].iloc[-1]:,.0f}<br>"
            f"ER Upper: ${r['er_upper'].iloc[-1]:,.0f}<br>"
            f"Risk: {risk_val:.1%} | Score: {score_val:.1%}"
        ),
        showarrow=False, font=dict(size=11, family='monospace'),
        bgcolor='rgba(255,255,255,0.9)', bordercolor='gray',
        xanchor='right', yanchor='bottom',
    )

    return fig


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("EQM Dashboard Generator")
    print("=" * 50)

    all_results = {}
    for name, cfg in ASSETS.items():
        print(f"\n[{name}]")
        df = fetch_prices(name, cfg)
        if len(df) < cfg['min_window'] + 30:
            print(f"  SKIP: not enough data ({len(df)} < {cfg['min_window'] + 30})")
            continue
        result = compute_eqm(df, cfg['genesis'], cfg['min_window'], cfg['rolling_window'])
        all_results[name] = result

        # Print summary
        p = result['prices'].iloc[-1]
        risk = result['risk'].dropna()
        score = result['score'].dropna()
        print(f"  Price: ${p:,.0f}")
        print(f"  ER Lower: ${result['er_lower'].iloc[-1]:,.0f}")
        print(f"  ER Median: ${result['er_median'].iloc[-1]:,.0f}")
        print(f"  ER Upper: ${result['er_upper'].iloc[-1]:,.0f}")
        if len(risk) > 0:
            print(f"  Risk: {risk.iloc[-1]:.1%}")
        if len(score) > 0:
            print(f"  Score: {score.iloc[-1]:.1%}")

    print(f"\nBuilding dashboard...")
    fig = build_dashboard(all_results)

    out_path = os.path.join(DIR, "eqm_dashboard.html")
    fig.write_html(out_path, include_plotlyjs='cdn', full_html=True,
                   config={'displayModeBar': True, 'scrollZoom': True})
    print(f"Saved to {out_path}")

    # Print final summary
    print(f"\n{'=' * 50}")
    for name, r in all_results.items():
        p = r['prices'].iloc[-1]
        risk = r['risk'].dropna()
        print(f"{name}: ${p:,.0f} | Risk: {risk.iloc[-1]:.1%}" if len(risk) > 0 else f"{name}: ${p:,.0f}")
