"""Check EQM Risk levels and send email alert if any asset enters high risk zone (>70%)."""
import json
import os
import smtplib
from email.mime.text import MIMEText

DASHBOARD_HTML = "eqm_dashboard.html"
RISK_THRESHOLD = 0.70
RECIPIENT = "christophpp@gmail.com"

def extract_risk_from_html():
    """Extract risk values from the embedded DATA JSON in the dashboard HTML."""
    with open(DASHBOARD_HTML, 'r') as f:
        html = f.read()
    # Find the DATA = {...} block
    start = html.index('const DATA = ') + len('const DATA = ')
    # Find matching closing brace by tracking depth
    depth = 0
    end = start
    for i, c in enumerate(html[start:], start):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    data = json.loads(html[start:end])

    results = {}
    for asset, d in data.items():
        info = d['info']
        results[asset] = {
            'risk': info['risk'],
            'price': info['price'],
            'er_upper': info['er_upper'],
            'score': info['score'],
            'date': info['date'],
        }
    return results


def send_alert(alerts):
    """Send email alert via GitHub Actions using sendmail or SMTP."""
    lines = ["EQM HIGH RISK ALERT", "=" * 40, ""]
    for asset, info in alerts.items():
        lines.append(f"{asset}: Risk {info['risk']*100:.1f}%")
        lines.append(f"  Price: ${info['price']:,.2f}")
        lines.append(f"  ER Upper Band: ${info['er_upper']:,.2f}")
        lines.append(f"  Score: {info['score']*100:.1f}%")
        lines.append(f"  Date: {info['date']}")
        lines.append("")
    lines.append("This means the asset is in the DISTRIBUTION zone.")
    lines.append("Consider reducing exposure or taking profits.")
    lines.append("")
    lines.append("Dashboard: https://cand1d.github.io/eqm-dashboard/")

    body = "\n".join(lines)
    asset_names = ", ".join(alerts.keys())

    # Write to GitHub Actions step summary
    summary_file = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary_file:
        with open(summary_file, 'a') as f:
            f.write(f"## EQM High Risk Alert\n\n")
            for asset, info in alerts.items():
                f.write(f"- **{asset}**: Risk {info['risk']*100:.1f}% | ${info['price']:,.2f}\n")

    # Send via sendmail (available on GitHub Actions ubuntu runners)
    msg = MIMEText(body)
    msg['Subject'] = f"EQM ALERT: {asset_names} in High Risk Zone (>{RISK_THRESHOLD*100:.0f}%)"
    msg['From'] = "eqm-dashboard@github.com"
    msg['To'] = RECIPIENT

    try:
        with smtplib.SMTP('localhost') as s:
            s.send_message(msg)
        print(f"Email sent to {RECIPIENT}")
    except Exception as e:
        print(f"sendmail failed: {e}")
        # Fallback: just print — the GitHub Action notification email from failure will alert
        print(f"\n{'='*50}")
        print(body)
        print(f"{'='*50}")
        # Exit with error to trigger GitHub notification
        if alerts:
            raise SystemExit(f"ALERT: {asset_names} in high risk zone!")


if __name__ == '__main__':
    results = extract_risk_from_html()

    print("EQM Risk Check")
    print("=" * 40)
    alerts = {}
    for asset, info in results.items():
        risk = info['risk']
        status = "HIGH RISK" if risk >= RISK_THRESHOLD else "OK"
        print(f"  {asset}: Risk {risk*100:.1f}% [{status}]")
        if risk >= RISK_THRESHOLD:
            alerts[asset] = info

    if alerts:
        print(f"\n{len(alerts)} asset(s) in high risk zone!")
        send_alert(alerts)
    else:
        print("\nAll assets below risk threshold. No alert needed.")
