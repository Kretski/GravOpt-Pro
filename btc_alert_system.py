"""
ORAC-NT Structural Alert System — BTC/USDT
==========================================
Author : Dimitar Kretski
DOI    : 10.5281/zenodo.20315517

Real-time structural health monitor for BTC market.
NOT a price predictor. A structural state detector.

Alerts when:
  - SR drops below fluid threshold (market unusually calm)
  - SR rises above rigid threshold (market under stress)
  - Composite signal changes regime
  - Volume anomaly detected
  - Kurtosis spike (fat tail event)

Usage:
    pip install ccxt pandas numpy matplotlib scipy plyer
    python btc_alert_system.py

Runs every 1 hour. Sends desktop notifications.
"""

import ccxt
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timezone
from scipy.stats import kurtosis
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

SYMBOL           = 'BTC/USDT'
TIMEFRAME        = '1d'
WINDOW           = 30
COARSE_THRESHOLD = 0.03
FINE_THRESHOLD   = 0.005
CHECK_INTERVAL   = 3600   # seconds (1 hour)
LOG_FILE         = 'structural_alerts.log'
STATE_FILE       = 'market_state.json'

# Alert thresholds
SR_FLUID_THRESHOLD  = 0.05   # SR below this = unusually fluid
SR_RIGID_THRESHOLD  = 0.45   # SR above this = high rigidity
KURT_THRESHOLD      = 7.0    # kurtosis above this = fat tail event
VOLUME_Z_THRESHOLD  = 2.5    # volume z-score above this = anomaly
COMPOSITE_HIGH      = 0.70   # composite above this = stress

# ─────────────────────────────────────────────────────────────
# Desktop notifications
# ─────────────────────────────────────────────────────────────

def notify(title, message, urgency='normal'):
    """Send desktop notification."""
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_name='ORAC-NT Structural Monitor',
            timeout=15
        )
    except Exception:
        pass
    # Always print to console
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    print(f"\n{'='*60}")
    print(f"  [{urgency.upper()}] {title}")
    print(f"  {message}")
    print(f"  {ts}")
    print(f"{'='*60}")

# ─────────────────────────────────────────────────────────────
# Data fetch
# ─────────────────────────────────────────────────────────────

def fetch_data(days=60):
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.milliseconds() - days * 24 * 3600 * 1000
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=days+5)
    df = pd.DataFrame(ohlcv,
                      columns=['timestamp','open','high','low','close','volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('date').drop(columns=['timestamp'])
    return df

# ─────────────────────────────────────────────────────────────
# Compute sensors
# ─────────────────────────────────────────────────────────────

def compute_sensors(df):
    df = df.copy()
    df['returns'] = df['close'].pct_change()

    df['p_coarse'] = df['returns'].abs().rolling(WINDOW).apply(
        lambda x: np.mean(np.abs(x) > COARSE_THRESHOLD), raw=True)
    df['p_fine'] = df['returns'].abs().rolling(WINDOW).apply(
        lambda x: np.mean(np.abs(x) > FINE_THRESHOLD), raw=True)
    df['SR'] = (df['p_coarse'].clip(lower=1e-3) /
                df['p_fine'].clip(lower=1e-3))

    df['volatility'] = df['returns'].rolling(WINDOW).std() * np.sqrt(365)

    df['kurtosis_val'] = df['returns'].rolling(WINDOW).apply(
        lambda x: kurtosis(x, fisher=False, bias=False)
        if len(x) > 4 else 3.0, raw=True)

    df['volume_z'] = df['volume'].rolling(WINDOW).apply(
        lambda x: (x[-1] - np.mean(x)) / (np.std(x) + 1e-10), raw=True)

    vol_z_norm = (df['volume_z'] - df['volume_z'].mean()) / \
                 (df['volume_z'].std() + 1e-10)
    df['composite'] = df['SR'] * (1 + 0.3 * vol_z_norm.clip(lower=0))

    return df.dropna()

# ─────────────────────────────────────────────────────────────
# Regime classification
# ─────────────────────────────────────────────────────────────

def classify_regime(sr, composite, kurt, volume_z):
    if sr > SR_RIGID_THRESHOLD and composite > COMPOSITE_HIGH:
        return "STRESSED_RIGID"
    elif sr < SR_FLUID_THRESHOLD:
        return "ULTRA_FLUID"
    elif kurt > KURT_THRESHOLD:
        return "FAT_TAIL_EVENT"
    elif volume_z > VOLUME_Z_THRESHOLD:
        return "VOLUME_ANOMALY"
    elif sr > SR_RIGID_THRESHOLD:
        return "RIGID"
    elif sr < 0.15:
        return "FLUID"
    else:
        return "NORMAL"

REGIME_DESCRIPTIONS = {
    "STRESSED_RIGID":  "High SR + High Composite. Market under structural stress.",
    "ULTRA_FLUID":     "SR near zero. Historically precedes volatility expansion.",
    "FAT_TAIL_EVENT":  "Kurtosis spike. Extreme move distribution detected.",
    "VOLUME_ANOMALY":  "Volume 2.5+ std above average. Unusual activity.",
    "RIGID":           "SR above rigid threshold. Large moves dominating.",
    "FLUID":           "SR below fluid threshold. Small moves dominating.",
    "NORMAL":          "All sensors within normal range.",
}

# ─────────────────────────────────────────────────────────────
# Alert logic
# ─────────────────────────────────────────────────────────────

def check_alerts(current, previous_regime):
    sr        = current['SR']
    composite = current['composite']
    kurt      = current['kurtosis_val']
    volume_z  = current['volume_z']
    price     = current['close']

    regime = classify_regime(sr, composite, kurt, volume_z)
    alerts = []

    # Regime change alert
    if regime != previous_regime:
        alerts.append({
            'level':   'REGIME_CHANGE',
            'title':   f'Market regime: {regime}',
            'message': f'{REGIME_DESCRIPTIONS[regime]}\n'
                       f'SR={sr:.4f} | Composite={composite:.3f} | '
                       f'Kurt={kurt:.1f} | BTC=${price:,.0f}',
            'urgency': 'critical' if regime in
                       ['STRESSED_RIGID','FAT_TAIL_EVENT'] else 'normal'
        })

    # Threshold alerts
    if sr < SR_FLUID_THRESHOLD:
        alerts.append({
            'level':   'ULTRA_FLUID',
            'title':   'SR at historic low',
            'message': f'SR={sr:.4f} (threshold={SR_FLUID_THRESHOLD})\n'
                       f'Market unusually fluid. Monitor for expansion.\n'
                       f'BTC=${price:,.0f}',
            'urgency': 'normal'
        })

    if sr > SR_RIGID_THRESHOLD:
        alerts.append({
            'level':   'HIGH_RIGIDITY',
            'title':   'High structural rigidity detected',
            'message': f'SR={sr:.4f} (threshold={SR_RIGID_THRESHOLD})\n'
                       f'Large moves dominating. Execution risk elevated.\n'
                       f'BTC=${price:,.0f}',
            'urgency': 'normal'
        })

    if kurt > KURT_THRESHOLD:
        alerts.append({
            'level':   'FAT_TAIL',
            'title':   'Kurtosis spike detected',
            'message': f'Kurtosis={kurt:.1f} (threshold={KURT_THRESHOLD})\n'
                       f'Extreme move distribution. Fat tail event.\n'
                       f'BTC=${price:,.0f}',
            'urgency': 'critical'
        })

    if volume_z > VOLUME_Z_THRESHOLD:
        alerts.append({
            'level':   'VOLUME_SPIKE',
            'title':   'Volume anomaly detected',
            'message': f'Volume Z={volume_z:.2f} (threshold={VOLUME_Z_THRESHOLD})\n'
                       f'Unusual trading activity.\nBTC=${price:,.0f}',
            'urgency': 'normal'
        })

    return regime, alerts

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

def log_state(state):
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    line = (f"[{ts}] "
            f"BTC=${state['close']:,.0f} | "
            f"SR={state['SR']:.4f} | "
            f"Composite={state['composite']:.3f} | "
            f"Kurt={state['kurtosis_val']:.2f} | "
            f"VolZ={state['volume_z']:.2f} | "
            f"Regime={state['regime']}")
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')
    print(line)

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float))
                   else str(v) for k, v in state.items()}, f, indent=2)

def load_previous_regime():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            return data.get('regime', 'NORMAL')
        except Exception:
            pass
    return 'NORMAL'

# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────

def run_once():
    """Single check — useful for testing."""
    print(f"\n{'='*60}")
    print(f"  ORAC-NT STRUCTURAL MONITOR — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")

    df     = fetch_data(days=60)
    df     = compute_sensors(df)
    latest = df.iloc[-1].to_dict()

    prev_regime      = load_previous_regime()
    regime, alerts   = check_alerts(latest, prev_regime)
    latest['regime'] = regime

    log_state(latest)
    save_state(latest)

    if alerts:
        for alert in alerts:
            notify(alert['title'], alert['message'], alert['urgency'])
    else:
        print(f"  No alerts. Regime: {regime}")

    print(f"\n  Current state:")
    print(f"    BTC Price  : ${latest['close']:,.0f}")
    print(f"    SR         : {latest['SR']:.4f}")
    print(f"    Composite  : {latest['composite']:.3f}")
    print(f"    Kurtosis   : {latest['kurtosis_val']:.2f}")
    print(f"    Volume Z   : {latest['volume_z']:.2f}")
    print(f"    Regime     : {regime}")
    print(f"    Description: {REGIME_DESCRIPTIONS[regime]}")

    return regime, alerts

def run_continuous():
    """Continuous monitoring loop."""
    print("ORAC-NT Structural Monitor started.")
    print(f"Checking every {CHECK_INTERVAL//3600} hour(s).")
    print("Press Ctrl+C to stop.\n")

    while True:
        try:
            run_once()
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break
        except Exception as e:
            print(f"Error: {e}. Retrying in 60s...")
            time.sleep(60)

# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if '--continuous' in sys.argv:
        run_continuous()
    else:
        run_once()
        print("\nFor continuous monitoring run:")
        print("  python btc_alert_system.py --continuous")
