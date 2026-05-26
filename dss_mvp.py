"""
ORAC-NT Distributed Structural Sensors — MVP
=============================================
Author : Dimitar Kretski
DOI    : 10.5281/zenodo.20315517

4 Sensors from public Binance data:
  S1: SR Macro     = P_coarse / P_fine (GravOpt freezing)
  S2: Volume Stress = Z-score anomaly (EdgeSense)
  S3: Kurtosis     = fat tail detector (ORAC-NT veto)
  S4: Micro-SR     = hourly vs daily SR ratio

W = Q·D − T composite health score:
  Q = sensor quality (how clean the signal)
  D = market importance (volatility context)
  T = structural entropy (combined stress)

W > 0 → market healthy
W < 0 → market freezing / stressed

Usage:
    pip install ccxt pandas numpy matplotlib scipy
    python dss_mvp.py
    python dss_mvp.py --continuous
"""

import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kurtosis as scipy_kurtosis
import time, json, os
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

SYMBOL           = 'BTC/USDT'
WINDOW           = 30
COARSE_THRESHOLD = 0.03
FINE_THRESHOLD   = 0.005
CHECK_INTERVAL   = 3600
LOG_FILE         = 'dss_alerts.log'
STATE_FILE       = 'dss_state.json'

# Sensor thresholds
SR_FLUID  = 0.05
SR_RIGID  = 0.45
KURT_HIGH = 7.0
VOLZ_HIGH = 2.5
MICRO_HIGH = 2.0   # hourly SR / daily SR ratio

# W formula weights
ALPHA = 1.0   # Q weight
BETA  = 0.65  # T threshold (T_K from spin glass research)

# ─────────────────────────────────────────────────────────────
# Fetch data — daily + hourly
# ─────────────────────────────────────────────────────────────

def fetch_daily(days=90):
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.milliseconds() - days * 24 * 3600 * 1000
    ohlcv = exchange.fetch_ohlcv(SYMBOL, '1d', since=since, limit=days+5)
    df = pd.DataFrame(ohlcv,
                      columns=['ts','open','high','low','close','volume'])
    df['date'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.set_index('date').drop(columns=['ts'])
    return df[~df.index.duplicated()]

def fetch_hourly(hours=72):
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.milliseconds() - hours * 3600 * 1000
    ohlcv = exchange.fetch_ohlcv(SYMBOL, '1h', since=since, limit=hours+5)
    df = pd.DataFrame(ohlcv,
                      columns=['ts','open','high','low','close','volume'])
    df['date'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.set_index('date').drop(columns=['ts'])
    return df[~df.index.duplicated()]

# ─────────────────────────────────────────────────────────────
# Sensor computations
# ─────────────────────────────────────────────────────────────

def sensor_sr_macro(df_daily):
    """S1: SR Macro — GravOpt freezing observable"""
    r = df_daily['close'].pct_change()
    p_c = r.abs().rolling(WINDOW).apply(
        lambda x: np.mean(np.abs(x) > COARSE_THRESHOLD), raw=True)
    p_f = r.abs().rolling(WINDOW).apply(
        lambda x: np.mean(np.abs(x) > FINE_THRESHOLD), raw=True)
    sr = p_c.clip(lower=1e-3) / p_f.clip(lower=1e-3)
    return float(sr.iloc[-1])

def sensor_volume_stress(df_daily):
    """S2: Volume stress — EdgeSense Z-score"""
    vz = df_daily['volume'].rolling(WINDOW).apply(
        lambda x: (x[-1] - np.mean(x)) / (np.std(x) + 1e-10), raw=True)
    return float(vz.iloc[-1])

def sensor_kurtosis(df_daily):
    """S3: Kurtosis — ORAC-NT fat tail veto"""
    r = df_daily['close'].pct_change().dropna()
    if len(r) < WINDOW:
        return 3.0
    kurt = scipy_kurtosis(r.tail(WINDOW), fisher=False, bias=False)
    return float(kurt)

def sensor_micro_sr(df_hourly, sr_daily):
    """S4: Micro-SR — hourly vs daily SR ratio (ORAC adaptive)"""
    r = df_hourly['close'].pct_change()
    coarse_h = 0.015  # 1.5% hourly = coarse
    fine_h   = 0.003  # 0.3% hourly = fine
    p_c = np.mean(np.abs(r.tail(48)) > coarse_h)
    p_f = np.mean(np.abs(r.tail(48)) > fine_h)
    sr_hourly = max(p_c, 1e-3) / max(p_f, 1e-3)
    # Ratio of hourly to daily SR
    ratio = sr_hourly / (sr_daily + 1e-10)
    return float(sr_hourly), float(ratio)

# ─────────────────────────────────────────────────────────────
# W formula — composite health score
# ─────────────────────────────────────────────────────────────

def compute_W(sr, volume_z, kurt, sr_hourly, micro_ratio):
    """
    W = Q·D − T

    Q = sensor signal quality (normalized)
    D = market importance (volatility context)
    T = structural entropy (combined stress)
    """

    # Normalize sensors to [0,1]
    sr_norm    = np.clip(sr / SR_RIGID, 0, 1)
    volz_norm  = np.clip(abs(volume_z) / VOLZ_HIGH, 0, 1)
    kurt_norm  = np.clip((kurt - 3) / (KURT_HIGH - 3), 0, 1)
    micro_norm = np.clip(micro_ratio / MICRO_HIGH, 0, 1)

    # Q: average sensor quality
    Q = np.mean([sr_norm, volz_norm, kurt_norm, micro_norm])

    # D: market importance
    # High when sensors disagree (heterogeneous signals = more information)
    sensor_vals = [sr_norm, volz_norm, kurt_norm, micro_norm]
    D = 1.0 + np.std(sensor_vals)

    # T: structural entropy (stress)
    # T > BETA → system freezing
    T = 0.3*sr_norm + 0.3*volz_norm + 0.2*kurt_norm + 0.2*micro_norm

    # W = Q·D − T
    W = ALPHA * Q * D - T

    return {
        'W':          W,
        'Q':          Q,
        'D':          D,
        'T':          T,
        'sr_norm':    sr_norm,
        'volz_norm':  volz_norm,
        'kurt_norm':  kurt_norm,
        'micro_norm': micro_norm,
    }

# ─────────────────────────────────────────────────────────────
# Regime classification
# ─────────────────────────────────────────────────────────────

def classify(W, sr, kurt, volume_z):
    if W > 0.5:
        return "HEALTHY"
    elif W > 0.0:
        return "MODERATE"
    elif sr < SR_FLUID:
        return "ULTRA_FLUID"
    elif sr > SR_RIGID:
        return "STRESSED_RIGID"
    elif kurt > KURT_HIGH:
        return "FAT_TAIL"
    elif abs(volume_z) > VOLZ_HIGH:
        return "VOLUME_ANOMALY"
    else:
        return "WEAK"

REGIME_DESC = {
    "HEALTHY":        "All sensors normal. Market structurally healthy.",
    "MODERATE":       "Mixed signals. Monitor closely.",
    "ULTRA_FLUID":    "SR near zero. Historically precedes volatility expansion.",
    "STRESSED_RIGID": "High SR + stress. Large moves dominating.",
    "FAT_TAIL":       "Kurtosis spike. Extreme move distribution.",
    "VOLUME_ANOMALY": "Unusual volume. Abnormal activity detected.",
    "WEAK":           "W below zero. Market health declining.",
}

# ─────────────────────────────────────────────────────────────
# Notification
# ─────────────────────────────────────────────────────────────

def notify(title, message, urgency='normal'):
    try:
        from plyer import notification
        notification.notify(title=title, message=message,
                            app_name='DSS Monitor', timeout=15)
    except Exception:
        pass
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    print(f"\n{'='*60}")
    print(f"  [{urgency.upper()}] {title}")
    print(f"  {message}")
    print(f"  {ts}")
    print(f"{'='*60}")

# ─────────────────────────────────────────────────────────────
# Run one check
# ─────────────────────────────────────────────────────────────

def run_once():
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    print(f"\n{'='*60}")
    print(f"  ORAC-NT DSS MONITOR — {ts}")
    print(f"{'='*60}")

    # Fetch
    df_d = fetch_daily(days=90)
    df_h = fetch_hourly(hours=72)

    # Compute sensors
    sr        = sensor_sr_macro(df_d)
    volume_z  = sensor_volume_stress(df_d)
    kurt      = sensor_kurtosis(df_d)
    sr_h, micro_ratio = sensor_micro_sr(df_h, sr)

    # W formula
    w = compute_W(sr, volume_z, kurt, sr_h, micro_ratio)
    W = w['W']

    # Regime
    price  = float(df_d['close'].iloc[-1])
    regime = classify(W, sr, kurt, volume_z)

    # Print
    print(f"\n  BTC Price    : ${price:,.0f}")
    print(f"\n  SENSORS:")
    print(f"    S1 SR Macro   : {sr:.4f}  (norm={w['sr_norm']:.3f})")
    print(f"    S2 Volume Z   : {volume_z:.2f}  (norm={w['volz_norm']:.3f})")
    print(f"    S3 Kurtosis   : {kurt:.2f}  (norm={w['kurt_norm']:.3f})")
    print(f"    S4 Micro SR   : {sr_h:.4f}  ratio={micro_ratio:.2f}  (norm={w['micro_norm']:.3f})")
    print(f"\n  W FORMULA:")
    print(f"    Q = {w['Q']:.4f}  (signal quality)")
    print(f"    D = {w['D']:.4f}  (market importance)")
    print(f"    T = {w['T']:.4f}  (structural entropy)")
    print(f"    W = Q·D − T = {W:.4f}")
    print(f"\n  REGIME: {regime}")
    print(f"  {REGIME_DESC[regime]}")

    # Log
    log_line = (f"[{ts}] BTC=${price:,.0f} | "
                f"W={W:.4f} | SR={sr:.4f} | VolZ={volume_z:.2f} | "
                f"Kurt={kurt:.2f} | MicroSR={sr_h:.4f} | Regime={regime}")
    with open(LOG_FILE, 'a') as f:
        f.write(log_line + '\n')

    # Save state
    prev_regime = 'NORMAL'
    if os.path.exists(STATE_FILE):
        try:
            prev_regime = json.load(open(STATE_FILE)).get('regime', 'NORMAL')
        except Exception:
            pass

    state = {'regime': regime, 'W': W, 'SR': sr, 'price': price,
             'volume_z': volume_z, 'kurt': kurt, 'micro_sr': sr_h}
    json.dump(state, open(STATE_FILE, 'w'), indent=2)

    # Alert on regime change
    if regime != prev_regime:
        notify(
            f'DSS Regime: {regime}',
            f'{REGIME_DESC[regime]}\nW={W:.4f} | SR={sr:.4f} | BTC=${price:,.0f}',
            urgency='critical' if regime in ['STRESSED_RIGID','FAT_TAIL'] else 'normal'
        )

    return state

# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def plot_historical():
    print("\n  Generating historical DSS chart...")
    df_d = fetch_daily(days=365)
    r    = df_d['close'].pct_change()

    # Compute all sensors historically
    p_c  = r.abs().rolling(WINDOW).apply(
        lambda x: np.mean(np.abs(x) > COARSE_THRESHOLD), raw=True)
    p_f  = r.abs().rolling(WINDOW).apply(
        lambda x: np.mean(np.abs(x) > FINE_THRESHOLD), raw=True)
    sr   = p_c.clip(lower=1e-3) / p_f.clip(lower=1e-3)

    vz   = df_d['volume'].rolling(WINDOW).apply(
        lambda x: (x[-1]-np.mean(x))/(np.std(x)+1e-10), raw=True)

    kurt_s = r.rolling(WINDOW).apply(
        lambda x: scipy_kurtosis(x, fisher=False, bias=False)
        if len(x)>4 else 3.0, raw=True)

    # W historical (simplified)
    sr_n   = sr.clip(0, SR_RIGID) / SR_RIGID
    vz_n   = vz.abs().clip(0, VOLZ_HIGH) / VOLZ_HIGH
    kurt_n = (kurt_s - 3).clip(0, KURT_HIGH-3) / (KURT_HIGH-3)
    Q_hist = (sr_n + vz_n + kurt_n) / 3
    T_hist = 0.4*sr_n + 0.3*vz_n + 0.3*kurt_n
    W_hist = Q_hist * 1.2 - T_hist

    df_clean = sr.dropna().index

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True,
                              facecolor='#0e1117')
    fig.suptitle(
        "ORAC-NT Distributed Structural Sensors — BTC/USDT\n"
        "W = Q·D − T  |  4-Sensor Composite Health Score\n"
        "Dimitar Kretski  |  doi.org/10.5281/zenodo.20315517",
        fontsize=11, fontweight='bold', color='white'
    )

    # Panel 1 — Price
    axes[0].plot(df_d.index, df_d['close'], color='#f7931a', lw=1.5)
    axes[0].set_ylabel("BTC Price", color='white')
    axes[0].set_yscale('log')

    # Panel 2 — W score
    W_s = W_hist.reindex(df_d.index)
    axes[1].plot(df_d.index, W_s, color='#00ffcc', lw=2)
    axes[1].axhline(0, color='white', lw=1.5, ls='--', alpha=0.6,
                    label='W=0 (healthy/weak boundary)')
    axes[1].fill_between(df_d.index, 0, W_s,
                          where=W_s > 0, alpha=0.3, color='#00ffcc',
                          label='Healthy (W>0)')
    axes[1].fill_between(df_d.index, W_s, 0,
                          where=W_s < 0, alpha=0.3, color='red',
                          label='Weak (W<0)')
    axes[1].set_ylabel("W score", color='white')
    axes[1].legend(fontsize=8, facecolor='#0e1117', labelcolor='white')

    # Panel 3 — Individual sensors
    axes[2].plot(df_d.index, sr.reindex(df_d.index),
                 color='#4488ff', lw=1.5, label='S1: SR Macro', alpha=0.9)
    axes[2].plot(df_d.index, vz.abs().reindex(df_d.index)/5,
                 color='#ff9900', lw=1.2, label='S2: |VolZ|/5', alpha=0.8)
    axes[2].plot(df_d.index, kurt_n.reindex(df_d.index),
                 color='#ff4488', lw=1.2, label='S3: Kurt norm', alpha=0.8)
    axes[2].axhline(SR_FLUID, color='red', lw=1, ls=':', alpha=0.6)
    axes[2].axhline(SR_RIGID, color='orange', lw=1, ls=':', alpha=0.6)
    axes[2].set_ylabel("Sensor values", color='white')
    axes[2].legend(fontsize=7, facecolor='#0e1117', labelcolor='white')

    # Panel 4 — T entropy
    T_s = T_hist.reindex(df_d.index)
    axes[3].fill_between(df_d.index, 0, T_s, alpha=0.6, color='#ff6600')
    axes[3].axhline(BETA, color='red', lw=1.5, ls='--', alpha=0.8,
                    label=f'T_K={BETA} (spin glass threshold)')
    axes[3].set_ylabel("T entropy", color='white')
    axes[3].set_xlabel("Date", color='white')
    axes[3].legend(fontsize=8, facecolor='#0e1117', labelcolor='white')

    for ax in axes:
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.12)
        for spine in ax.spines.values():
            spine.set_color('#444')

    plt.tight_layout()
    plt.savefig('dss_historical.png', dpi=150,
                facecolor='#0e1117', bbox_inches='tight')
    print("  Saved: dss_historical.png")

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    import sys
    if '--plot' in sys.argv:
        plot_historical()
    elif '--continuous' in sys.argv:
        print("DSS Monitor started. Ctrl+C to stop.")
        while True:
            try:
                run_once()
                time.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except Exception as e:
                print(f"Error: {e}. Retrying in 60s...")
                time.sleep(60)
    else:
        run_once()
        plot_historical()
        print("\nFor continuous: python dss_mvp.py --continuous")
        print("For plot only:  python dss_mvp.py --plot")

if __name__ == "__main__":
    main()
