"""
BTC/USDT Structural Rigidity Analysis — v2
===========================================
Author : Dimitar Kretski
DOI    : 10.5281/zenodo.20315517

5 Structural Sensors:
  1. SR = P_coarse / P_fine     (rigidity)
  2. Volatility                  (macro stress)
  3. Kurtosis                    (tail risk)
  4. Volume Z-score              (liquidity stress)
  5. SR x Volume composite       (combined signal)

Usage:
    pip install ccxt pandas numpy matplotlib scipy
    python btc_sr_analysis.py
"""

import ccxt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import warnings
warnings.filterwarnings('ignore')

SYMBOL           = 'BTC/USDT'
TIMEFRAME        = '1d'
START_DATE       = '2021-01-01T00:00:00Z'
WINDOW           = 30
COARSE_THRESHOLD = 0.03
FINE_THRESHOLD   = 0.005

EVENTS = {
    'BTC ATH':       '2021-11-10',
    'LUNA crash':    '2022-05-09',
    'FTX collapse':  '2022-11-08',
    'BTC bottom':    '2022-11-21',
    'BTC recovery':  '2023-01-14',
    'ETF approval':  '2024-01-10',
    'BTC ATH 2024':  '2024-03-14',
}

EVENT_COLORS = [
    '#ff4444','#ff6600','#ffaa00',
    '#ffdd00','#88ff00','#00ffcc','#4488ff'
]

def fetch_btc_data():
    print("Fetching BTC/USDT daily data from Binance...")
    exchange = ccxt.binance({'enableRateLimit': True})
    all_ohlcv = []
    since = exchange.parse8601(START_DATE)
    while True:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
        if not ohlcv: break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000: break
    df = pd.DataFrame(all_ohlcv,
                      columns=['timestamp','open','high','low','close','volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('date').drop(columns=['timestamp'])
    df = df[~df.index.duplicated(keep='first')]
    print(f"Downloaded: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
    return df

def compute_sensors(df):
    df = df.copy()
    df['returns'] = df['close'].pct_change()

    # Sensor 1: SR
    df['p_coarse'] = df['returns'].abs().rolling(WINDOW).apply(
        lambda x: np.mean(np.abs(x) > COARSE_THRESHOLD), raw=True)
    df['p_fine'] = df['returns'].abs().rolling(WINDOW).apply(
        lambda x: np.mean(np.abs(x) > FINE_THRESHOLD), raw=True)
    df['SR'] = (df['p_coarse'].clip(lower=1e-3) /
                df['p_fine'].clip(lower=1e-3))

    # Sensor 2: Volatility
    df['volatility'] = df['returns'].rolling(WINDOW).std() * np.sqrt(365)

    # Sensor 3: Kurtosis
    df['kurtosis'] = df['returns'].rolling(WINDOW).apply(
        lambda x: kurtosis(x, fisher=False, bias=False)
        if len(x) > 4 else 3.0, raw=True)

    # Sensor 4: Volume Z-score (liquidity stress)
    df['volume_z'] = df['volume'].rolling(WINDOW).apply(
        lambda x: (x[-1] - np.mean(x)) / (np.std(x) + 1e-10), raw=True)

    # Sensor 5: Composite = SR * norm(volume_z)
    # High SR + high volume = stressed rigid market
    vol_z_norm = (df['volume_z'] - df['volume_z'].mean()) / (df['volume_z'].std() + 1e-10)
    df['composite'] = df['SR'] * (1 + 0.3 * vol_z_norm.clip(lower=0))

    return df.dropna()

def analyze_events(df):
    print("\n" + "="*70)
    print("  SENSOR READINGS AROUND KEY EVENTS")
    print("="*70)
    print(f"  {'Event':20s} | {'SR pre':>6} | {'SR post':>7} | "
          f"{'VolZ pre':>8} | {'VolZ post':>9} | {'Delta SR':>8}")
    print(f"  {'-'*68}")

    for name, date in EVENTS.items():
        try:
            d = pd.Timestamp(date)
            sr_pre   = df.loc[d-pd.Timedelta(days=30):d, 'SR'].mean()
            sr_post  = df.loc[d:d+pd.Timedelta(days=30), 'SR'].mean()
            vz_pre   = df.loc[d-pd.Timedelta(days=30):d, 'volume_z'].mean()
            vz_post  = df.loc[d:d+pd.Timedelta(days=30), 'volume_z'].mean()
            print(f"  {name:20s} | {sr_pre:6.3f} | {sr_post:7.3f} | "
                  f"{vz_pre:8.2f} | {vz_post:9.2f} | {sr_post-sr_pre:+8.3f}")
        except Exception as e:
            print(f"  {name}: {e}")

    print(f"\n  Current SR  : {df['SR'].iloc[-1]:.4f}")
    print(f"  Current VolZ: {df['volume_z'].iloc[-1]:.2f}")
    print(f"  SR mean     : {df['SR'].mean():.4f}")

def plot_results(df):
    fig, axes = plt.subplots(5, 1, figsize=(16, 18), sharex=True,
                              facecolor='#0e1117')
    fig.suptitle(
        "ORAC-NT Distributed Structural Sensors — BTC/USDT\n"
        "SR + Volatility + Kurtosis + Volume Stress + Composite\n"
        "Dimitar Kretski  |  doi.org/10.5281/zenodo.20315517",
        fontsize=11, fontweight='bold', color='white'
    )

    sr_mean = df['SR'].mean()
    sr_std  = df['SR'].std()

    # Panel 1 — Price
    axes[0].plot(df.index, df['close'], color='#f7931a', lw=1.5)
    axes[0].set_ylabel("BTC Price (USD)", color='white')
    axes[0].set_yscale('log')

    # Panel 2 — SR
    axes[1].plot(df.index, df['SR'], color='#00ffcc', lw=1.5)
    axes[1].axhline(sr_mean, color='white', lw=1, ls='--', alpha=0.5,
                    label=f'Mean={sr_mean:.3f}')
    axes[1].axhspan(sr_mean+sr_std, df['SR'].max(),
                    alpha=0.1, color='red', label='High rigidity')
    axes[1].set_ylabel("SR = P_c/P_f", color='white')
    axes[1].legend(fontsize=7, facecolor='#0e1117', labelcolor='white')

    # Panel 3 — Volume Z-score
    vz = df['volume_z']
    axes[2].fill_between(df.index, 0, vz,
                          where=vz > 0, color='#ff4444', alpha=0.6,
                          label='Above avg volume')
    axes[2].fill_between(df.index, 0, vz,
                          where=vz < 0, color='#4488ff', alpha=0.6,
                          label='Below avg volume')
    axes[2].axhline(0, color='white', lw=1, ls='--', alpha=0.4)
    axes[2].set_ylabel("Volume Z-score", color='white')
    axes[2].legend(fontsize=7, facecolor='#0e1117', labelcolor='white')

    # Panel 4 — Kurtosis
    axes[3].plot(df.index, df['kurtosis'], color='#4488ff', lw=1.2)
    axes[3].axhline(3.0, color='white', lw=1, ls='--', alpha=0.5,
                    label='Gaussian=3')
    axes[3].set_ylabel("Kurtosis (30d)", color='white')
    axes[3].legend(fontsize=7, facecolor='#0e1117', labelcolor='white')

    # Panel 5 — Composite signal
    axes[4].plot(df.index, df['composite'], color='#ff9900', lw=1.8)
    axes[4].axhline(df['composite'].mean(), color='white', lw=1,
                    ls='--', alpha=0.5, label='Mean composite')
    axes[4].set_ylabel("Composite SR×Vol", color='white')
    axes[4].set_xlabel("Date", color='white')
    axes[4].legend(fontsize=7, facecolor='#0e1117', labelcolor='white')

    # Event lines
    for i, (name, date) in enumerate(EVENTS.items()):
        col = EVENT_COLORS[i % len(EVENT_COLORS)]
        for ax in axes:
            ax.axvline(pd.Timestamp(date), color=col, lw=1.2, ls='--', alpha=0.7)
        axes[0].text(pd.Timestamp(date), df['close'].quantile(0.92),
                     name, rotation=90, fontsize=7, color=col, alpha=0.9,
                     verticalalignment='bottom')

    for ax in axes:
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.15)
        for spine in ax.spines.values():
            spine.set_color('#444')

    plt.tight_layout()
    plt.savefig('btc_sr_analysis.png', dpi=150,
                facecolor='#0e1117', bbox_inches='tight')
    print("\n  Saved: btc_sr_analysis.png")

def main():
    print("="*70)
    print("  DISTRIBUTED STRUCTURAL SENSORS — BTC/USDT")
    print("  5 Sensors: SR + Vol + Kurt + VolumeZ + Composite")
    print("="*70)

    df_raw = fetch_btc_data()
    df     = compute_sensors(df_raw)
    analyze_events(df)
    plot_results(df)

    print("\n" + "="*70)
    print("  WHO WOULD BE INTERESTED AND WHY")
    print("="*70)
    print("""
  TARGET AUDIENCE:

  1. QUANT FUNDS / HFT FIRMS
     Why: SR measures when large moves dominate small moves.
     This is directly related to execution toxicity and
     slippage estimation. High SR = bad time to execute
     large orders.

  2. CRYPTO EXCHANGES / MARKET MAKERS
     Why: Volume Z-score + SR composite detects liquidity
     stress BEFORE it becomes visible in price. Useful for
     dynamic fee adjustment and risk management.

  3. DEFI PROTOCOLS
     Why: Composite signal can trigger circuit breakers
     or adjust collateral ratios during high-rigidity
     periods. No prediction needed — just state detection.

  4. RISK DESKS (banks, family offices)
     Why: SR regime classification (fluid vs rigid) is a
     clean, interpretable metric for crypto allocation.
     "We reduce exposure when SR > mean + 1 std."

  5. SYSTEMIC RISK RESEARCHERS
     Why: The connection to spin glass freezing theory
     gives a theoretical framework that most market
     indicators lack. This is a publishable angle.

  KEY SELLING POINT:
  "Not a price predictor. A structural health monitor.
   Like a seismograph — it does not predict earthquakes,
   but it tells you when the ground is stressed."
""")

if __name__ == "__main__":
    main()
