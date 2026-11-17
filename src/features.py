import numpy as np
import pandas as pd

def pine_rma(series, length):
    return series.ewm(alpha=1.0 / length, adjust=False).mean()

def engineer_features(df):
    df = df.copy()
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]
    eur_c, btc_c = df["eur_close"], df["btc_close"]

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = pine_rma(tr, 14)
    atr_pct = (atr14 / c) + 1e-10
    df["atr"] = atr14

    bar_shock = (h - l) / (atr14 + 1e-10)
    jump_shock = (c - c.shift(2)).abs() / (atr14 + 1e-10)
    df["shock_ratio"] = pd.concat([bar_shock, bar_shock.shift(1), jump_shock / 1.5], axis=1).max(axis=1)

    # Multi-horizon volatility-adjusted momentum
    df["f1"] = ((c - c.shift(2)) / c.shift(2)) / atr_pct
    df["f2"] = ((c - c.shift(4)) / c.shift(4)) / atr_pct
    df["f3"] = ((c - c.shift(12)) / c.shift(12)) / atr_pct

    # Session VWAP deviation
    typ_p = (h + l + c) / 3.0
    vwap24 = (typ_p * v).rolling(24).sum() / (v.rolling(24).sum() + 1e-10)
    df["vwap"] = vwap24
    df["f4"] = ((c - vwap24) / c) / atr_pct

    # Wilder RSI(14)
    delta = c.diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=df.index)
    rsi14 = 100.0 - (100.0 / (1.0 + (pine_rma(gain, 14) / (pine_rma(loss, 14) + 1e-10))))
    df["rsi14_raw"] = rsi14
    df["f5"] = (rsi14 - 50.0) / 50.0

    # Normalized MACD Histogram
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_hist = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    df["f6"] = macd_hist / (atr14 + 1e-10)

    # 16-period Stochastic Range Position
    hh16 = h.rolling(16).max()
    ll16 = l.rolling(16).min()
    df["f7"] = np.where((hh16 - ll16) == 0, 0.0, ((c - ll16) / (hh16 - ll16)) - 0.5)

    # Multi-scale EMA Trend Hierarchy
    ema20  = c.ewm(span=20, adjust=False).mean()
    ema60  = c.ewm(span=60, adjust=False).mean()
    ema180 = c.ewm(span=180, adjust=False).mean()
    df["ema20"]  = ema20
    df["ema60"]  = ema60
    df["ema180"] = ema180
    df["f8"] = ((ema20 - ema60) / c) / atr_pct
    df["f9"] = ((ema60 - ema180) / c) / atr_pct

    # Volume-Spread Order Flow Pressure
    rng = (h - l) + 1e-10
    signed_vol = ((c - l) - (h - c)) / rng * v
    df["f10"] = signed_vol.rolling(4).sum() / (v.rolling(4).sum() + 1e-10)

    # Cross-Asset Macro Inputs (USD Proxy & Liquidity Index)
    eur_ret4  = (eur_c - eur_c.shift(4)) / eur_c.shift(4)
    gold_ret4 = (c - c.shift(4)) / c.shift(4)
    btc_ret4  = (btc_c - btc_c.shift(4)) / btc_c.shift(4)
    df["f11"] = eur_ret4 * 100.0
    df["f12"] = (eur_ret4 - gold_ret4 + 0.15 * btc_ret4) * 100.0

    # Kaufman Efficiency Ratio (12)
    net_move = (c - c.shift(12)).abs()
    sum_moves = c.diff().abs().rolling(12).sum()
    df["er12"] = np.where(sum_moves == 0, 0.0, net_move / sum_moves)

    return df
