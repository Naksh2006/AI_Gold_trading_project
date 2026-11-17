import os
import pandas as pd

def load_synchronized_data(interval="15m"):
    symbols = ["paxgusdt", "eurusdt", "btcusdt"]
    dfs = {}
    for sym in symbols:
        loaded = False
        for days in [1460, 730, 365]:
            candidate_paths = [
                os.path.join("data", "cache", f"{sym}_{interval}_{days}d.csv"),
                os.path.join("data", f"{sym}_{interval}_{days}d.csv"),
                f"{sym}_{interval}_{days}d.csv",
                os.path.join(os.path.dirname(__file__), "cache", f"{sym}_{interval}_{days}d.csv"),
                os.path.join(os.path.dirname(__file__), "..", "data", "cache", f"{sym}_{interval}_{days}d.csv"),
                os.path.join(os.path.dirname(__file__), f"{sym}_{interval}_{days}d.csv")
            ]
            for fname in candidate_paths:
                if os.path.exists(fname):
                    df = pd.read_csv(fname)
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    dfs[sym] = df
                    loaded = True
                    break
            if loaded:
                break
        if not loaded:
            raise FileNotFoundError(f"Dataset for {sym} not found in data/cache/, data/, or working directory.")

    df_gold = dfs["paxgusdt"]
    df_eur  = dfs["eurusdt"][["datetime", "close"]].rename(columns={"close": "eur_close"})
    df_btc  = dfs["btcusdt"][["datetime", "close"]].rename(columns={"close": "btc_close"})

    merged = df_gold.merge(df_eur, on="datetime", how="inner").merge(df_btc, on="datetime", how="inner")
    return merged.sort_values("datetime").reset_index(drop=True)
