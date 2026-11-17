#!/usr/bin/env python3
"""
organize_project.py
===================
Automated workspace reorganization script for the AI Gold Quantitative Trading System.

This script structures the workspace into clean, modular, production-ready folders:
  - pine/                    : Production strategy (AI_Gold_Quant_Strategy.pine)
  - pine/comparison/         : Historical research & comparative Pine scripts (v1-v12)
  - src/                     : Core production modules (features, model_engine, backtester, pine_compiler)
  - data/                    : Synchronized multi-asset data loader & fetch utility
  - data/cache/              : Cached OHLCV historical CSV datasets
  - frontend/                : Interactive Live Strategy Studio (gold_live_dashboard.html)
  - frontend/archive/        : Previous dashboard prototypes & generators
  - research/phases/         : Historical research phases (phase2 to phase18)
  - research/scripts/        : Evaluation & metric generation utilities
  - results/metrics/         : Out-of-sample benchmark & daily performance CSVs
  - results/screenshots/     : Structured screenshot galleries for visual verification
  - docs/                    : Project brief and architecture specifications

Usage:
    python3 organize_project.py           # Executes the reorganization
    python3 organize_project.py --dry-run # Previews all file operations without moving
"""

import os
import sys
import shutil
import glob

DRY_RUN = "--dry-run" in sys.argv

def log_action(action, src, dest=None):
    tag = "[DRY-RUN]" if DRY_RUN else "[OK]"
    if dest:
        print(f"{tag:<10} {action:<6}: {src} -> {dest}")
    else:
        print(f"{tag:<10} {action:<6}: {src}")

def ensure_dir(path):
    if not os.path.exists(path):
        log_action("MKDIR", path)
        if not DRY_RUN:
            os.makedirs(path, exist_ok=True)

def safe_move(src, dest_dir, rename_to=None):
    if not os.path.exists(src):
        return
    ensure_dir(dest_dir)
    target_name = rename_to if rename_to else os.path.basename(src)
    dest_path = os.path.join(dest_dir, target_name)
    
    # Don't move if source and destination are identical
    if os.path.abspath(src) == os.path.abspath(dest_path):
        return

    log_action("MOVE", src, dest_path)
    if not DRY_RUN:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        shutil.move(src, dest_path)

def safe_copy(src, dest_dir, rename_to=None):
    if not os.path.exists(src):
        return
    ensure_dir(dest_dir)
    target_name = rename_to if rename_to else os.path.basename(src)
    dest_path = os.path.join(dest_dir, target_name)
    
    if os.path.abspath(src) == os.path.abspath(dest_path):
        return

    log_action("COPY", src, dest_path)
    if not DRY_RUN:
        shutil.copy2(src, dest_path)

def main():
    print("=" * 80)
    print("AI GOLD QUANTITATIVE SYSTEM — WORKSPACE REORGANIZATION TOOL")
    print("=" * 80)
    if DRY_RUN:
        print(">>> RUNNING IN PREVIEW (DRY-RUN) MODE — NO FILES WILL BE MODIFIED <<<\n")

    # 1. Ensure Standard Directories
    target_dirs = [
        "pine",
        "pine/comparison",
        "src",
        "data",
        "data/cache",
        "frontend",
        "frontend/archive",
        "research/phases",
        "research/scripts",
        "research/models",
        "results",
        "results/metrics",
        "results/screenshots/backtest_overview",
        "results/screenshots/ml_model_stats",
        "results/screenshots/validation_phases",
        "results/screenshots/daily_breakdown",
        "docs"
    ]
    for d in target_dirs:
        ensure_dir(d)

    # 2. Organize Pine Scripts
    # Keep pine/AI_Gold_Quant_Strategy.pine as production; move other comparison/phase scripts
    print("\n--- [1/7] Organizing Pine Scripts ---")
    for pine_file in glob.glob("*.pine"):
        if pine_file == "AI_Gold_Quant_Strategy.pine":
            safe_move(pine_file, "pine")
        else:
            safe_move(pine_file, "pine/comparison")

    # 3. Organize Research Phases
    print("\n--- [2/7] Organizing Research Python Phases ---")
    for phase_file in glob.glob("phase*.py"):
        safe_move(phase_file, "research/phases")

    # Auxiliary research scripts & models
    safe_move("generate_report_metrics.py", "research/scripts")
    safe_move("fix_all.py", "research/scripts")
    safe_move("model_weights.pkl", "research/models")

    # 4. Organize Frontend & Live Dashboard
    print("\n--- [3/7] Organizing Frontend & Dashboard Files ---")
    # Live dashboard HTML is final
    if os.path.exists("gold_live_dashboard.html"):
        safe_copy("gold_live_dashboard.html", "frontend")
        safe_copy("gold_live_dashboard.html", "frontend", rename_to="index.html")
    # updated_dashboard.py is the primary dashboard builder
    if os.path.exists("updated_dashboard.py"):
        safe_copy("updated_dashboard.py", "frontend", rename_to="build_dashboard.py")
        safe_move("updated_dashboard.py", "frontend/archive")
    # Archive older dashboard builders
    safe_move("build_gold_dashboard.py", "frontend/archive")
    safe_move("gold_live_dashboard.py", "frontend/archive")
    safe_move("gold_quant_dashboard.html", "frontend/archive")

    # 5. Organize Data Files & Utilities
    print("\n--- [4/7] Organizing Data Files & Binance Cache ---")
    safe_move("fetch_data.py", "data")
    for csv_file in glob.glob("*.csv"):
        # Don't move results CSVs if in root by mistake
        if "metric" in csv_file or "benchmark" in csv_file:
            safe_move(csv_file, "results/metrics")
        else:
            safe_move(csv_file, "data/cache")

    # 6. Organize Documentation
    print("\n--- [5/7] Organizing Documentation & Specs ---")
    safe_move("AI_Trading_Indicator_Project_Brief(2).pdf", "docs")
    safe_move("DESIGN.md", "docs")

    # 7. Organize Results
    print("\n--- [6/7] Structuring Results Section ---")
    if os.path.exists("results/model_benchmarks.csv"):
        safe_copy("results/model_benchmarks.csv", "results/metrics")
    if os.path.exists("results/daily_test_metrics.csv"):
        safe_copy("results/daily_test_metrics.csv", "results/metrics")

    # 8. Update data/loader.py for Multi-Path Resolution
    print("\n--- [7/7] Updating Multi-Path Data Resolvers ---")
    updated_loader = '''import os
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
'''
    if not DRY_RUN:
        with open("data/loader.py", "w") as f:
            f.write(updated_loader)
        log_action("UPDATE", "data/loader.py (Multi-Path resolver enabled)")

    print("\n" + "=" * 80)
    print("SUCCESS: Project structure is now organized!")
    print("""
Target Layout:
├── pine/
│   ├── AI_Gold_Quant_Strategy.pine    <-- [FINAL PRODUCTION STRATEGY]
│   └── comparison/                    <-- [Historical Pine scripts v1-v12]
├── src/                               <-- [Production Modules]
│   ├── features.py
│   ├── model_engine.py
│   ├── backtester.py
│   └── pine_compiler.py
├── data/
│   ├── loader.py                      <-- [Robust multi-path synchronized loader]
│   ├── fetch_data.py
│   └── cache/                         <-- [All 15m OHLCV historical CSV datasets]
├── frontend/
│   ├── gold_live_dashboard.html       <-- [FINAL INTERACTIVE LIVE DASHBOARD]
│   ├── index.html                     <-- [Default web entrypoint]
│   ├── build_dashboard.py             <-- [Generator script for dashboard]
│   └── archive/                       <-- [Older dashboard prototypes]
├── research/
│   ├── phases/                        <-- [phase2 to phase18 progression files]
│   ├── scripts/                       <-- [generate_report_metrics.py, etc.]
│   └── models/                        <-- [Legacy model weights]
├── results/
│   ├── README.md                      <-- [Comprehensive Markdown benchmark report]
│   ├── metrics/                       <-- [model_benchmarks.csv, daily_test_metrics.csv]
│   └── screenshots/                   <-- [Structured screenshot folders]
│       ├── backtest_overview/
│       ├── ml_model_stats/
│       ├── validation_phases/
│       └── daily_breakdown/
├── docs/                              <-- [PDF Brief & Design specs]
├── .gitignore                         <-- [Production-grade git ignore]
├── requirements.txt                   <-- [Clean dependency list]
└── setup_project.py                   <-- [Master pipeline execution script]
""")
    print("=" * 80)

if __name__ == "__main__":
    main()
