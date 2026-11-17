# AI Gold Quantitative Trading System (PAXG/USDT & XAU/USD)

An enterprise-grade, multi-asset quantitative trading framework and machine learning pipeline engineered for institutional gold trading on Binance (`PAXGUSDT`, 15-minute timeframe).

---

## Architecture Overview

```
AI_Trading_Project/
├── setup_project.py                    # Master walk-forward training & evaluation pipeline
├── requirements.txt                    # Python environment dependencies
├── .gitignore                          # Clean production Git exclusion rules
│
├── pine/                               # TradingView Pine Script v5 Strategies
│   ├── AI_Gold_Quant_Strategy.pine     # FINAL PRODUCTION Pine Script v5 Strategy
│   └── comparison/                     # Historical research & phase iterations (v1 - v12)
│
├── src/                                # Core Production Engine
│   ├── features.py                     # 12-feature institutional quant feature engineering
│   ├── model_engine.py                 # Walk-forward ensemble (L2-LR + Tanh MLP + GBDT)
│   ├── backtester.py                   # High-precision simulation & metric evaluation
│   └── pine_compiler.py                # Compiles frozen AI weights into Pine Script v5
│
├── data/                               # Data Infrastructure
│   ├── loader.py                       # Robust multi-path synchronized data loader
│   ├── fetch_data.py                   # Binance live & historical candle fetcher
│   └── cache/                          # Cached multi-year 15m OHLCV CSVs (PAXG, EUR, BTC)
│
├── frontend/                           # Live Strategy Studio & Interactive Dashboard
│   ├── gold_live_dashboard.html        # FINAL interactive live dashboard (WebGL/Canvas)
│   ├── index.html                      # Default dashboard web entry point
│   ├── build_dashboard.py              # Live dashboard generator & Binance sync script
│   └── archive/                        # Previous dashboard prototypes
│
├── research/                           # Experimental Research & Historical Progression
│   ├── phases/                         # Iterative phases (Phase 2 to Phase 18)
│   ├── scripts/                        # Statistical & metric extraction scripts
│   └── models/                         # Serialized legacy weights
│
├── results/                            # Verification Reports & Visual Evidence
│   ├── README.md                       # Comprehensive Markdown evaluation report
│   ├── metrics/                        # Quantitative CSV exports (benchmarks, daily PnL)
│   └── screenshots/                    # Structured screenshot verification galleries
│       ├── backtest_overview/
│       ├── ml_model_stats/
│       ├── validation_phases/
│       └── daily_breakdown/
│
└── docs/                               # Project documentation & brief specifications
```

---

## Quickstart

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Execute Master Pipeline
Run the walk-forward training, hyperparameter optimization, and Pine Script compilation:
```bash
python3 setup_project.py
```
This updates:
- Production Pine Script: `pine/AI_Gold_Quant_Strategy.pine`
- Metric benchmarks: `results/metrics/model_benchmarks.csv`
- Day-by-day test breakdown: `results/metrics/daily_test_metrics.csv`

### 3. Open Live Strategy Dashboard
Open the standalone interactive dashboard directly in your browser:
```bash
open frontend/gold_live_dashboard.html
# or serve locally:
python3 -m http.server 8080 --directory frontend
```

---

## Out-of-Sample Performance Summary (Nov 09 – Nov 16, 2026)

| Architecture | Trades | Win Rate | Profit Factor | Max Drawdown | Net Alpha vs Buy & Hold |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Buy & Hold Benchmark** | 62 | 33.87% | 0.48 | 4.85% | *0.00%* |
| **Standard RSI (30/70)** | 19 | 26.32% | 0.14 | 3.40% | +1.58% |
| **Hybrid AI Ensemble (L2-LR + MLP)** | **21** | **71.43%** | **2.29** | **0.49%** | **+7.53%** |

Full detailed metrics, classification reports, and screenshots are documented in [`results/README.md`](results/README.md).
