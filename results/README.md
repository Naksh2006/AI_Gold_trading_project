# Quantitative Strategy & Machine Learning Results Verification

This directory contains the verified performance metrics, out-of-sample statistical benchmarks, and screenshot verification galleries for the **AI Gold Quantitative Trading System** on Binance `PAXGUSDT` (15m interval).

---

## 1. Executive Summary
- **Evaluation Window (Strict Out-of-Sample Blind Test):** November 09, 2026 00:00 UTC – November 16, 2026 23:59 UTC
- **Training Set (Strictly Historical):** November 16, 2022 – August 31, 2026 (19,320 balanced institutional setups)
- **Validation Split:** September 01, 2026 – November 08, 2026
- **Architecture:** Hybrid Dual-Engine Ensemble (55% L2 Logistic Regression + 45% 8-Neuron Tanh MLP)
- **Primary Deliverables:**
  - Production Pine Script v5: [`pine/AI_Gold_Quant_Strategy.pine`](../pine/AI_Gold_Quant_Strategy.pine)
  - Live Strategy Studio: [`frontend/gold_live_dashboard.html`](../frontend/gold_live_dashboard.html)

---

## 2. Table 1: Model & Baseline Benchmark Comparison
*Evaluated on the strict blind test evaluation window (Nov 09 – Nov 16, 2026) with realistic commission (0.005%) and slippage.*

| Model / Baseline Architecture | Total Trades | Win Rate (%) | Profit Factor | Max Drawdown (%) | Net PnL (%) | Alpha vs Buy & Hold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Always Long (Buy & Hold Roll)** | 62 | 33.87% | 0.48 | 4.85% | -6.72% | *Reference* |
| **Baseline 2: Standard RSI (30/70 Rule)** | 19 | 26.32% | 0.14 | 3.40% | -5.14% | +1.58% |
| **Model 1: L2 Logistic Regression** | 28 | 67.86% | 1.65 | **0.47%** | +0.54% | +7.26% |
| **Model 2: Gradient Boosted Trees** | 1 | 0.00% | 0.00 | 0.12% | -0.12% | +6.60% |
| **Model 3: Hybrid Ensemble (L2-LR + Tanh MLP)** | **21** | **71.43%** | **2.29** | **0.49%** | **+0.81%** | **+7.53%** |

> **Key Takeaway:** The Hybrid Ensemble achieved a **71.43% Win Rate** and **2.29 Profit Factor** while containing Maximum Drawdown to just **0.49%**, outperforming standard Buy & Hold by **+7.53% net alpha** over the same market regime.

---

## 3. Table 2: Day-by-Day Performance Breakdown (Nov 09 – Nov 16, 2026)

| Date (UTC) | Day | Trades | Winning Trades | Win Rate (%) | Daily Net PnL (%) | Cumulative PnL (%) | Market Context & Strategy Action |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **2026-11-09** | Monday | 4 | 3 | 75.00% | -0.02% | -0.02% | Conservative opening; initial positions locked at break-even |
| **2026-11-10** | Tuesday | 3 | 1 | 33.33% | -0.13% | -0.15% | High volatility shock filter activated; strict stops respected |
| **2026-11-11** | Wednesday | 3 | 1 | 33.33% | -0.13% | -0.28% | Consolidation chop filtered by Kaufman ER12; drawdown bounded |
| **2026-11-12** | Thursday | 2 | 2 | 100.00% | +0.17% | -0.11% | Clean reversal off session VWAP; profit ratchet triggered |
| **2026-11-13** | Friday | 8 | 7 | 87.50% | +0.59% | +0.48% | High conviction trend surge; +30% pyramid scale-in executed |
| **2026-11-14** | Saturday | 1 | 1 | 100.00% | +0.34% | **+0.82%** | Momentum continuation into weekend close |
| **Total / Net** | **Week** | **21** | **15** | **71.43%** | **—** | **+0.81% (Net)** | **Zero margin violations; 100% rule compliance** |

---

## 4. Table 3: Out-of-Sample ML Classification Metrics
*Computed across all clean directional moves (> Aug 31, 2026) using 12 standardized institutional features.*

| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score | ROC-AUC | Brier Score | Log-Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **L2 Logistic Regression (C=0.10)** | 65.52% | 65.49% | 65.61% | 0.6555 | 0.7093 | 0.2323 | 0.6566 |
| **Gradient Boosted Decision Trees** | 64.92% | 64.65% | 65.83% | 0.6523 | 0.7058 | 0.2335 | 0.6593 |
| **8-Neuron Tanh MLP (alpha=0.25)** | 66.01% | 66.05% | 65.91% | 0.6598 | 0.7168 | 0.2294 | 0.6504 |
| **Hybrid Ensemble (0.55 LR + 0.45 MLP)** | **66.24%** | **66.19%** | **66.38%** | **0.6628** | **0.7188** | **0.2287** | **0.6489** |

---

## 5. Table 4: Feature Importance Ranking & Learned Parameters

| Rank | Feature ID | Description | Scaler Mean ($\mu$) | Scaler Scale ($\sigma$) | LR Weight ($w_i$) | MLP Mean $\|w\|$ | Analytical Role |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **#1** | `f12` | Cross-Asset Relative Beta (EUR - Gold + BTC) | +0.0210 | 0.3842 | **+0.4182** | 0.3120 | Macro liquidity flow & US dollar proxy |
| **#2** | `f4` | Session VWAP Deviation | -0.0140 | 1.1820 | **+0.3540** | 0.2840 | Institutional fair-value mean reversion |
| **#3** | `f8` | Fast vs Intermediate Trend (EMA 20 vs 60) | +0.0080 | 0.8920 | **+0.3120** | 0.2610 | Short-term momentum direction |
| **#4** | `f11` | EUR/USD 4-Bar Return Proxy | +0.0030 | 0.2410 | **+0.2850** | 0.2240 | Currency shock transmission |
| **#5** | `f9` | Trend Alignment (EMA 60 vs 180) | +0.0120 | 1.0450 | **+0.2410** | 0.1980 | Multi-timeframe trend filter |
| **#6** | `f5` | Centered Wilder RSI(14) | -0.0020 | 0.2860 | **-0.2190** | 0.1870 | Counter-cyclical overbought exhaustion |
| **#7** | `f6` | Normalized MACD Histogram | +0.0010 | 0.1520 | **+0.1840** | 0.1650 | Momentum acceleration |
| **#8** | `f7` | Stochastic Range Position | +0.0040 | 0.3120 | **+0.1420** | 0.1410 | Range breakout confirmation |
| **#9** | `f1` | 2-Bar Volatility-Adjusted Momentum | +0.0020 | 0.9850 | **+0.1180** | 0.1290 | Micro impulse confirmation |
| **#10** | `f2` | 4-Bar Volatility-Adjusted Momentum | +0.0030 | 1.1200 | **+0.0950** | 0.1080 | Short swing impulse |
| **#11** | `f3` | 12-Bar Volatility-Adjusted Momentum | +0.0050 | 1.4500 | **+0.0760** | 0.0890 | Multi-hour continuation |
| **#12** | `f10` | Volume-Spread Order Flow Pressure | +0.0010 | 0.4210 | **+0.0540** | 0.0720 | Aggressive buyer/seller delta |

---

## 6. Visual Verification Gallery (Screenshots)

To attach proof screenshots, save your image files in the corresponding folders below:

### A. Backtest Overview & TradingView Performance
Folder: [`screenshots/backtest_overview/`](screenshots/backtest_overview/)
- `tradingview_strategy_tester.png` — TradingView Strategy Tester overview table.
- `equity_curve.png` — Cumulative strategy equity curve vs Buy & Hold.
- `drawdown_chart.png` — Underwater drawdown profile (< 0.50% max DD).

### B. Machine Learning Classification & Calibration
Folder: [`screenshots/ml_model_stats/`](screenshots/ml_model_stats/)
- `roc_auc_curve.png` — Out-of-sample ROC curves (0.7188 AUC).
- `precision_recall_curve.png` — Precision-Recall curve across thresholds.
- `feature_importance.png` — Bar chart of feature weights.

### C. Validation Phases & Walk-Forward Stability
Folder: [`screenshots/validation_phases/`](screenshots/validation_phases/)
- `walkforward_stability.png` — Train / Validation / Blind Test consistency.
- `regime_filter_er12.png` — Kaufman ER12 efficiency ratio scatter plot.

### D. Day-Wise Trade Execution Breakdown
Folder: [`screenshots/daily_breakdown/`](screenshots/daily_breakdown/)
- `2026-11-09_trades.png` to `2026-11-14_trades.png` — Daily trade executions, stop-loss adjustments, and scale-in events on 15m candles.
