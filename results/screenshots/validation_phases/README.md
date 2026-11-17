# Validation Phases & Walk-Forward Stability Screenshots

This directory stores comparative performance captures across different development phases and walk-forward regimes.

### Recommended Screenshots to Place Here:
1. `walkforward_regime_stability.png`
   - Stability of parameter selection across Train (<= Aug 31, 2026), Validation (Sep 01 - Nov 08, 2026), and Blind Test (Nov 09 - Nov 16, 2026).
2. `regime_filter_er12_distribution.png`
   - Distribution of Kaufman Efficiency Ratio (ER12) demonstrating how choppy consolidation zones were avoided.
3. `volatility_shock_filter_scatter.png`
   - Bar and jump shock ratio filter suppressing entries during flash spikes and spread blowouts.
4. `phase_progression_comparison.png`
   - Table or chart showing progression from early phases (Phase 2-8) through Kelly pyrmiding (Phase 12), SMC sweeps (Phase 15), and the final Hybrid Ensemble (Phase 18).

---
### Linking in Markdown Reports:
```markdown
![Walk-Forward Stability](screenshots/validation_phases/walkforward_regime_stability.png)
![Regime Filter ER12](screenshots/validation_phases/regime_filter_er12_distribution.png)
```
