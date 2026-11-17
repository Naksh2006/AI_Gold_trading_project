# Machine Learning Model Performance & Calibration Screenshots

This directory stores statistical plots and metric visual proofs for the AI models (L2-LR, Tanh MLP, GBDT, Hybrid Ensemble).

### Recommended Screenshots to Place Here:
1. `roc_auc_curve.png`
   - Receiver Operating Characteristic (ROC) curve comparing the individual and ensemble classifiers on clean out-of-sample directional moves.
2. `precision_recall_curve.png`
   - Precision-Recall curve showing confidence thresholding vs trade precision.
3. `calibration_brier_curve.png`
   - Probability reliability diagram showing empirical probability vs predicted confidence (Brier Score = 0.2319).
4. `feature_importance_ranking.png`
   - Bar chart displaying the 12 standardized features sorted by combined institutional weights (e.g., F12 Cross-Asset Beta, F4 VWAP Deviation, F8 RSI).
5. `confusion_matrix.png`
   - Normalized confusion matrix showing classification precision and recall on out-of-sample test splits.

---
### Linking in Markdown Reports:
```markdown
![ROC-AUC Comparison](screenshots/ml_model_stats/roc_auc_curve.png)
![Feature Importance](screenshots/ml_model_stats/feature_importance_ranking.png)
```
