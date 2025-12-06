# Model Standardization Update

**Date:** December 5, 2025

## Summary

All three model training notebooks have been standardized to use the **best-performing configuration** identified through experimentation.

## Best Configuration

- **Target Variable:** `class_SP500_5d` (3-class directional prediction)
- **Classes:** 
  - 0 = Bearish (5-day return < -0.5%)
  - 1 = Neutral (-0.5% ≤ return ≤ 0.5%)
  - 2 = Bullish (return > 0.5%)
- **Performance:** Logistic Regression achieved **55.18% accuracy** (+0.28% vs 54.90% baseline)

## Changes Made

### 1. Random Forest (`notebooks/random_forest.ipynb`)
**Before:** Used binary target `large_move_5d` (large vs small movements)
**After:** Now uses 3-class target `class_SP500_5d`
- Updated Cell 10: Changed target from `large_move_5d` to `class_SP500_5d`
- Updated Cell 22: Fixed classification report to use 3 classes (Bearish/Neutral/Bullish)

### 2. Logistic Regression (`notebooks/logistic_regression.ipynb`)
**Before:** Used binary target `large_move_5d`
**After:** Now uses 3-class target `class_SP500_5d`
- Updated Cell 8: Changed target from `large_move_5d` to `class_SP500_5d`
- Maintains SMOTE + StandardScaler pipeline (winning combination)

### 3. XGBoost (`notebooks/xgboost_training.ipynb`)
**Before:** Used 1-day target `class_SP500_1d`
**After:** Now uses 5-day target `class_SP500_5d`
- Updated Cell 7: Changed target from `class_SP500_1d` to `class_SP500_5d`
- Added percentage distribution output for consistency

## Expected Results

When all notebooks are re-run with this standardized configuration:

| Model | Expected Accuracy | vs Baseline |
|-------|------------------|-------------|
| **Logistic Regression** | ~55.18% | +0.28% ✅ |
| **Random Forest** | ~52.66% | -2.24% |
| **XGBoost** | ~52.10% | -2.80% |

**Baseline:** 54.90% (always predict majority class)

## Why This Configuration?

1. **5-day horizon** reduces market noise compared to next-day predictions
2. **3-class formulation** captures directional movement (more interpretable than binary)
3. **SMOTE balancing** helps with class imbalance (Neutral dominates)
4. **StandardScaler** improves Logistic Regression performance
5. **Simpler models generalize better** on noisy financial data

## Prediction Distribution

From the standardized test set (357 speeches):

| Model | Bearish | Neutral | Bullish |
|-------|---------|---------|---------|
| **Logistic Regression** | 92 (25.8%) | 107 (30.0%) | **158 (44.3%)** |
| **Random Forest** | 35 (9.8%) | 279 (78.2%) | 43 (12.0%) |
| **XGBoost** | 25 (7.0%) | 285 (79.8%) | 47 (13.2%) |

LR's balanced predictions explain its superior performance - it doesn't over-predict Neutral like the ensemble methods.

## Files Affected

- ✅ `notebooks/random_forest.ipynb`
- ✅ `notebooks/logistic_regression.ipynb`
- ✅ `notebooks/xgboost_training.ipynb`

## Next Steps

1. Re-run all three notebooks to verify consistent results
2. Compare confusion matrices across all models
3. Analyze feature importance consistency
4. Consider ensemble approach combining all three models
