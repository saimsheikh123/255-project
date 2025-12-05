# Fed Speech Market Movement Prediction

Predicting stock market movements from Federal Reserve speeches using NLP and machine learning (1996-2020).

## 🎯 Project Overview

This project analyzes 1,785 Federal Reserve speeches to predict whether they cause significant stock market movements. Using advanced NLP models (FinBERT, FOMC-RoBERTa, BERTopic) and multiple ML classifiers, we achieved **55.18% accuracy** with Logistic Regression, beating the naive baseline by +0.28%.

## 📊 Results Summary

| Model | Accuracy | vs Baseline |
|-------|----------|-------------|
| **Logistic Regression** | **55.18%** | **+0.28%** ✅ |
| Random Forest | 52.66% | -2.24% |
| XGBoost | 52.10% | -2.80% |
| Baseline (Majority) | 54.90% | - |

**Key Insight**: Simple linear models outperform complex tree-based models on noisy financial data.

## 📁 Project Structure

```
255-project/
├── data/                          # Raw and processed datasets
│   ├── fed_speeches_1996_2020.csv
│   ├── index_prices_1996_2020.csv
│   ├── preprocessed_final.csv
│   └── ...
├── notebooks/                     # Jupyter notebooks
│   ├── preprocessing_final.ipynb  # Data preprocessing pipeline
│   ├── random_forest.ipynb        # Random Forest experiments
│   ├── logistic_regression.ipynb  # Logistic Regression (best model)
│   └── xgboost_training.ipynb     # XGBoost experiments
├── models/                        # Trained models
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── xgboost_model.json
├── results/                       # Model outputs
│   ├── *_predictions.csv
│   ├── *_feature_importance.csv
│   └── ...
├── scripts/                       # Utility scripts
│   └── test_models.py
├── MODEL_PERFORMANCE_REPORT.md    # Detailed analysis
└── README.md
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn
pip install transformers torch yfinance bertopic
```

### Run Model Comparison
```bash
cd scripts
python test_models.py
```

### Explore Notebooks
1. **Data Preprocessing**: `notebooks/preprocessing_final.ipynb`
2. **Best Model**: `notebooks/logistic_regression.ipynb`
3. **Alternative Models**: `notebooks/random_forest.ipynb`, `notebooks/xgboost_training.ipynb`

## 🔬 Methodology

### Data Pipeline
1. **Fed Speeches** (1996-2020): 1,785 speeches from FRED
2. **Market Data**: S&P 500, DJIA, NASDAQ, Russell 2000 (yfinance)
3. **NLP Processing**:
   - FinBERT: Financial sentiment analysis
   - Loughran-McDonald: Financial polarity scoring
   - FOMC-RoBERTa: Hawkish/dovish classification
   - BERTopic: Topic modeling
4. **Feature Engineering**: 49 features (sentiment, topics, market context, macro indicators)

### Target Evolution
- **Initial**: 3-class next-day direction → **57.7% (-7.28% vs baseline)**
- **Iteration 1**: 3-class 5-day direction → **48.74% (-3.64% vs baseline)**
- **Iteration 2**: Binary large movement (>1%) → **52.66% (-2.24% vs baseline)**
- **Final**: Logistic Regression + SMOTE → **55.18% (+0.28% vs baseline)** ✅

### Key Improvements
1. **Target Reformulation** (+5.04%): Switched from directional to magnitude prediction
2. **SMOTE Oversampling**: Balanced training classes (849:849)
3. **Model Selection** (+2.52%): Logistic Regression outperformed complex models
4. **Feature Scaling**: StandardScaler normalization for linear models

## 📈 Feature Importance

Top predictive features:
1. **Market Context (59.8%)**: Recent volatility and returns
2. **Sentiment (14.5%)**: FinBERT scores, hawkish/dovish signals
3. **Topics (13.7%)**: BERTopic-derived themes

**Insight**: Model learned that volatility begets volatility (backward-looking pattern).

## 🎓 Key Learnings

1. **Efficient Markets**: Public Fed speeches have minimal predictive power
2. **Simple > Complex**: Linear models generalize better than ensemble methods on noisy data
3. **Target Matters**: Problem formulation impacts results more than algorithm choice
4. **Realistic Expectations**: 0.28% improvement validates market efficiency

## 📄 Detailed Report

See [MODEL_PERFORMANCE_REPORT.md](MODEL_PERFORMANCE_REPORT.md) for:
- Complete evolution timeline (-7.28% → +0.28%)
- Technical model comparisons
- Why Logistic Regression won
- Limitations and honest assessment

## 🛠️ Technologies

- **Python 3.11**
- **ML**: scikit-learn, XGBoost, imbalanced-learn
- **NLP**: Transformers (FinBERT, FOMC-RoBERTa), BERTopic
- **Data**: pandas, numpy, yfinance
- **Visualization**: matplotlib, seaborn

## 📊 Dataset Sources

- Fed Speeches: FRED Economic Data
- Stock Prices: Yahoo Finance (yfinance)
- Sentiment Dictionaries: Loughran-McDonald Master Dictionary

## 📝 Citation

If you use this work, please cite:
```
Fed Speech Market Movement Prediction
CMPE 255 - Data Mining Project
San Jose State University, 2025
```

## 👤 Author

Saim - CMPE 255 Data Mining Project

## 📜 License

This project is for educational purposes (CMPE 255 coursework).

---

**Note**: The 0.28% improvement over baseline demonstrates the extreme difficulty of predicting markets from public information, validating the Efficient Market Hypothesis rather than indicating modeling failure.
