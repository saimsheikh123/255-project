import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Load data
df = pd.read_csv('preprocessed_final.csv')
df['large_move_5d'] = (df['target_SP500_5d'].abs() > 0.01).astype(int)

# Prepare features (exclude ALL target columns including new binary one)
metadata_cols = ['date', 'next_trading_day', 'speaker']
target_cols = [c for c in df.columns if c.startswith('target_') or c.startswith('class_') or c.startswith('large_move')]
feature_cols = [c for c in df.columns if c not in metadata_cols + target_cols]

X = df[feature_cols]
y = df['large_move_5d']

# Split
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Baseline
baseline = max(y_test.value_counts()) / len(y_test)

print("="*60)
print("MODEL COMPARISON - Binary Large Movement (5d, >1%)")
print("="*60)

# Logistic Regression
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train_smote)
X_te_sc = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_tr_sc, y_train_smote)
y_pred_lr = lr.predict(X_te_sc)

acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr, average='weighted')
rec_lr = recall_score(y_test, y_pred_lr, average='weighted')
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')

print(f"\nLogistic Regression:")
print(f"  Accuracy:  {acc_lr:.4f}")
print(f"  Precision: {prec_lr:.4f}")
print(f"  Recall:    {rec_lr:.4f}")
print(f"  F1 Score:  {f1_lr:.4f}")
print(f"  vs Baseline: {(acc_lr-baseline)*100:+.2f}%")

# XGBoost
xgb_model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_model.fit(X_train_smote, y_train_smote)
y_pred_xgb = xgb_model.predict(X_test)

acc_xgb = accuracy_score(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb, average='weighted')
rec_xgb = recall_score(y_test, y_pred_xgb, average='weighted')
f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')

print(f"\nXGBoost:")
print(f"  Accuracy:  {acc_xgb:.4f}")
print(f"  Precision: {prec_xgb:.4f}")
print(f"  Recall:    {rec_xgb:.4f}")
print(f"  F1 Score:  {f1_xgb:.4f}")
print(f"  vs Baseline: {(acc_xgb-baseline)*100:+.2f}%")

# Random Forest (for comparison)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)  # Using unbalanced data like in notebook
y_pred_rf = rf.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, average='weighted')
rec_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

print(f"\nRandom Forest (original):")
print(f"  Accuracy:  {acc_rf:.4f}")
print(f"  Precision: {prec_rf:.4f}")
print(f"  Recall:    {rec_rf:.4f}")
print(f"  F1 Score:  {f1_rf:.4f}")
print(f"  vs Baseline: {(acc_rf-baseline)*100:+.2f}%")

print(f"\n{'='*60}")
print(f"Baseline (always predict majority): {baseline:.4f}")
print(f"{'='*60}")
