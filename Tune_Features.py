import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# === Load saved objects ===
X_train = joblib.load("X_train.joblib")
y_train = joblib.load("y_train.joblib")
X_test = joblib.load("X_test.joblib")
y_test = joblib.load("y_test.joblib")
preprocessor = joblib.load("model_preprocessor.joblib")

print("[INFO] Loaded training and testing data.")

# === Step 1: Baseline Logistic Regression ===
base_model = LogisticRegression(max_iter=1000, random_state=42)
base_model.fit(X_train, y_train)
base_auc = roc_auc_score(y_test, base_model.predict_proba(X_test)[:, 1])
print(f"[BASELINE] ROC AUC: {base_auc:.4f}")

# === Step 2: Permutation Feature Importance ===
print("\n🔍 Performing permutation feature importance...")
feature_names = preprocessor.transformers_[0][2]  # ColumnTransformer -> 'num' -> columns
perm_result = permutation_importance(base_model, X_train, y_train, scoring='roc_auc', n_repeats=30, random_state=42)

importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance_Mean': perm_result['importances_mean'],
    'Importance_Std': perm_result['importances_std']
}).sort_values(by='Importance_Mean', ascending=False)

print("\n📊 Permutation Importance:")
print(importances_df)

# OPTIONAL: You could drop features with Importance_Mean < threshold if needed

# === Step 3: Hyperparameter Tuning using GridSearchCV ===
print("\n🎯 Running GridSearchCV on Logistic Regression...")

param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'solver': ['liblinear', 'lbfgs'],
    'penalty': ['l2']
}

grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# === Best Model & Evaluation ===
best_model = grid_search.best_estimator_
best_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

print(f"\n✅ Best Hyperparameters: {grid_search.best_params_}")
print(f"🎯 ROC AUC after tuning: {best_auc:.4f}")

# === Save the tuned model ===
joblib.dump(best_model, "best_model_tuned.joblib")
print("💾 Saved best tuned model as 'best_model_tuned.joblib'")
