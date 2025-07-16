import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

# === Load preprocessed datasets ===
X_train = joblib.load("X_train.joblib")
X_test = joblib.load("X_test.joblib")
y_train = joblib.load("y_train.joblib")
y_test = joblib.load("y_test.joblib")

# === Initialize models ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

results = {}

# === Train & Evaluate ===
for name, model in models.items():
    print(f"\n🚀 Training {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    print(f"ROC AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    results[name] = {
        "model": model,
        "roc_auc": auc
    }

# === Identify Best Model ===
best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
best_model = results[best_model_name]['model']
print(f"\n✅ Best Model: {best_model_name} (ROC AUC = {results[best_model_name]['roc_auc']:.4f})")

# === Save Best Model ===
joblib.dump(best_model, "best_model.joblib")
print("💾 Saved best model as 'best_model.joblib'")
