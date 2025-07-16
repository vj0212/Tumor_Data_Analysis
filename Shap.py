import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === Load trained model and data ===
model = joblib.load("best_model_tuned.joblib")
X_test = joblib.load("X_test.joblib")
X_train = joblib.load("X_train.joblib")

# Define feature names based on original preprocessing order
feature_names = ['gene_A_zscore', 'gene_B_expr', 'gene_C_expr_cleaned', 'tumor_size_imputed', 'age']

# Ensure X_test and X_train are converted to DataFrames with the correct feature names
X_test_df = pd.DataFrame(X_test, columns=feature_names)
X_train_df = pd.DataFrame(X_train, columns=feature_names)

# Print first few rows to verify
print("X_test_df head:\n", X_test_df.head())

# === Create SHAP explainer using X_train as background ===
explainer = shap.Explainer(model, X_train_df)
shap_values = explainer(X_test_df)

# === Print SHAP values to debug ===
print("SHAP Values Shape:", shap_values.values.shape)
print("\nSHAP Values by Feature (Mean Absolute):")
for i, feature in enumerate(feature_names):
    mean_abs_shap = np.mean(np.abs(shap_values.values[:, i]))
    print(f"{feature}: {mean_abs_shap:.4f}")

# === Summary plot ===
plt.figure()
shap.summary_plot(shap_values, X_test_df, plot_type="bar", feature_names=feature_names, show=False)
plt.tight_layout()
plt.show()
plt.close()

# === Full summary plot with feature values ===
plt.figure()
shap.summary_plot(shap_values, X_test_df, feature_names=feature_names, show=False)
plt.tight_layout()
plt.show()
plt.close()