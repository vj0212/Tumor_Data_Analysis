import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# === App Configuration ===
st.set_page_config(page_title="Biological Risk Prediction", layout="wide")

# === Sidebar ===
st.sidebar.title("Biological Risk Prediction")
st.sidebar.markdown("""
This application predicts high-risk biological outcomes based on gene expression and clinical features.
Enter values manually or upload a CSV file for batch predictions.
""")
st.sidebar.subheader("Instructions")
st.sidebar.markdown("""
1. **Manual Input**: Enter values for Gene A Z-Score, Gene B Expression, Gene C Expression, Tumor Size, and Age.
2. **Batch Input**: Upload a CSV file with columns: `gene_A_zscore`, `gene_B_expr`, `gene_C_expr_cleaned`, `tumor_size_imputed`, `age`.
3. Click "Predict" to view results.
4. Download predictions as needed.
""")

# === Main Page ===
st.title("Biological Risk Prediction Dashboard")
st.markdown("Predict high-risk outcomes using gene expression and clinical data.")

# === Load Model and Preprocessor ===
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("best_model_tuned.joblib")
        preprocessor = joblib.load("model_preprocessor.joblib")
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model or preprocessor: {e}")
        return None, None

model, preprocessor = load_artifacts()
if model is None or preprocessor is None:
    st.stop()

# === Define Feature Names ===
feature_names = ['gene_A_zscore', 'gene_B_expr', 'gene_C_expr_cleaned', 'tumor_size_imputed', 'age']

# === Input Section ===
st.subheader("Input Data")
input_method = st.radio("Choose input method:", ("Manual Input", "Upload CSV"))

# === Manual Input ===
if input_method == "Manual Input":
    with st.form(key="manual_input_form"):
        col1, col2 = st.columns(2)
        with col1:
            gene_a = st.number_input("Gene A Z-Score", value=0.0, step=0.1, format="%.4f")
            gene_b = st.number_input("Gene B Expression", value=0.0, step=0.1, format="%.4f")
            gene_c = st.number_input("Gene C Expression", value=0.0, step=0.1, format="%.4f")
        with col2:
            tumor_size = st.number_input("Tumor Size (mm)", min_value=0.0, value=0.0, step=0.1, format="%.1f")
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=30, step=1)
        submit_button = st.form_submit_button("Predict")

    if submit_button:
        if any(np.isnan([gene_a, gene_b, gene_c, tumor_size, age])):
            st.error("Please provide valid numerical inputs.")
        else:
            input_data = pd.DataFrame(
                [[gene_a, gene_b, gene_c, tumor_size, age]],
                columns=feature_names
            )
            try:
                processed_data = preprocessor.transform(input_data)
                prediction = model.predict(processed_data)[0]
                prob = model.predict_proba(processed_data)[0][1]

                st.success(f"🧬 Prediction: {'High Risk' if prediction == 1 else 'Low Risk'} (Probability: {prob:.2f})")

                # Download Prediction
                result = {"Prediction": "High Risk" if prediction == 1 else "Low Risk", "Probability": prob}
                result_df = pd.DataFrame([result])
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Prediction",
                    data=csv,
                    file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Prediction error: {e}")

# === CSV Upload ===
else:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if not all(col in df.columns for col in feature_names):
                st.error(f"CSV must contain columns: {', '.join(feature_names)}")
            else:
                processed_data = preprocessor.transform(df)
                predictions = model.predict(processed_data)
                probs = model.predict_proba(processed_data)[:, 1]

                result_df = df.copy()
                result_df["Prediction"] = ["High Risk" if p == 1 else "Low Risk" for p in predictions]
                result_df["Probability"] = probs

                st.write("Batch Predictions:")
                st.dataframe(result_df)

                # Download batch predictions
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Batch Predictions",
                    data=csv,
                    file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# === Footer ===
st.markdown("---")
st.markdown("Developed with Streamlit and scikit-learn. For support, contact 'jhavikram533@gmail.com' .")