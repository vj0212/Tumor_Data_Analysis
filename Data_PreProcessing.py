import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib

# === CONFIGURATION ===
db_user = 'VJ07'
db_password = quote_plus('AV0212@MUJ')
db_host = 'localhost'
db_port = 3306
db_name = 'bio_pipeline'
table_name = 'cleaned_biological_data'

# === CONNECT TO MYSQL & LOAD DATA ===
try:
    engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
    with engine.connect() as connection:
        print("[INFO] Database connection successful!")
except Exception as e:
    raise ConnectionError(f"Failed to connect to database: {e}")

query = f"""
    SELECT gene_A_zscore, gene_B_expr, gene_C_expr_cleaned, tumor_size_imputed, age, High_Risk_Flag
    FROM {table_name}
"""

df = pd.read_sql(query, engine)
print(f"[INFO] Loaded {len(df)} records from MySQL.")

# === FIX non-numeric error values and standardize missingness ===
df.replace(to_replace=['#DIV/0!', 'N/A', 'na', 'NA', 'null', 'NULL', 'NaN'], value=np.nan, inplace=True)

# Ensure all numerical columns are floats
numeric_cols = ['gene_A_zscore', 'gene_B_expr', 'gene_C_expr_cleaned', 'tumor_size_imputed', 'age']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# === CLEANING & ENCODING TARGET ===
print(f"[DEBUG] Unique values before mapping:\n{df['High_Risk_Flag'].unique()}")
df['High_Risk_Flag'] = df['High_Risk_Flag'].map({'High Risk': 1, 'Low Risk': 0})
print(f"[DEBUG] Rows before dropping nulls: {len(df)}")
df.dropna(subset=['High_Risk_Flag'], inplace=True)
print(f"[DEBUG] Rows after dropping nulls: {len(df)}")

if df.empty:
    raise ValueError("All rows dropped due to null High_Risk_Flag. Check data quality.")

# === DEFINE FEATURES & TARGET ===
X = df[numeric_cols]
y = df['High_Risk_Flag']

# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === PREPROCESSING PIPELINE ===
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols)
])

# === FIT PREPROCESSOR ON TRAINING DATA ===
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"[SUCCESS] Data split and preprocessing complete.")
print(f"[INFO] X_train shape: {X_train_processed.shape}, X_test shape: {X_test_processed.shape}")

# === SAVE ARTIFACTS FOR REUSE ===
joblib.dump(preprocessor, 'model_preprocessor.joblib')
joblib.dump(X_train_processed, 'X_train.joblib')
joblib.dump(X_test_processed, 'X_test.joblib')
joblib.dump(y_train, 'y_train.joblib')
joblib.dump(y_test, 'y_test.joblib')
print("[INFO] Preprocessor and data splits saved successfully.")
