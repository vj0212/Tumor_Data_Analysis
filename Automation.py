import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import subprocess
import os
from datetime import datetime

# === CONFIGURATION ===
db_user = 'VJ07'
db_password = quote_plus('AV0212@MUJ')  # URL-encode for special characters
db_host = 'localhost'
db_port = 3306
db_name = 'bio_pipeline'
table_name = 'cleaned_biological_data'
csv_path = r'C:\Users\jhavi\OneDrive\Desktop\Tumor_Data_Cleaned.csv'  # ✅ Adjust path
dbt_project_path = r'C:/Users/jhavi/OneDrive/Desktop/bio_dbt_project'

# === STEP 1: Load CSV ===
try:
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded CSV with {len(df)} rows and {len(df.columns)} columns.")
except Exception as e:
    print("[ERROR] Failed to load CSV:", e)
    exit(1)

# === STEP 2: Clean column names ===
df.columns = [col.strip().replace(' ', '_').replace('-', '_') for col in df.columns]

# === STEP 3: Load into MySQL ===
engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        df.to_sql(table_name, con=engine, if_exists='replace', index=False)
        print(f"[INFO] Data loaded into MySQL table '{table_name}'")
except Exception as e:
    print("[ERROR] MySQL operation failed:", e)
    exit(1)

# === STEP 4: Run dbt ===
try:
    print("[INFO] Running dbt models...")
    result = subprocess.run(["dbt", "run"], cwd=dbt_project_path, shell=True, capture_output=True, text=True)
    print("[INFO] dbt output:")
    print(result.stdout)
    if result.returncode != 0:
        print("[ERROR] dbt run failed with error:")
        print(result.stderr)
        exit(1)
except Exception as e:
    print("[ERROR] Failed to run dbt:", e)
    exit(1)

# === STEP 5: Log Completion ===
print(f"[SUCCESS] Pipeline completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
