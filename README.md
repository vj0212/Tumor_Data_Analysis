# 🧬 Tumor\_Data\_Analysis

> An end-to-end computational biology project to predict high-risk tumor cases using gene expression and clinical features — with a focus on **Excel-based EDA**, **Python automation**, and **machine learning deployment**.

---

## 🚀 Project Overview

This project aims to **predict high-risk tumor patients** using a mix of **gene expression data** and **clinical variables**. The complete pipeline spans **manual data exploration in Excel**, **data ingestion and transformation via Python and MySQL**, and **model training with scikit-learn**, followed by a **Streamlit web app for deployment**.

---

## 🗂️ Dataset

* **Source**: [xenabrowser.net](https://xenabrowser.net/)
* **Samples**: 5,010 patient records
* **Target**: `High_Risk_Flag` — 1 for high-risk, 0 for low-risk
* **Important Features**:

  * `gene_A_zscore` (normalized gene expression)
  * `gene_B_expr`, `gene_C_expr_cleaned`
  * `tumor_size_imputed`
  * `age`

---
---

## 📊1. Excel-Based Data Exploration & Transformation

Excel served as the **primary tool for initial data cleaning, feature engineering, and exploratory analysis**, providing valuable visual insights before pipeline automation.

### 🧼 Data Cleaning with Power Query

* Imported raw biological data (\~5,000+ rows) into Power Query.
* Removed or flagged:

  * Rows with missing `patient_id`
  * `"unknown"` entries in `tumor_size_cm`
  * `-999` values in `gene_B_expr` (treated as invalid outliers)
  * Blank or missing `gene_A_expr` entries
* Standardized categorical values (e.g., `"MALIGNANT"` → `"malignant"`)
* Converted column data types to appropriate formats (numeric, date, text)

### 🧮 Feature Engineering

* **Z-Score Normalization**
  Computed `gene_A_zscore` using Excel formulas:

  ```
  =(gene_A_expr - MEAN) / STDEV.P
  ```

* **Risk Labeling**
  Defined a `High Risk` flag using logical rules:

  ```
  =IF(AND([diagnosis]="malignant", [tumor_size_imputed]>5), "High Risk", "Low Risk")
  ```

* **Age Group Bucketing**
  Classified patients into:

  * Young (<40)
  * Middle-aged (40–59)
  * Elderly (60+)

* **Outlier Detection & Missing Flags**

  * Marked extreme gene expression values (e.g., Z > 2)
  * Added binary flags for missing or abnormal entries

### 📊 Pivot Tables & Aggregated Analysis

Created pivot tables to summarize and analyze key metrics:

* Patient counts by `diagnosis` and `age_group`
* Average gene expression levels (`gene_A_expr`, `gene_B_expr`)
* Mean `tumor_size_cm` per demographic group
* High vs. Low Risk patient distribution

### 📈 Visual Dashboards & Charts

Interactive dashboards were built using Excel’s charting tools:

* **Pie Chart** → Diagnosis category breakdown
* **Bar Chart** → Mean gene expressions by diagnosis
* **Pivot Chart** → Tumor size vs. age group
* **KPI Cards** → Summarized key stats:

  * Total patients
  * % missing `gene_A_expr`
  * % with extreme Z-scores
  * Risk distribution
* **Slicers** enabled dynamic filtering by `age_group` and `sample_quality`
* **Heatmaps** (via conditional formatting) visualized missing data and gene intensity

### 📁 Export for Automation

* Final dataset exported as `Tumor_Data_Cleaned.csv`, including:

  * Cleaned and standardized features
  * Engineered columns (`gene_A_zscore`, `risk_flag`, etc.)
  * Ready for loading into MySQL, dbt modeling, and ML pipelines

> 🔍 These Excel-driven insights directly informed feature selection, data validation, and risk modeling strategy used in the automated pipeline.

---
---

2️⃣ Python: CSV → MySQL Ingestion
CSV File: Tumor_Data_Cleaned.csv

Steps Automated in Python:

Cleaned column names (removed spaces, hyphens, replaced with underscores)

Encoded special characters in MySQL password (e.g., @) using quote_plus

Connected to MySQL database bio_pipeline via SQLAlchemy + PyMySQL

Created/replaced table cleaned_biological_data

Used pandas.to_sql() to load cleaned data efficiently

Printed confirmation messages for successful execution

✅ Outcome: Raw data from Excel is now stored and queryable in MySQL.

3️⃣ dbt: SQL-Based Transformation
DBT Project: bio_dbt_project

Target Database: MySQL (bio_pipeline schema)

✅ What was done:
Configured profiles.yml to connect to MySQL using credentials

Built dbt model gene_summary.sql to:

Group data by Age_Group and High_Risk_Flag

Compute:

Total patients

Avg expression of gene_A, gene_B, gene_C

Avg tumor size

Registered sources and model metadata in schema.yml

Ran dbt run to compile and execute the transformation

✅ Output: Created a clean, query-ready view gene_summary in MySQL, suitable for Power BI dashboards or downstream modeling.

4️⃣ Windows Task Scheduler: Automation
Due to Airflow’s incompatibility with Python 3.13, a lightweight automation approach was used.

🧠 Automated Script: Automation.py
Handles the complete pipeline:

Loads latest cleaned CSV

Pushes data to MySQL (cleaned_biological_data)

Triggers dbt run to refresh gene_summary

Logs success/error messages with timestamps to automation_log.txt

🛠 .bat File Setup
Calls Automation.py using the local Python 3.13 path

Writes logs to the project directory

⏰ Scheduled Task Setup
Scheduled weekly run via Windows Task Scheduler

Triggers .bat file automatically every Monday at 2:00 AM

Manually tested and verified success through logs and dbt view refresh

✅ Outcome: Entire pipeline — from CSV to transformed dbt model — runs automatically and reliably every week.



---

## 🤖 5. Machine Learning Models

### Models Trained:

* Logistic Regression ✅ (best)
* Random Forest
* XGBoost
* Support Vector Machine (SVM)

### Evaluation Strategy:

* Train-test split: 80/20 with stratification
* Key metrics:

  * **ROC AUC**: 0.7826 (best with Logistic Regression)
  * **Precision (Class 1)**: 0.48
  * **Recall (Class 1)**: 0.34
  * **Confusion Matrix**:

    ```
    [[615, 105],
     [185,  97]]
    ```
* **Interpretability**:

  * Used **permutation importance** and **SHAP** for feature contribution analysis

### Deployment:

* Final model and preprocessor saved using `joblib`

  * `best_model_tuned.joblib`
  * `model_preprocessor.joblib`

---

## 🌐 6. Streamlit Web App

A lightweight frontend built using **Streamlit** for clinical use:

* Accepts **manual input** or **bulk CSV upload**
* Returns:

  * High/Low Risk prediction
  * Probability score
* Downloadable predictions as CSV
* Validates input format and handles missing columns

> App is responsive, runs locally, and deployable via Streamlit Cloud or Docker.

---

## 🛠️ Tech Stack

| Layer               | Tools Used                                       |
| ------------------- | ------------------------------------------------ |
| Data Exploration    | Microsoft Excel (EDA, Imputation, Visualization) |
| Data Storage        | MySQL                                            |
| Data Transformation | dbt (Data Build Tool)                            |
| Scripting & ML      | Python, Pandas, NumPy, Scikit-learn, XGBoost     |
| Automation          | Windows Task Scheduler                           |
| Deployment          | Streamlit, joblib                                |
| Interpretability    | SHAP, Permutation Importance                     |

---


---

## ⚠️ Challenges Faced

Throughout this project, several technical and analytical challenges were encountered:

* **Gene Expression Noise & Feature Redundancy**
  Outliers and varying scales across gene expression values required careful normalization using z-scores. Additionally, correlation analysis was necessary to identify and eliminate redundant genes that contributed little to predictive power.

* **Class Imbalance in Risk Prediction**
  Only \~28% of samples were labeled as high-risk, which initially skewed model performance — particularly recall. To address this, stratified train-test splits were applied. Although SMOTE was considered, it was excluded from the final pipeline for simplicity and interpretability.

* **Credential Handling in MySQL Connection**
  Special characters like `@` in database passwords caused connection issues. This was resolved by safely URL-encoding credentials using `quote_plus`.

* **Airflow Incompatibility with Python 3.13**
  Since Apache Airflow is not compatible with Python 3.13, a more lightweight solution was implemented using Windows Task Scheduler and `.bat` automation scripts.

* **Translating Excel Logic into SQL**
  Pivot-style summaries and aggregations originally done in Excel had to be carefully replicated in dbt's `gene_summary.sql` model to maintain consistency and accuracy.

* **Automation Logging & Monitoring**
  Debugging Task Scheduler runs was initially challenging. This was addressed by implementing detailed timestamped logging (`automation_log.txt`) and robust error handling in the Python script.

---

---

## 📁 Repository Structure

```bash
Tumor_Data_Analysis/
├── data/
│   └── cleaned_dataset.csv
├── notebooks/
│   └── eda_excel_summary.xlsx
├── dbt_project/
│   └── models/
├── app/
│   └── streamlit_app.py
├── models/
│   ├── best_model_tuned.joblib
│   └── model_preprocessor.joblib
├── utils/
│   └── data_loader.py
├── README.md
```

---

## 🧠 Key Takeaways

* Excel is powerful for **visual, intuitive EDA**, but full automation with Python/dbt enables **scalability**.
* In medical ML tasks, **recall and interpretability** matter more than accuracy alone.
* Deployment isn’t just model prediction — it includes **data validation, input UX, and continuous automation**.

---


