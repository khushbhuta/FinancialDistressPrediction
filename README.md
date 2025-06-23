# Predictive Modeling for Financial Distress

This repository contains the complete workflow for a machine learning project titled **"Predictive Modeling for Financial Distress"**, aimed at predicting company bankruptcy based on financial data spanning 10 years (2014–2024). The project covers data preprocessing, feature engineering (including Altman Z-Score and leverage ratios), imbalance handling using SMOTE, and classification using Random Forest and XGBoost. Performance is evaluated using precision, recall, F1-score, and confusion matrices with a strong focus on minority-class prediction.

---

## 🚀 Objective

The primary goal of this project is to build a robust classifier that can identify companies under financial distress using historical financial indicators. Companies are recorded over a 10-year window, with each year representing a row. The final binary label indicates whether the company is likely to go bankrupt. This model could be deployed by investors, financial institutions, or regulators for early-warning systems.

---

## 📃 Dataset Overview

* **Source:** Proprietary dataset containing financial records of Indian companies.
* **Timeframe:** 2014 – 2024
* **Structure:** Each company appears 10 times (one per year), with 82+ columns of financial metrics.
* **Label:** Binary (1 for financial distress, 0 for healthy)

Each row contains values such as total income, sales, profit after tax, capital structure, liabilities, and various turnover and liquidity ratios.

---

## 📆 Project Steps & Methodology

### 1. **Data Preprocessing**

#### a. Row-to-Column Transformation

Since each company appears in 10 rows (once per year), the dataset was pivoted to convert this long-form data into wide-form:

```python
# Assign year numbers
df["Year"] = df.groupby("Name").cumcount() + 1

# Pivot the table
df_pivot = df.pivot(index="Name", columns="Year")
df_pivot.columns = [f"{feature}_{year}" for feature, year in df_pivot.columns]
df_pivot.reset_index(inplace=True)
```

This transformation enables the model to view a company’s entire 10-year financial history in one row, resulting in \~800 columns per sample.

#### b. Label Aggregation Strategy

To shift the task to per-company prediction:

* Companies were labeled as **bankrupt** (1) if they were in distress for 3 or more out of 10 years.
* Otherwise, they were labeled as **non-distressed** (0).

This aggregation balances long-term patterns while minimizing short-term noise.

#### c. Missing Value & Infinity Handling

* Infinite values were replaced with `NaN`.
* `NaN` values were filled using median imputation.
* Columns were filtered to retain only relevant financial indicators.

```python
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)
```

---

### 2. **Feature Engineering**

#### a. Altman Z-Score Components

Five financial ratios were engineered based on the Altman Z-Score formula:

* **Working Capital / Total Assets**
* **Retained Earnings / Total Assets**
* **PBIT / Total Assets**
* **Market Value of Equity / Total Liabilities**
* **Net Sales / Total Assets**

These were implemented with:

```python
df["Altman_Z"] = (
    1.2 * df["Net working capital"] / df["Total assets"] +
    1.4 * df["Retained profits/losses during the year"] / df["Total assets"] +
    3.3 * df["PBIT"] / df["Total assets"] +
    0.6 * df["Market Value Equity"] / df["Total liabilities"] +
    1.0 * df["Net sales"] / df["Total assets"]
)
```

#### b. Debt-Related Ratios

To improve leverage visibility, the following ratios were added:

* Debt / Total Assets
* Debt / Equity
* Debt / Net Income
* Debt / Net Sales

These features showed a stronger correlation with the bankruptcy label (>0.4 in some cases), helping boost model recall.

---

### 3. **Class Imbalance Handling**

#### a. SMOTE Oversampling

The original dataset was highly imbalanced. To resolve this, **SMOTE (Synthetic Minority Oversampling Technique)** was applied *after train-test split*:

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(k_neighbors=2, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```

#### b. Class Weights

Additionally, classifiers like Random Forest and XGBoost were trained with `class_weight='balanced'` or `scale_pos_weight` to help penalize false negatives more.

---

### 4. **Modeling and Evaluation**

#### a. GridSearchCV for Hyperparameter Tuning

Exhaustive tuning was performed using `GridSearchCV` on 5-fold cross-validation to optimize:

* For **Random Forest**:

  * `n_estimators`, `max_depth`, `min_samples_split`, `class_weight`

* For **XGBoost**:

  * `learning_rate`, `n_estimators`, `max_depth`, `subsample`, `colsample_bytree`, `scale_pos_weight`

```python
grid_search = GridSearchCV(pipeline, param_grid, scoring="f1_macro", cv=5)
grid_search.fit(X_train, y_train)
```

#### b. Classifiers Used

* **Random Forest (before and after SMOTE)**
* **XGBoost** (final optimized model)

---

## 📊 Results Summary

| Model Version             | Accuracy   | Precision  | Recall     | F1 Score   |
| ------------------------- | ---------- | ---------- | ---------- | ---------- |
| Random Forest (pre-SMOTE) | 93.89%     | 55.31%     | 85.65%     | 57.19%     |
| **XGBoost (optimized)**   | **96.90%** | **63.58%** | **88.04%** | **69.57%** |

* **Recall** improved by **+32%** from baseline RF to final XGBoost.
* **F1 Score** improved from **57.19%** → **69.57%**, making the model far more useful in real-world bankruptcy detection.

---

## 🌎 Visualizations & Interpretations

### Confusion Matrix (XGBoost)

* True Positives (Bankrupt correctly predicted): High
* False Negatives: Greatly reduced

### Precision-Recall Curve

* XGBoost maintained good balance between detecting bankruptcies and avoiding false alarms.

---

## 🔄 Pipeline Architecture

```python
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

pipeline = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(k_neighbors=2)),
    ("classifier", XGBClassifier(...))
])
```

This ensured all preprocessing (scaling, balancing, fitting) was consistently applied.

---

## ✅ Key Achievements

1. **Recall improved from 55.19% → 88.04%**
2. **Achieved F1 Score of 69.57%** on minority class
3. **Feature vector expanded using Altman Z-Score and debt ratios**
4. **Trained and tuned XGBoost and Random Forest using GridSearchCV**

---

## 🎓 Learnings & Future Work

* **Feature Optimization**: Including financial theory-driven metrics like Altman Z-Score adds substantial value.
* **Model Selection**: XGBoost significantly outperformed Random Forest in recall and F1.
* **Imbalance Techniques**: SMOTE was critical for boosting minority class performance.

### Future Directions

* Implement LSTM/GRU for time-series learning across years.
* Add external indicators: market sentiment, GDP, inflation.
* Try ensemble stacking to combine different model families.

---

## 👥 Authors

* **Khush Bhuta**
  BITS Pilani, Hyderabad Campus
  \[2022A7PS1333H]

---

## 📁 Repository Structure

* `/notebooks/`: Exploratory data analysis, feature engineering
* `/models/`: Trained models, tuning scripts
* `/data/`: Input CSVs, transformed datasets
* `/results/`: Evaluation metrics, confusion matrices, plots

---

## 👤 Contact

If you'd like to collaborate or have questions, feel free to reach out!

---

## ✉️ License

MIT License © 2025 Khush Bhuta
