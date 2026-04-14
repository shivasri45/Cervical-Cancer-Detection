# 🧬 Cervical Cancer Detection System

> ⚠️ **Disclaimer:** This is a screening aid, not a diagnostic tool. All results should be reviewed by a qualified medical professional before any clinical decision is made.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-Research%20Only-lightgrey?style=flat-square)

An ML-powered **Streamlit** web application for cervical cancer risk screening using patient demographics, behavioral habits, and medical history data. The app provides a complete end-to-end pipeline — from raw data upload to interactive patient screening.

---

## ✨ Features

| Tab | Name | Description |
|-----|------|-------------|
| 1 | 📂 Patient Data | Upload CSV/Excel, select diagnosis target, view PCA projection |
| 2 | 📊 EDA | Distributions, missing value analysis, statistical profiling |
| 3 | 🧹 Cleaning | Imputation (mean/median/most_frequent) + outlier removal (IQR or Isolation Forest) |
| 4 | 🎯 Risk Factors | Correlation-based feature selection with adjustable threshold |
| 5 | ✂️ Data Split | Train/test split with optional StandardScaler normalization |
| 6 | ⚙️ Model Setup | Choose Random Forest, SVM, or Logistic Regression |
| 7 | 🏋️ Train & Validate | K-Fold cross-validation + feature importance chart |
| 8 | 📈 Diagnostics | Accuracy metrics, classification report, confusion matrix |
| 9 | 🔧 Tuning | GridSearchCV or RandomizedSearchCV hyperparameter optimization |
| 10 | 🩺 **Patient Screening** | **Interactive form → risk prediction + class probabilities** |
| 📦 | Export | Download trained model (.pkl) and test predictions (.csv) |

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/cervical-cancer-detection.git
cd cervical-cancer-detection

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run appV6.py
```

Opens at **http://localhost:8501**

---

## 📦 Dependencies

```
streamlit
pandas
numpy
scikit-learn
plotly
matplotlib
seaborn
joblib
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🗂️ Dataset

Designed for the **UCI Cervical Cancer (Risk Factors) Dataset**, which includes:

- Age, number of sexual partners, age at first intercourse
- Number of pregnancies
- Smoking history and duration
- Hormonal contraceptive and IUD use
- STD history and diagnoses
- Diagnosis targets: `Biopsy`, `Hinselmann`, `Schiller`, `Citology`

🔗 [UCI ML Repository — Cervical Cancer Risk Factors](https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29)

> The app also accepts any CSV or Excel file with similar tabular structure.

---

## 🤖 Model Comparison

| Model | Best For | Notes |
|-------|----------|-------|
| ⭐ **Random Forest** | General use | Recommended default. Handles feature interactions and noisy data well. |
| **SVM** | High-dimensional data | Enable StandardScaler in Tab 5 for best results. |
| **Logistic Regression** | Clinical reporting | Most interpretable. Good for understanding feature contributions and odds ratios. |

---

## 🔄 Recommended Workflow

Follow the tabs in order for best results:

1. **Upload** your dataset → select `Biopsy` (or similar) as the diagnosis target
2. **Explore** data in the EDA tab — check distributions and missing values
3. **Clean** — apply imputation *before* outlier removal
4. **Select features** — adjust the correlation threshold until 5–15 features remain
5. **Split** — 80/20 is recommended; enable scaling if using SVM
6. **Choose** Random Forest as a strong starting model
7. **Train** with 5-fold cross-validation; review feature importances
8. **Check diagnostics** — a train/test gap > 0.15 indicates overfitting
9. **Tune** hyperparameters to squeeze out extra performance (optional)
10. **Screen patients** interactively using the risk factor input form
11. **Export** the model and predictions for reporting

---

## 🏗️ Project Structure

```
cervical-cancer-detection/
├── appV6.py            # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🔩 Technical Notes

- **Missing values** — UCI dataset encodes missing as `?`; handled automatically on file load.
- **Feature encoding** — `pd.get_dummies(drop_first=True)` applied before splitting; prediction inputs are aligned to training columns to prevent shape mismatches.
- **Target encoding** — `LabelEncoder` applied automatically when the diagnosis column contains string labels.
- **State management** — Streamlit session state persists pipeline steps; upstream changes trigger downstream resets to prevent stale model artifacts.
- **Scaler handling** — If StandardScaler is used during training, it is saved to session state and applied consistently at prediction time.

---

## ⚖️ License

For **educational and research purposes only**. Not for clinical use without appropriate validation and regulatory approval.
