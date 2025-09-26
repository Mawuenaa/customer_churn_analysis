# 🚀 Telco Customer Churn Prediction

Predicting customer churn for a fictional telecom company using machine-learning techniques.

---

## 📊 Dataset
**Source**: [IBM Sample Data Sets](https://www.kaggle.com/blastchar/telco-customer-churn)  
**Shape**: 7,043 customers × 21 features  
**Target**: `Churn` – whether the customer left within the last month (`Yes`/`No`)

---

## 🧪 Models Benchmarked
| Model | Accuracy | Notes |
|-------|----------|-------|
| Random Forest | 0.7913 | Baseline with default settings |
| XGBoost | 0.7970 | Fast, good out-of-the-box |
| LightGBM | 0.8062 | Slightly better than XGBoost |
| ExtraTrees (tuned) | **0.8148** | Best score after hyper-parameter search |

---

## 🔧 Pre-processing Pipeline
1. **Numeric**:
   - `tenure`, `MonthlyCharges`, `TotalCharges` → `StandardScaler`
2. **Categorical**:
   - 16 binary & multi-class columns → `OneHotEncoder(sparse=False)`
3. **Extras**:
   - `TotalCharges` blanks → imputed to `0`
   - `Churn` → encoded to `0/1`

---

## ⚙️ Hyper-parameter Tuning
**Algorithm**: `RandomizedSearchCV` (10 iter × 5-fold CV)  
**Estimator**: `ExtraTreesClassifier`  
**Best params**:
```python
{'n_estimators': 50,
 'min_samples_split': 5,
 'min_samples_leaf': 4,
 'max_features': 'log2'}
```
---

## 📈 Feature Importance (Top 2)
| Feature  | Importance |
| -------- | ---------- |
| Contract | 0.2026     |
| tenure   | 0.1194     |


## 🚀 Quick Start
1. Clone the repo

git clone https://github.com/Mawuenaa/telco-churn-ml.git
cd telco-churn-ml

2. Create & activate a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

3. Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

4. Launch Jupyter

jupyter notebook notebooks/Telco_Churn_Prediction.ipynb

## 📁 Repository Structure
```python
telco-churn-ml/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # original dataset
│
├── notebooks/
│   └── Telco_Churn_Prediction.ipynb           # full end-to-end pipeline
│
├── README.md
├── requirements.txt
└── .gitignore                                 # keeps data/ out of repo (optional)
```

## 🧪 Reproducing the Results
The notebook is self-contained:

| Step | Description                                              |
| ---- | -------------------------------------------------------- |
| 1    | Loads & cleans the data                                  |
| 2    | Trains 3 baseline models (RF, XGB, LGBM)                 |
| 3    | Performs randomized hyper-parameter search on ExtraTrees |
| 4    | Evaluates the tuned model                                |
| 5    | Lists feature importances                                |

No extra scripts needed—just run the notebook cells sequentially.

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch (git checkout -b feature/amazing-idea)
3. Commit your changes (git commit -m 'Add amazing idea')
4. Push to the branch (git push origin feature/amazing-idea)
5. Open a Pull Request
