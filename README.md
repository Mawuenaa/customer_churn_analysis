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
