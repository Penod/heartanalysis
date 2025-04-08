# ❤️ Heart Failure Prediction using Machine Learning

This project applies machine learning techniques to predict the likelihood of heart failure based on patient clinical features. The dataset is derived from real medical records and includes variables such as age, blood pressure, ejection fraction, and more.

The project walks through exploratory data analysis, preprocessing, model training, and evaluation using classification algorithms.

---

## 📊 Dataset Overview

The dataset contains 299 patient records with the following key features:
- Age, Sex
- Ejection Fraction
- Serum Creatinine
- High Blood Pressure
- Smoking, Diabetes
- Death Event (Target Variable)

---

## 🧠 Methods Used

- **EDA**: Correlation matrix, pairplots, histograms
  ## 🔍 Exploratory Data Visualizations

### 📊 Age Distribution
![Age Histogram](images/histogram.pn)

### 🧩 Feature Correlation
![Correlation Matrix](https://github.com/Penod/heartanalysis/blob/main/correlation_matrix.png)

### 🧪 Feature Distributions by Outcome
![Boxplots](https://github.com/Penod/heartanalysis/blob/main/Boxplots.png)

- **Data Preprocessing**: Handling imbalance, feature scaling
- **Modeling**:  
  - Logistic Regression  
  - Random Forest  

- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix

---

## 🏆 Key Findings

- Random Forest yielded the highest predictive performance.
- Time, ejection fraction and serum creatinine were the most influential predictors of heart failure.
- Visual diagnostics aided interpretation and model comparison.

---

## 🔧 Tech Stack

- Python (Pandas, NumPy, Matplotlib)
- Scikit-learn
- Jupyter Notebook

---

## 📂 File Structure

| File | Description |
|------|-------------|
| `heart_failure_project.ipynb` | Full notebook containing EDA, modeling, and results |
| `histogram.png` | Visualization of age distribution (generated during EDA) |

---

## 🚀 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/heartanalysis.git
   cd heartanalysis
