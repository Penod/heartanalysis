# Heart Failure Prediction

Predicts mortality risk (`DEATH_EVENT`) for heart failure patients from clinical records, using Logistic Regression and Random Forest. The dataset is the public [Heart Failure Clinical Records dataset](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records) (299 patients).

## The headline finding isn't the model - it's a data leakage bug I caught and fixed

The first version of this notebook reported ~78% accuracy with `time` (days of follow-up before the outcome was recorded) as the single strongest predictor. That number was inflated: `time` isn't a real clinical signal, it's a byproduct of how the study was conducted - patients who died were, by definition, more likely to have shorter follow-up windows. At the point you'd actually want a prediction, you don't know in advance how long a patient will be followed.

I tested this directly rather than just asserting it (see the notebook's "Checking the `time` Leakage Claim" section): cross-validated accuracy is actually *higher and more stable* with `time` included (0.809 ± 0.039) than without it (0.716 ± 0.068). That's worth sitting with - leakage doesn't always show up as noisy, unstable metrics. Sometimes it just makes a model look better, with nothing in the numbers themselves to flag it. The only way to catch it is by reasoning about whether a feature would actually exist at prediction time, not by watching the metrics for a red flag. `time` is excluded from every model in the current version, and the reported numbers are honestly lower than the original because of it.

## Dataset Overview

299 patient records:
- Age, sex, smoking, diabetes, high blood pressure, anaemia
- Ejection fraction, serum creatinine, serum sodium, creatinine phosphokinase, platelets
- `time` (follow-up days) - present in the data, excluded from modeling for the reason above
- `DEATH_EVENT` (target), 68% survived / 32% died

## Methods

- **EDA**: correlation matrix, histograms, boxplots by outcome.
- **Leakage check**: cross-validated comparison of the model with vs. without `time`, before deciding to drop it.
- **Preprocessing**: features standardized with `StandardScaler`, fit on the training split only (fitting before the split would leak test-set statistics into training - a subtle but real bug the first version had).
- **Modeling**: Logistic Regression and Random Forest, both with `class_weight="balanced"` to account for the 68/32 class split. Random Forest hyperparameters (`max_depth`, `min_samples_leaf`, `n_estimators`) are tuned with `GridSearchCV` against 5-fold cross-validated ROC-AUC, not left at defaults.
- **Evaluation**: accuracy, precision, recall, F1, ROC-AUC, confusion matrices - ROC-AUC and recall on the death class specifically, since accuracy alone is a weak signal on an imbalanced medical outcome.

## Results

With `time` removed, class weighting applied, and the Random Forest properly tuned: both models land around **ROC-AUC 0.79-0.80** (Logistic Regression 0.791 test / 0.780 CV; Random Forest 0.802 test / 0.800 CV), with accuracy around 75-77%.

The more important number for this kind of problem is recall on the death class - missing an at-risk patient is generally worse than a false alarm. Logistic Regression catches 74% of actual deaths (up from 52% before class weighting was added), at the cost of dropping precision on that class from 93% to 61%. Random Forest is more balanced (63% recall / 60% precision on deaths). Neither model is dramatically better than the other.

Feature importance (tuned Random Forest, `time` excluded): `serum_creatinine` (0.44) and `ejection_fraction` (0.34) dominate, with `age` (0.14) and `serum_sodium` (0.09) contributing less - consistent with the correlation analysis, which is a reasonable sanity check.

## Limitations

299 patients from a single study, no external validation on a different population - performance here shouldn't be assumed to generalize. Precision on the death class is moderate at best, so this would need a clinician reviewing flagged cases in any real use, not an automated decision. This is a modeling exercise demonstrating a leakage catch and a properly tuned pipeline, not a validated clinical tool.

## Tech Stack

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn
- Jupyter Notebook

## File Structure

| File | Description |
|------|-------------|
| `heart_failure_project.ipynb` | Full notebook: EDA, the leakage check, preprocessing, tuned modeling, and results |
| `heart_failure_clinical_records_dataset.csv` | Dataset |
| `histogram.png`, `correlation_matrix.png`, `Boxplots.png` | Exported EDA visualizations |
| `requirements.txt` | Package versions used |

## How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/Penod/heartanalysis.git
   cd heartanalysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `heart_failure_project.ipynb` and run top to bottom.
