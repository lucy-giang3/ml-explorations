# Heart Disease Prediction

This project builds an educational binary classifier from the UCI Heart Disease dataset distributed through Kaggle. It predicts whether the original diagnosis severity `num` is greater than zero. It is for learning and analysis, not diagnosis or treatment.

## Experiments

- Explore the 920-row dataset, inspect missing values, and convert `num` (0–4) into a binary target (`num > 0`).
- Exclude the row identifier and source-site label from the predictors.
- Use a stratified 80/20 train/test split (seed 42) and compare class-weighted logistic regression and random forest pipelines with five-fold stratified cross-validation.
- Tune the selected model using training-only `GridSearchCV`; evaluate once on the held-out test set.
- Inspect fitted coefficients and save the final preprocessing-and-model pipeline with joblib.

## Results

Logistic regression was selected by baseline mean CV ROC AUC (0.8850 versus 0.8827 for random forest), a very small difference relative to fold-to-fold variation. Grid search selected `C=0.1` (`lbfgs`), with mean CV ROC AUC 0.887. On the 184-row test split, it achieved 83.7% accuracy, 83.4% balanced accuracy, 0.846 positive-class precision, 0.863 recall, 0.854 F1, and 0.902 ROC AUC. At the default 0.5 threshold, the confusion matrix was TN=66, FP=16, FN=14, TP=88. These are results from one split of a small, multi-site dataset; see [results.md](results.md) for the EDA, method, interpretation, and limitations.

## Project contents

- `heart_disease_prediction.ipynb` — executed analysis notebook with outputs.
- `results.md` — results + analysis
- `heart_disease_model.joblib` — fitted preprocessing and tuned classifier pipeline from my own run.
- `data/heart_disease_uci.csv` — input dataset (not included)
- `requirements.txt` — Python dependencies.

## Run

Use Python 3.10 or newer. From this project directory, install the dependencies and launch Jupyter:

```bash
python -m pip install -r requirements.txt
jupyter notebook heart_disease_prediction.ipynb
```

Run the notebook from top to bottom. It expects the CSV at `data/heart_disease_uci.csv` relative to the project directory and saves the fitted model as `heart_disease_model.joblib` in the working directory.

## Reproducibility and interpretation

The train/test split, stratified folds, and estimators use seed 42. Preprocessing is inside the model pipelines, and tuning uses training data only before final holdout evaluation. Results may differ across dataset revisions and library versions. The random split does not test generalization to a completely unseen site, and several variables have substantial missingness. Feature coefficients describe model associations on transformed inputs; they are not causal effects. The default classification threshold is not optimized for clinical tradeoffs.

Load only trusted joblib artifacts, and use compatible scikit-learn versions. This model is not a medical device and must not be used for clinical decision-making.

## Dataset and reference

The dataset is not included in this repository.

- [UCI Heart Disease dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data).

## Author and License

_Author: Lucy Giang (lucy-giang3), updated Jan. 2026_

This code is shared for free for learning and personal use only.
