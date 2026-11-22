# Capstone 1 — Churn Data Leakage Audit


## Executive summary

This experiment intentionally builds an invalid churn model, then corrects the data preparation and evaluation process. The leaky score is **not a trustworthy estimate of deployment performance**. The corrected score is a more defensible holdout estimate, although it is still not a guarantee of future performance.

## Four leakage bugs

| Bug | Why it leaks | Real-world risk | Correction |
|---|---|---|---|
| **1. Pre-split preprocessing** | The encoder and scaler are fitted using the entire dataset, including validation/test rows. Test-set distribution information influences preprocessing. | Evaluation is no longer a clean simulation of unseen data; estimates may be optimistic or unstable. | Split first. Fit preprocessing only inside a training pipeline; transform test rows with the fitted training transformers. |
| **2. Target-derived churn-risk aggregate** | `churn_risk_score` is created from churn labels grouped by contract and internet-service combination, using labels from the full dataset before splitting. | The model indirectly receives information calculated from outcomes it is supposed to predict, including validation/test outcomes. | Remove the target-derived feature. If target encoding is ever justified, compute it fold-safely from training data only. |
| **3. Duplicate customers across train/test** | The leaky experiment intentionally copies a sample of training rows into its test set. The same customer/record is seen during training and evaluation. | Test metrics measure memorization of seen records rather than generalization to new customers. | Deduplicate by customer ID before splitting and keep the test set independent. |
| **4. Post-outcome churn flag** | `post_outcome_churn_flag` directly marks whether churn has already occurred. It is unavailable at the time a retention prediction should be made. | A model can appear highly accurate by using information that only exists after the decision window. | Exclude post-outcome variables. Define a prediction timestamp and use only information available before it. |


## Experimental setup

- **Dataset:** Telco Customer Churn (IBM sample)
- **Target:** `Churn` (`Yes` = 1, `No` = 0)
- **Model:** Logistic Regression in both experiments
- **Leaky version:** preprocessing fit before split; target-derived feature; post-outcome flag; non-stratified split; copied training rows appended to test
- **Fixed version:** customer-ID deduplication; stratified train/test split; preprocessing/model in one scikit-learn Pipeline; no target-derived or post-outcome columns
- **Random seed:** 42


## Dataset / split diagnostics

- Source rows before leaky experiment: **7043**
- Fixed dataset rows after customer-ID deduplication: **7043**
- Duplicate customer rows removed in fixed version: **0**
- Leaky training rows: **5634**
- Original leaky test rows: **1409**
- Training rows copied into leaky test: **845**
- Train/test customer-ID overlap in leaky version: **845**
- Train/test customer-ID overlap in fixed version: **0** (should be 0)


## Before vs. after metrics


| Metric | Leaky pipeline | Fixed pipeline | Fixed − leaky |
|---|---:|---:|---:|
| Accuracy | 1.0000 | 0.8055 | -0.1945 |
| Precision | 1.0000 | 0.6572 | -0.3428 |
| Recall | 1.0000 | 0.5588 | -0.4412 |
| F1 | 1.0000 | 0.6040 | -0.3960 |
| Roc Auc | 1.0000 | 0.8419 | -0.1581 |


## Interpretation

The leaky pipeline's metrics are invalid because its test set contains records copied from training, its features contain outcome information, and its preprocessing has seen the full dataset. Any apparent improvement must not be interpreted as model quality. The fixed pipeline uses a clean split and training-only fit; the metrics are the appropriate comparison for this exercise.

The difference between the two columns quantifies how much the invalid experiment differs from the corrected evaluation in this run. The size and direction of the gap can vary by random seed and dataset; **a lower fixed score is expected to be a useful reality check, not a failure**.

## Reproducibility

Run the entire comparison from the project folder:


```bash
python run_all.py
```

The script regenerates the metric JSON files, confusion-matrix CSV files, and this report using the dataset you run locally. Do not copy placeholder or synthetic metrics into a submission.


## Data source

- IBM public Telco Customer Churn sample mirror: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- Kaggle dataset listing: https://www.kaggle.com/datasets/blastchar/telco-customer-churn


