# Capstone 1 — Churn Data Leakage Audit


A reproducible internship capstone demonstrating why a suspiciously excellent churn model can be invalid. This project intentionally creates a leaky experiment, audits the four leakage sources, and compares it with a corrected scikit-learn pipeline.


## Project files

- `leaky_pipeline.py` — deliberately flawed pipeline; **do not use for real predictions**.
- `fixed_pipeline.py` — corrected pipeline with deduplication, stratified split, and training-only preprocessing.
- `common.py` — shared loading, preprocessing, metrics, and report utilities.
- `run_all.py` — runs both pipelines and generates `LEAKAGE_AUDIT.md` with the actual measured metrics.
- `LEAKAGE_AUDIT.md` — explanation of the four bugs and a report template; automatically updated by `run_all.py`.


## Leakage intentionally included in the flawed pipeline

1. Scaling and one-hot encoding are fitted to the full dataset before the split.
2. `churn_risk_score` is target-encoded from the entire dataset's churn labels.
3. Training records are copied into the test set, causing train/test customer overlap.
4. `post_outcome_churn_flag` directly records whether churn has already happened.


## Dataset

Telco Customer Churn (IBM sample), with `Churn` as the binary target. The script first checks local dataset locations, then attempts to download a public IBM mirror if it cannot find a CSV.


- IBM public raw CSV: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4D/master/data/Telco-Customer-Churn.csv
- Kaggle dataset page: https://www.kaggle.com/datasets/blastchar/telco-customer-churn


If the automatic download fails, download the CSV yourself and run the project with `--data-path`.


## Run it


### 1) Create a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2) Run both experiments

```bash
python run_all.py
```

Or specify your local dataset:
```bash
python run_all.py --data-path "data/Telco-Customer-Churn.csv"
```

### 3) Review outputs

- `outputs/leaky_metrics.json`
- `outputs/fixed_metrics.json`
- `outputs/leaky_confusion_matrix.csv`
- `outputs/fixed_confusion_matrix.csv`
- `LEAKAGE_AUDIT.md` (regenerated with actual metric values and diagnostics)


## How to interpret the results

- The leaky model's score is **invalid**. Its test set is contaminated with training records, its features include target-derived/post-outcome information, and its preprocessing has already seen the full dataset.
- The fixed model's metrics are a better holdout estimate. They are still only one split, so a production model should be validated with cross-validation, temporal validation when applicable, and monitoring.
- A lower fixed score is not a failure; it is the consequence of evaluating a model without shortcuts.


## GitHub note

Do not commit the downloaded CSV dataset unless your dataset license and repository policy explicitly allow it. The `.gitignore` excludes the downloaded `data/*.csv` files.
