"""Shared utilities for Capstone 1: Churn Data Leakage Audit."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any



DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def load_telco_data(data_path: str | None = None):
    """Load Telco churn CSV from a supplied path, local data folder, or public mirror."""
    import pandas as pd



    candidates = []
    if data_path:
        candidates.append(Path(data_path).expanduser())
    candidates.extend([
        DATA_DIR / "Telco-Customer-Churn.csv",
        PROJECT_DIR / "Telco-Customer-Churn.csv",
        PROJECT_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    ])

    source_path = next((p for p in candidates if p.exists()), None)
    if source_path is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        source_path = DATA_DIR / "Telco-Customer-Churn.csv"
        print(f"Dataset not found locally. Downloading from public IBM mirror:\n{DATA_URL}")
        try:
            request = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                source_path.write_bytes(response.read())
        except Exception as exc:
            raise RuntimeError(
                "Could not download the dataset automatically. Download the Telco Customer "
                "Churn CSV from Kaggle or the IBM public mirror, then run with "
                "--data-path path/to/file.csv.\nDataset URL: " + DATA_URL
            ) from exc



    df = pd.read_csv(source_path)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"customerID", "Churn"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


    # Standardize common text blanks and convert TotalCharges to numeric.
    df = df.replace(r"^\s*$", float("nan"), regex=True)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].astype(str).str.strip()
    df = df[df["Churn"].isin(["Yes", "No"])].copy()
    if df.empty:
        raise ValueError("No rows with a valid Yes/No Churn target were found.")
    print(f"Loaded dataset: {source_path} | shape={df.shape}")
    return df


def target_to_binary(target):
    """Convert the Yes/No target to 1/0 without changing its meaning."""
    import numpy as np
    import pandas as pd


    if isinstance(target, pd.Series):
        cleaned = target.astype(str).str.strip().str.lower()
        mapped = cleaned.map({"yes": 1, "no": 0, "1": 1, "0": 0})
        if mapped.isna().any():
            raise ValueError("Unexpected values in Churn target: " + str(cleaned[mapped.isna()].unique()))
        return mapped.astype(int).to_numpy()
    arr = np.asarray(target)
    mapped = np.where(np.char.lower(arr.astype(str)) == "yes", 1,
                    np.where(np.char.lower(arr.astype(str)) == "no", 0, arr.astype(int)))
    return mapped.astype(int)



def build_preprocessor(X):
    """Build numeric/categorical preprocessing. Fit it only on training data in fixed pipeline."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    import numpy as np



    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()


    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ], remainder="drop")
    return preprocessor



def calculate_metrics(y_true, y_pred, y_prob):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    return metrics



def save_result(name: str, metrics: dict[str, Any], metadata: dict[str, Any]):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"model": name, "metrics": metrics, "metadata": metadata}
    path = OUTPUT_DIR / f"{name}_metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    cm_path = OUTPUT_DIR / f"{name}_confusion_matrix.csv"
    cm = metrics["confusion_matrix"]
    cm_path.write_text(
        "actual_label,predicted_no,predicted_yes\n" + f"No,{cm[0][0]},{cm[0][1]}\nYes,{cm[1][0]},{cm[1][1]}\n",
        encoding="utf-8",
    )
    return path



def _metric_table_value(value: float) -> str:
    return f"{value:.4f}"



def write_leakage_audit(leaky: dict[str, Any], fixed: dict[str, Any], path: Path | None = None):
    """Write the final audit report with metrics measured by run_all.py."""
    path = path or PROJECT_DIR / "LEAKAGE_AUDIT.md"
    lm, fm = leaky["metrics"], fixed["metrics"]
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    table_rows = []
    for key in metric_keys:
        label = key.replace("_", " ").title()
        table_rows.append(
            f"| {label} | {_metric_table_value(lm[key])} | {_metric_table_value(fm[key])} | "
            f"{fm[key] - lm[key]:+.4f} |"
        )
    overlap = leaky["metadata"].get("train_test_customer_overlap", "N/A")
    duplicate_test_rows = leaky["metadata"].get("duplicate_rows_added_to_test", "N/A")
    base_test_rows = leaky["metadata"].get("original_test_rows", "N/A")
    fixed_overlap = fixed["metadata"].get("train_test_customer_overlap", "N/A")
    rows_before = leaky["metadata"].get("source_rows", "N/A")
    fixed_rows = fixed["metadata"].get("rows_after_customer_dedup", "N/A")
    fixed_duplicates_removed = fixed["metadata"].get("duplicate_customer_rows_removed", "N/A")
    leaky_train_rows = leaky["metadata"].get("train_rows", "N/A")

    report = f"""# Capstone 1 — Churn Data Leakage Audit


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

- Source rows before leaky experiment: **{rows_before}**
- Fixed dataset rows after customer-ID deduplication: **{fixed_rows}**
- Duplicate customer rows removed in fixed version: **{fixed_duplicates_removed}**
- Leaky training rows: **{leaky_train_rows}**
- Original leaky test rows: **{base_test_rows}**
- Training rows copied into leaky test: **{duplicate_test_rows}**
- Train/test customer-ID overlap in leaky version: **{overlap}**
- Train/test customer-ID overlap in fixed version: **{fixed_overlap}** (should be 0)


## Before vs. after metrics


| Metric | Leaky pipeline | Fixed pipeline | Fixed − leaky |
|---|---:|---:|---:|
{chr(10).join(table_rows)}


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

- IBM public Telco Customer Churn sample mirror: {DATA_URL}
- Kaggle dataset listing: https://www.kaggle.com/datasets/blastchar/telco-customer-churn


"""
    path.write_text(report, encoding="utf-8")
    print(f"Audit report written to: {path}")
    return path
