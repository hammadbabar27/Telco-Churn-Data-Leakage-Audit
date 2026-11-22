"""Deliberately flawed churn pipeline for Capstone 1 leakage audit.

DO NOT use this approach for real predictive modeling. It intentionally includes
multiple data-leakage errors so they can be measured and discussed.
"""
from __future__ import annotations

import argparse
from pathlib import Path



from common import (
    load_telco_data, target_to_binary, build_preprocessor,
    calculate_metrics, save_result,
)



def run_leaky_pipeline(df):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split



    data = df.copy().reset_index(drop=True)
    y = target_to_binary(data["Churn"])
    ids = data["customerID"].astype(str).to_numpy()


    # LEAK #2: outcome-derived target encoding is calculated using ALL labels,
    # before a train/test split. This leaks test-set label information.
    contract_cols = [c for c in ["Contract", "InternetService"] if c in data.columns]
    if contract_cols:
        data["churn_risk_score"] = data.groupby(contract_cols, dropna=False)["Churn"].transform(
            lambda group: (group == "Yes").mean()
        )
    else:
        data["churn_risk_score"] = data["Churn"].map({"Yes": 1.0, "No": 0.0})

    # LEAK #4: this feature records the outcome itself and is only knowable
    # after churn has already happened. It must not be available at prediction time.
    data["post_outcome_churn_flag"] = y.astype(float)



    X = data.drop(columns=["Churn", "customerID"], errors="ignore")
    preprocessor = build_preprocessor(X)



    # LEAK #1: fit and transform on the complete dataset before splitting.
    X_all_prepared = preprocessor.fit_transform(X)



    # Deliberately non-stratified split to reproduce a flawed evaluation design.
    all_indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(
        all_indices, test_size=0.20, random_state=42
    )

    # LEAK #3: copy some training records into the test set after the split.
    # These records (including their customer IDs and labels) have already been
    # seen during model training, so the evaluation set is contaminated.
    rng = np.random.RandomState(42)
    duplicate_count = max(1, int(round(0.15 * len(train_idx))))
    duplicate_idx = rng.choice(train_idx, size=duplicate_count, replace=False)
    leaky_test_idx = np.concatenate([test_idx, duplicate_idx])



    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_all_prepared[train_idx], y[train_idx])
    y_pred = model.predict(X_all_prepared[leaky_test_idx])
    y_prob = model.predict_proba(X_all_prepared[leaky_test_idx])[:, 1]
    metrics = calculate_metrics(y[leaky_test_idx], y_pred, y_prob)



    train_ids = set(ids[train_idx])
    test_ids = set(ids[leaky_test_idx])
    metadata = {
        "source_rows": int(len(data)),
        "train_rows": int(len(train_idx)),
        "original_test_rows": int(len(test_idx)),
        "duplicate_rows_added_to_test": int(len(duplicate_idx)),
        "evaluated_test_rows_total": int(len(leaky_test_idx)),
        "train_test_customer_overlap": int(len(train_ids.intersection(test_ids))),
        "split_stratified": False,
        "preprocessing_fit_scope": "full_dataset_before_split",
        "leaky_features": ["churn_risk_score", "post_outcome_churn_flag"],
    }
    save_result("leaky", metrics, metadata)
    print("\n=== DELIBERATELY LEAKY MODEL ===")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{key:>10}: {metrics[key]:.4f}")
    print(f"Train/test customer-ID overlap: {metadata['train_test_customer_overlap']}")
    print(f"Training rows copied into test: {metadata['duplicate_rows_added_to_test']}")
    print("WARNING: These scores are intentionally invalid and must not be presented as model performance.")
    return {"metrics": metrics, "metadata": metadata}



def main():
    parser = argparse.ArgumentParser(description="Run the deliberately leaky churn pipeline.")
    parser.add_argument("--data-path", default=None, help="Path to Telco Customer Churn CSV (optional).")
    args = parser.parse_args()
    df = load_telco_data(args.data_path)
    run_leaky_pipeline(df)



if __name__ == "__main__":
    main()
