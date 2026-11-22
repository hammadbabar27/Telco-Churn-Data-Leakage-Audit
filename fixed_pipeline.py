"""Corrected churn pipeline: deduplicate, split, then fit preprocessing on train."""
from __future__ import annotations



import argparse
from common import load_telco_data, target_to_binary, build_preprocessor, calculate_metrics, save_result



def run_fixed_pipeline(df):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline



    data = df.copy().reset_index(drop=True)
    rows_before = len(data)
    duplicate_customer_rows_removed = int(data["customerID"].duplicated(keep="first").sum())
    # Clean data before splitting: each customer should appear only once.
    data = data.drop_duplicates(subset="customerID", keep="first").reset_index(drop=True)
    y = target_to_binary(data["Churn"])
    ids = data["customerID"].astype(str).to_numpy()



    # Drop identifiers, target, and known target/post-outcome derived features.
    leaky_columns = [
        "churn_risk_score",
        "post_outcome_churn_flag",
        "churn_followup_flag",
        "churn_case_flag",
        "churn_after_event_flag",
    ]
    X = data.drop(columns=["Churn", "customerID"] + leaky_columns, errors="ignore")



    # Correct: stratify first. Preprocessing is part of the Pipeline, so it is
    # fit only on training rows when pipeline.fit() is called below.
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.20, random_state=42, stratify=y
    )
    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_prob)



    metadata = {
        "source_rows": int(rows_before),
        "rows_after_customer_dedup": int(len(data)),
        "duplicate_customer_rows_removed": duplicate_customer_rows_removed,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_test_customer_overlap": int(len(set(ids_train).intersection(set(ids_test)))),
        "split_stratified": True,
        "preprocessing_fit_scope": "training_split_only_inside_pipeline",
        "leaky_features_removed": leaky_columns,
    }
    save_result("fixed", metrics, metadata)
    print("\n=== FIXED MODEL ===")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{key:>10}: {metrics[key]:.4f}")
    print(f"Train/test customer-ID overlap: {metadata['train_test_customer_overlap']}")
    print(f"Customer-ID duplicate rows removed: {metadata['duplicate_customer_rows_removed']}")
    print("Preprocessing fit: training data only")
    return {"metrics": metrics, "metadata": metadata}



def main():
    parser = argparse.ArgumentParser(description="Run the corrected churn pipeline.")
    parser.add_argument("--data-path", default=None, help="Path to Telco Customer Churn CSV (optional).")
    args = parser.parse_args()
    df = load_telco_data(args.data_path)
    run_fixed_pipeline(df)



if __name__ == "__main__":
    main()
