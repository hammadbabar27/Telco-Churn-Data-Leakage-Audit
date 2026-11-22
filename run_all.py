"""Run both experiments and regenerate the leakage audit report."""
from __future__ import annotations



import argparse
from common import load_telco_data, write_leakage_audit
from leaky_pipeline import run_leaky_pipeline
from fixed_pipeline import run_fixed_pipeline



def main():
    parser = argparse.ArgumentParser(description="Run Capstone 1 leakage audit end-to-end.")
    parser.add_argument("--data-path", default=None, help="Path to Telco Customer Churn CSV (optional).")
    args = parser.parse_args()

    data = load_telco_data(args.data_path)
    leaky = run_leaky_pipeline(data)
    fixed = run_fixed_pipeline(data)
    report = write_leakage_audit(leaky, fixed)

    print("\n=== FINAL SIDE-BY-SIDE ===")
    print(f"{'Metric':<12} {'Leaky':>10} {'Fixed':>10} {'Delta':>10}")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{key:<12} {leaky['metrics'][key]:>10.4f} {fixed['metrics'][key]:>10.4f} "
              f"{fixed['metrics'][key] - leaky['metrics'][key]:>+10.4f}")
    print(f"\nReport: {report}")
    print("Outputs saved in the outputs/ folder.")



if __name__ == "__main__":
    main()
