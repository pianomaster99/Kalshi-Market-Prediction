from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULT_DIR = Path("data/results")
PLOT_DIR = Path("data/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = RESULT_DIR / "logreg_summary.csv"
COEF_PATH = RESULT_DIR / "logreg_coefficients.csv"


def load_data():
    summary_df = pd.read_csv(SUMMARY_PATH)
    coef_df = pd.read_csv(COEF_PATH)
    return summary_df, coef_df


def get_best_model(summary_df: pd.DataFrame) -> pd.Series:
    """
    Pick the best model by:
    1. highest test PR-AUC
    2. then highest test F1 at best threshold
    3. then highest test ROC-AUC
    """
    best_row = (
        summary_df.sort_values(
            ["pr_auc_test", "test_f1_at_best_threshold", "roc_auc_test"],
            ascending=False
        )
        .iloc[0]
    )
    return best_row


def plot_feature_importance(summary_df: pd.DataFrame, coef_df: pd.DataFrame, top_n: int = 12):
    best_row = get_best_model(summary_df)

    best_horizon = best_row["horizon"]
    best_feature_set = best_row["feature_set"]

    print("Best model:")
    print(best_row.to_string())

    best_coef_df = coef_df[
        (coef_df["horizon"] == best_horizon) &
        (coef_df["feature_set"] == best_feature_set)
    ].copy()

    if best_coef_df.empty:
        raise ValueError("No matching coefficient rows found for best model.")

    best_coef_df = best_coef_df.sort_values("abs_coefficient", ascending=False).head(top_n)

    # Reverse for horizontal bar plot so biggest appears on top
    best_coef_df = best_coef_df.iloc[::-1]

    plt.figure(figsize=(9, 6))
    plt.barh(best_coef_df["feature"], best_coef_df["abs_coefficient"])
    plt.xlabel("Absolute coefficient")
    plt.ylabel("Feature")
    plt.title(f"Feature Importance (Best Logistic Model)\n{best_horizon} | {best_feature_set}")
    plt.tight_layout()

    out_path = PLOT_DIR / "best_model_feature_importance.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved feature importance plot to: {out_path}")

    # Also export the exact feature importance table
    table_out = RESULT_DIR / "best_model_feature_importance.csv"
    best_coef_df.to_csv(table_out, index=False)
    print(f"Saved feature importance table to: {table_out}")


if __name__ == "__main__":
    summary_df, coef_df = load_data()
    plot_feature_importance(summary_df, coef_df, top_n=12)