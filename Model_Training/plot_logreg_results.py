from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULT_DIR = Path("data/results")
PLOT_DIR = Path("data/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = RESULT_DIR / "logreg_summary.csv"
THRESHOLD_PATH = RESULT_DIR / "logreg_threshold_sweep.csv"


def load_results():
    summary_df = pd.read_csv(SUMMARY_PATH)
    threshold_df = pd.read_csv(THRESHOLD_PATH)

    # Ensure sorting columns exist / are numeric
    summary_df["horizon_seconds"] = pd.to_numeric(summary_df["horizon_seconds"], errors="coerce")
    threshold_df["horizon_seconds"] = pd.to_numeric(threshold_df["horizon_seconds"], errors="coerce")
    threshold_df["decision_threshold"] = pd.to_numeric(threshold_df["decision_threshold"], errors="coerce")

    return summary_df, threshold_df


def export_top_results_table(summary_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Export a short summary table for poster/report use.
    """
    cols = [
        "horizon",
        "feature_set",
        "n_features",
        "pr_auc_test",
        "roc_auc_test",
        "best_decision_threshold_by_val_f1",
        "test_precision_at_best_threshold",
        "test_recall_at_best_threshold",
        "test_f1_at_best_threshold",
    ]

    top_df = (
        summary_df.sort_values(["pr_auc_test", "test_f1_at_best_threshold"], ascending=False)
        .loc[:, cols]
        .head(top_n)
        .reset_index(drop=True)
    )

    out_path = RESULT_DIR / "top_logreg_results.csv"
    top_df.to_csv(out_path, index=False)
    print(f"Saved top results table to: {out_path}")
    print(top_df.head(top_n).to_string(index=False))
    return top_df


def plot_horizon_performance(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each horizon, keep the best feature set by PR-AUC on test.
    """
    best_per_horizon = (
        summary_df.sort_values(["horizon_seconds", "pr_auc_test"], ascending=[True, False])
        .groupby("horizon", as_index=False)
        .first()
        .sort_values("horizon_seconds")
        .reset_index(drop=True)
    )

    plt.figure(figsize=(9, 5))
    plt.plot(best_per_horizon["horizon_seconds"], best_per_horizon["pr_auc_test"], marker="o")
    plt.xlabel("Horizon (seconds)")
    plt.ylabel("Test PR-AUC")
    plt.title("Best Test PR-AUC by Horizon")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = PLOT_DIR / "horizon_performance.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved horizon performance plot to: {out_path}")
    return best_per_horizon


def plot_feature_performance(summary_df: pd.DataFrame, chosen_horizon: str) -> pd.DataFrame:
    """
    Compare feature sets at one chosen horizon.
    """
    horizon_df = (
        summary_df.loc[summary_df["horizon"] == chosen_horizon]
        .sort_values("pr_auc_test", ascending=False)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(11, 6))
    plt.bar(horizon_df["feature_set"], horizon_df["pr_auc_test"])
    plt.xlabel("Feature set")
    plt.ylabel("Test PR-AUC")
    plt.title(f"Feature Set Performance at Horizon = {chosen_horizon}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = PLOT_DIR / f"feature_performance_{chosen_horizon}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved feature performance plot to: {out_path}")
    return horizon_df


def plot_threshold_performance(
    threshold_df: pd.DataFrame,
    chosen_horizon: str,
    chosen_feature_set: str,
) -> pd.DataFrame:
    """
    Plot precision / recall / F1 across decision thresholds
    for one selected horizon + feature set.
    """
    df = (
        threshold_df[
            (threshold_df["horizon"] == chosen_horizon) &
            (threshold_df["feature_set"] == chosen_feature_set)
        ]
        .sort_values("decision_threshold")
        .reset_index(drop=True)
    )

    plt.figure(figsize=(9, 5))
    plt.plot(df["decision_threshold"], df["test_precision"], marker="o", label="Precision")
    plt.plot(df["decision_threshold"], df["test_recall"], marker="o", label="Recall")
    plt.plot(df["decision_threshold"], df["test_f1"], marker="o", label="F1")
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric value")
    plt.title(f"Threshold Performance: {chosen_horizon} | {chosen_feature_set}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_feature_name = chosen_feature_set.replace("+", "_plus_")
    out_path = PLOT_DIR / f"threshold_performance_{chosen_horizon}_{safe_feature_name}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved threshold performance plot to: {out_path}")
    return df


def export_best_setting(summary_df: pd.DataFrame) -> pd.Series:
    """
    Pick one overall best setting by test PR-AUC, then F1.
    """
    best_row = (
        summary_df.sort_values(
            ["pr_auc_test", "test_f1_at_best_threshold", "roc_auc_test"],
            ascending=False
        )
        .iloc[0]
    )

    out_path = RESULT_DIR / "best_logreg_setting.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for k, v in best_row.items():
            f.write(f"{k}: {v}\n")

    print(f"Saved best setting summary to: {out_path}")
    print("\nBest setting:")
    print(best_row.to_string())
    return best_row


if __name__ == "__main__":
    summary_df, threshold_df = load_results()

    # 1. Export top table
    top_df = export_top_results_table(summary_df, top_n=10)

    # 2. Plot best PR-AUC by horizon
    best_per_horizon = plot_horizon_performance(summary_df)

    # 3. Pick overall best setting
    best_row = export_best_setting(summary_df)
    best_horizon = best_row["horizon"]
    best_feature_set = best_row["feature_set"]

    # 4. Feature comparison at best horizon
    feature_df = plot_feature_performance(summary_df, chosen_horizon=best_horizon)

    # 5. Threshold curve for best horizon + best feature set
    threshold_curve_df = plot_threshold_performance(
        threshold_df,
        chosen_horizon=best_horizon,
        chosen_feature_set=best_feature_set,
    )