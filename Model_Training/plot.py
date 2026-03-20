from __future__ import annotations

import argparse
from pathlib import Path
import re
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REG_MODELS = ["ridge_reg", "hist_gbrt_reg"]
CLF_MODELS = ["logreg_clf", "hist_gb_clf"]


def set_plot_style() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["legend.title_fontsize"] = 10
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["savefig.dpi"] = 300


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def pick_best_csv(
    best_csv: str | Path | None,
    grid_csv: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    best_df = load_csv(best_csv) if best_csv else None
    grid_df = load_csv(grid_csv) if grid_csv else None

    if best_df is None and grid_df is None:
        raise ValueError("Provide at least one of --best-csv or --grid-csv")

    if best_df is None:
        best_df = build_best_from_grid(grid_df)
    return best_df, grid_df


def build_best_from_grid(grid_df: pd.DataFrame) -> pd.DataFrame:
    frames = []

    reg = grid_df[grid_df["task"] == "regression"].copy()
    if not reg.empty:
        reg = reg.sort_values(
            ["dataset_name", "val_mae", "model"],
            ascending=[True, True, True],
        )
        reg_best = reg.groupby("dataset_name", as_index=False).first()
        reg_best["is_best_for_horizon_regression"] = 1.0
        frames.append(reg_best)

    clf = grid_df[grid_df["task"] == "classification"].copy()
    if not clf.empty:
        clf = clf.sort_values(
            ["dataset_name", "val_pr_auc", "model"],
            ascending=[True, False, True],
        )
        clf_best = clf.groupby("dataset_name", as_index=False).first()
        clf_best["is_best_for_horizon_classification"] = 1.0
        frames.append(clf_best)

    if not frames:
        raise ValueError("Could not build best-by-file table from grid CSV.")

    return pd.concat(frames, ignore_index=True)


def short_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    return name.replace("-features.parquet", "")


def market_label(name: str) -> str:
    """
    Extract short matchup label like:
    KXNBAGAME-26MAR07GSWOKC-GSW-features.parquet -> GSW-OKC
    KXNBAGAME-26MAR07UTAMIL-MIL-features.parquet -> UTA-MIL
    """
    s = short_name(name)

    m = re.search(r"KXNBAGAME-\d+[A-Z]{3}([A-Z]{3})([A-Z]{3})-[A-Z]{3}$", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    parts = s.split("-")
    if len(parts) >= 2:
        maybe_game = parts[0]
        team_side = parts[1]
        if len(maybe_game) >= 6:
            trailing = maybe_game[-6:]
            if trailing.isalpha():
                return f"{trailing[:3]}-{trailing[3:]}"
        return f"{maybe_game[-3:]}-{team_side}"
    return s


def _pretty_model_name(model: str) -> str:
    mapping = {
        "ridge_reg": "Ridge Regression",
        "hist_gbrt_reg": "Histogram GBRT",
        "logreg_clf": "Logistic Regression",
        "hist_gb_clf": "Histogram GBC",
    }
    return mapping.get(model, model)


def plot_win_counts(best_df: pd.DataFrame, plot_dir: Path) -> None:
    reg = best_df[best_df["task"] == "regression"].copy()
    clf = best_df[best_df["task"] == "classification"].copy()

    reg_counts = reg["model"].value_counts().reindex(REG_MODELS, fill_value=0)
    clf_counts = clf["model"].value_counts().reindex(CLF_MODELS, fill_value=0)

    reg_colors = {"ridge_reg": "#4C78A8", "hist_gbrt_reg": "#F58518"}
    clf_colors = {"logreg_clf": "#54A24B", "hist_gb_clf": "#E45756"}

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(
        [_pretty_model_name(x) for x in reg_counts.index],
        reg_counts.values,
        color=[reg_colors.get(x, "#999999") for x in reg_counts.index],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Number of Files Won")
    ax.set_title("Figure 1A: Regression Model Win Counts")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for i, v in enumerate(reg_counts.values):
        ax.text(i, v + 0.05, str(int(v)), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(plot_dir / "figure_1A_regression_win_counts.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(
        [_pretty_model_name(x) for x in clf_counts.index],
        clf_counts.values,
        color=[clf_colors.get(x, "#999999") for x in clf_counts.index],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Number of Files Won")
    ax.set_title("Figure 1B: Classification Model Win Counts")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for i, v in enumerate(clf_counts.values):
        ax.text(i, v + 0.05, str(int(v)), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(plot_dir / "figure_1B_classification_win_counts.png", bbox_inches="tight")
    plt.close(fig)


def plot_regression_dotplot(best_df: pd.DataFrame, plot_dir: Path) -> None:
    reg = best_df[best_df["task"] == "regression"].copy()
    reg["file_label"] = reg["dataset_name"].map(market_label)
    reg = reg.sort_values("file_label").reset_index(drop=True)

    reg_colors = {"ridge_reg": "#4C78A8", "hist_gbrt_reg": "#F58518"}
    reg["color"] = reg["model"].map(lambda m: reg_colors.get(m, "#999999"))
    reg["xpos"] = np.arange(len(reg))

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.scatter(
        reg["xpos"],
        reg["val_mae"],
        s=95,
        c=reg["color"],
        edgecolors="black",
        linewidths=0.7,
        zorder=3,
    )

    ymin = max(reg["val_mae"].min() * 0.7, 1e-6)
    for _, row in reg.iterrows():
        ax.vlines(
            x=row["xpos"],
            ymin=ymin,
            ymax=row["val_mae"],
            color=row["color"],
            alpha=0.35,
            linewidth=1.0,
            zorder=2,
        )

    # 给所有点加标签，和 2B 保持一致
    for _, row in reg.iterrows():
        label = f"{row['val_mae']:.1e}" if row["val_mae"] < 1e-3 else f"{row['val_mae']:.4f}"
        ax.annotate(
            label,
            (row["xpos"], row["val_mae"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(reg["xpos"])
    ax.set_xticklabels(reg["file_label"], rotation=45, ha="right")
    ax.set_xlabel("Market")
    ax.set_ylabel("Validation MAE")
    ax.set_title("Figure 2A: Regression Model Comparison Across Markets")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)

    legend_handles = []
    for model in REG_MODELS:
        if model in set(reg["model"]):
            legend_handles.append(
                plt.Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=reg_colors.get(model, "#999999"),
                    markeredgecolor="black",
                    markeredgewidth=0.7,
                    markersize=8,
                    label=_pretty_model_name(model),
                )
            )
    ax.legend(handles=legend_handles, title="Winning Model", frameon=True)

    fig.tight_layout()
    fig.savefig(plot_dir / "figure_2A_regression_dotplot.png", bbox_inches="tight")
    plt.close(fig)


def plot_classification_dotplot(best_df: pd.DataFrame, plot_dir: Path) -> None:
    clf = best_df[best_df["task"] == "classification"].copy()
    clf["file_label"] = clf["dataset_name"].map(market_label)
    clf = clf.sort_values("file_label").reset_index(drop=True)

    clf_colors = {"logreg_clf": "#54A24B", "hist_gb_clf": "#E45756"}
    clf["color"] = clf["model"].map(lambda m: clf_colors.get(m, "#999999"))
    clf["xpos"] = np.arange(len(clf))

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.scatter(
        clf["xpos"],
        clf["val_pr_auc"],
        s=95,
        c=clf["color"],
        edgecolors="black",
        linewidths=0.7,
        zorder=3,
    )

    for _, row in clf.iterrows():
        ax.vlines(
            x=row["xpos"],
            ymin=0,
            ymax=row["val_pr_auc"],
            color=row["color"],
            alpha=0.35,
            linewidth=1.0,
            zorder=2,
        )

    for _, row in clf.iterrows():
        ax.annotate(
            f"{row['val_pr_auc']:.3f}",
            (row["xpos"], row["val_pr_auc"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(clf["xpos"])
    ax.set_xticklabels(clf["file_label"], rotation=45, ha="right")
    ax.set_xlabel("Market")
    ax.set_ylabel("Validation PR-AUC")
    ax.set_title("Figure 2B: Classification Model Comparison Across Markets")
    ax.set_ylim(0, max(0.55, clf["val_pr_auc"].max() + 0.05))
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)

    legend_handles = []
    for model in CLF_MODELS:
        if model in set(clf["model"]):
            legend_handles.append(
                plt.Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=clf_colors.get(model, "#999999"),
                    markeredgecolor="black",
                    markeredgewidth=0.7,
                    markersize=8,
                    label=_pretty_model_name(model),
                )
            )
    ax.legend(handles=legend_handles, title="Winning Model", frameon=True)

    fig.tight_layout()
    fig.savefig(plot_dir / "figure_2B_classification_dotplot.png", bbox_inches="tight")
    plt.close(fig)


def _best_row(best_df: pd.DataFrame, task: str, dataset_name: str | None = None) -> pd.Series:
    df = best_df[best_df["task"] == task].copy()
    if dataset_name:
        mask = df["dataset_name"] == dataset_name
        if not mask.any():
            raise ValueError(f"Dataset {dataset_name!r} not found for task={task}")
        return df.loc[mask].iloc[0]

    if task == "regression":
        return df.sort_values("val_mae", ascending=True).iloc[0]
    return df.sort_values("val_pr_auc", ascending=False).iloc[0]


def _extract_feature_names_from_pipeline(pipe) -> list[str] | None:
    step = None
    if hasattr(pipe, "named_steps"):
        if "prep" in pipe.named_steps:
            step = pipe.named_steps["prep"]
        elif "preprocessor" in pipe.named_steps:
            step = pipe.named_steps["preprocessor"]

    if step is not None and hasattr(step, "get_feature_names_out"):
        try:
            return list(step.get_feature_names_out())
        except Exception:
            pass

    if hasattr(pipe, "get_feature_names_out"):
        try:
            return list(pipe.get_feature_names_out())
        except Exception:
            pass

    return None


def extract_importance_from_model_path(model_path: str | Path) -> pd.DataFrame:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    pipe = joblib.load(model_path)
    est = pipe
    if hasattr(pipe, "named_steps") and "model" in pipe.named_steps:
        est = pipe.named_steps["model"]

    feature_names = _extract_feature_names_from_pipeline(pipe)
    importance = None
    method = None

    if hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        if coef.ndim == 1:
            importance = np.abs(coef)
        else:
            importance = np.mean(np.abs(coef), axis=0)
        method = "coefficient"
    elif hasattr(est, "feature_importances_"):
        importance = np.asarray(est.feature_importances_)
        method = "feature_importance"
    else:
        raise ValueError("Model does not expose coef_ or feature_importances_.")

    importance = np.asarray(importance).reshape(-1)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importance))]

    n = min(len(feature_names), len(importance))
    feature_names = feature_names[:n]
    importance = importance[:n]

    out = pd.DataFrame(
        {"feature": feature_names, "importance": importance, "method": method}
    )
    out["abs_importance"] = out["importance"].abs()
    out = out.sort_values("abs_importance", ascending=False).reset_index(drop=True)
    return out


def plot_feature_importance(
    best_df: pd.DataFrame,
    plot_dir: Path,
    preferred_task: str,
    dataset_name: str | None,
) -> None:
    row = _best_row(best_df, preferred_task, dataset_name)
    model_col = (
        "saved_model_path_regression"
        if preferred_task == "regression"
        else "saved_model_path_classification"
    )
    model_path = row.get(model_col, "")

    if not isinstance(model_path, str) or not model_path:
        warnings.warn(f"No saved model path available for task={preferred_task}; skipping Figure 4.")
        return

    try:
        imp_df = extract_importance_from_model_path(model_path)
    except Exception as e:
        warnings.warn(f"Could not extract feature importance from {model_path}: {e}")
        return

    top_df = imp_df.head(10).copy().iloc[::-1]

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.barh(
        top_df["feature"],
        top_df["abs_importance"],
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.6,
    )
    method_label = top_df["method"].iloc[0] if "method" in top_df.columns else "unknown"
    ax.set_xlabel("Absolute Importance")
    ax.set_ylabel("Feature")
    ax.set_title(
        "Figure 4: Top 10 Features\n"
        f"Task = {preferred_task.capitalize()} | "
        f"Model = {_pretty_model_name(row['model'])} | "
        f"Method = {method_label}"
    )
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "figure_4_feature_importance.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--best-csv",
        type=str,
        default="data/results/mixed_tasks_active_5s/best_by_file_and_task.csv",
    )
    parser.add_argument(
        "--grid-csv",
        type=str,
        default="data/results/mixed_tasks_active_5s/grid_summary.csv",
    )
    parser.add_argument("--plot-dir", type=str, default="data/plots")
    parser.add_argument(
        "--importance-task",
        type=str,
        choices=["regression", "classification"],
        default="classification",
    )

    args = parser.parse_args()
    plot_dir = ensure_dir(args.plot_dir)
    set_plot_style()

    best_df, _ = pick_best_csv(args.best_csv, args.grid_csv)

    plot_win_counts(best_df, plot_dir)
    plot_regression_dotplot(best_df, plot_dir)
    plot_classification_dotplot(best_df, plot_dir)
    plot_feature_importance(
        best_df,
        plot_dir,
        preferred_task=args.importance_task,
        dataset_name=None,
    )

    print(f"Saved plots to: {plot_dir.resolve()}")


if __name__ == "__main__":
    main()