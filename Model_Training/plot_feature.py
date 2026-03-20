from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =========================================================
# User-fixed anchor dataset
# =========================================================
ANCHOR_DATASET_NAME = "KXNBAGAME-26MAR07UTAMIL-MIL"


# =========================================================
# Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = PROJECT_ROOT / "data" / "features"
RESULT_DIR = PROJECT_ROOT / "data" / "results" / "mixed_tasks_active_5s"
PLOT_DIR = PROJECT_ROOT / "data" / "plots" / "feature_importance"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CSV = RESULT_DIR / "best_by_file_and_task.csv"

TIME_COL = "ts"
PRICE_COL = "midprice"

DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15
DEFAULT_TEST_FRAC = 0.15

CLASSIFICATION_DEADBAND = 0.0
ACTIVE_MAX_FLAT_SECONDS = 30.0
RANDOM_STATE = 42


# =========================================================
# Helpers aligned with train_grid.py
# =========================================================
def sanitize_name(text: str) -> str:
    return (
        str(text)
        .replace(" ", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def build_future_change_target(
    feature_df: pd.DataFrame,
    horizon: str,
    time_col: str = TIME_COL,
    price_col: str = PRICE_COL,
    drop_unlabeled: bool = True,
) -> pd.DataFrame:
    if time_col not in feature_df.columns:
        raise ValueError(f"feature_df must contain '{time_col}'.")
    if price_col not in feature_df.columns:
        raise ValueError(f"feature_df must contain '{price_col}'.")

    df = feature_df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col, price_col]).sort_values(time_col).reset_index(drop=True)

    horizon_td = pd.to_timedelta(horizon)
    horizon_suffix = sanitize_name(horizon)

    left = df.copy()
    left["target_ts"] = left[time_col] + horizon_td

    right = (
        df[[time_col, price_col]]
        .rename(columns={time_col: "future_ts", price_col: "future_midprice"})
        .sort_values("future_ts")
        .reset_index(drop=True)
    )

    merged = pd.merge_asof(
        left.sort_values("target_ts"),
        right,
        left_on="target_ts",
        right_on="future_ts",
        direction="forward",
        allow_exact_matches=True,
    )

    realized_col = f"realized_horizon_seconds_{horizon_suffix}"
    change_col = f"midprice_change_{horizon_suffix}"
    target_col = f"target_{horizon_suffix}"

    merged[realized_col] = (merged["future_ts"] - merged[time_col]).dt.total_seconds()
    merged[change_col] = merged["future_midprice"] - merged[price_col]
    merged[target_col] = merged[change_col]

    if drop_unlabeled:
        merged = merged.dropna(subset=["future_midprice", target_col]).reset_index(drop=True)

    return merged


def add_direction_target(
    df: pd.DataFrame,
    horizon: str,
    deadband: float = CLASSIFICATION_DEADBAND,
) -> pd.DataFrame:
    out = df.copy()
    suffix = sanitize_name(horizon)
    change_col = f"midprice_change_{suffix}"
    direction_col = f"direction_target_{suffix}"

    if change_col not in out.columns:
        raise ValueError(f"Missing change column: {change_col}")

    if deadband < 0:
        raise ValueError("deadband must be non-negative")

    change = out[change_col]
    if deadband == 0:
        out[direction_col] = (change > 0).astype(int)
    else:
        out[direction_col] = np.where(
            change > deadband,
            1,
            np.where(change < -deadband, 0, np.nan),
        )

    return out


def add_time_split_column(
    df: pd.DataFrame,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> pd.DataFrame:
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    out = df.copy()
    n = len(out)

    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    split = np.empty(n, dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "val"
    split[val_end:] = "test"

    out["split"] = split
    return out


def infer_panel_seconds(df: pd.DataFrame, time_col: str = TIME_COL) -> float | None:
    ts = pd.to_datetime(df[time_col], utc=True, errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return None
    diffs = ts.diff().dropna().dt.total_seconds()
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    return float(diffs.median())


def filter_long_flat_runs(
    df: pd.DataFrame,
    price_col: str = PRICE_COL,
    time_col: str = TIME_COL,
    max_flat_seconds: float = ACTIVE_MAX_FLAT_SECONDS,
) -> pd.DataFrame:
    out = df.copy().sort_values(time_col).reset_index(drop=True)

    if max_flat_seconds <= 0:
        return out

    panel_seconds = infer_panel_seconds(out, time_col=time_col)
    if panel_seconds is None or not np.isfinite(panel_seconds) or panel_seconds <= 0:
        raise ValueError("Could not infer positive panel_seconds for flat-run filter.")

    max_flat_steps = max(1, int(round(max_flat_seconds / panel_seconds)))

    price_change = out[price_col].diff()
    is_flat = price_change.fillna(0).eq(0)

    run_id = is_flat.ne(is_flat.shift(fill_value=False)).cumsum()
    run_pos = is_flat.groupby(run_id).cumsum()

    keep_mask = (~is_flat) | (run_pos < max_flat_steps)
    return out.loc[keep_mask].reset_index(drop=True)


def is_datetime_like(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


def get_feature_columns(
    df: pd.DataFrame,
    target_cols: list[str],
    time_col: str = TIME_COL,
) -> list[str]:
    banned_prefixes = (
        "target_",
        "direction_target_",
        "midprice_change_",
        "future_",
        "realized_horizon_seconds_",
    )

    banned_exact = {time_col, "target_ts", "split", *target_cols}

    preferred_feature_order = [
        "best_yes_bid",
        "best_no_bid",
        "best_yes_ask",
        "best_no_ask",
        "midprice",
        "spread",
        "yes_depth_top1",
        "no_depth_top1",
        "yes_total_depth",
        "no_total_depth",
        "total_depth",
        "yes_depth_share",
        "imbalance_top1",
        "imbalance_total",
        "midprice_lag_1s",
        "midprice_change_1s",
        "spread_lag_1s",
        "spread_change_1s",
        "total_depth_lag_1s",
        "total_depth_change_1s",
        "imbalance_top1_lag_1s",
        "imbalance_top1_change_1s",
        "midprice_mean_3s",
        "midprice_std_3s",
        "spread_mean_3s",
        "imbalance_top1_mean_3s",
    ]

    available = []
    for col in df.columns:
        if col in banned_exact:
            continue
        if any(col.startswith(prefix) for prefix in banned_prefixes):
            continue
        if is_datetime_like(df[col]):
            continue
        available.append(col)

    feature_cols = [col for col in preferred_feature_order if col in available]

    if not feature_cols:
        raise ValueError("No usable feature columns found after leakage filtering.")

    return feature_cols


def drop_constant_columns(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    constant_cols = [c for c in X_train.columns if X_train[c].nunique(dropna=False) <= 1]
    if not constant_cols:
        return X_train, X_val, X_test, []

    return (
        X_train.drop(columns=constant_cols),
        X_val.drop(columns=constant_cols),
        X_test.drop(columns=constant_cols),
        constant_cols,
    )


# =========================================================
# Dataset reconstruction
# =========================================================
def infer_task_target_cols(horizon: str) -> tuple[str, str]:
    suffix = sanitize_name(horizon)
    reg_target_col = f"target_{suffix}"
    clf_target_col = f"direction_target_{suffix}"
    return reg_target_col, clf_target_col


def build_all_splits_for_same_dataset(
    feature_path: Path,
    horizon: str,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    classification_deadband: float = CLASSIFICATION_DEADBAND,
    active_max_flat_seconds: float = ACTIVE_MAX_FLAT_SECONDS,
):
    raw_df = pd.read_parquet(feature_path)

    ds_reg = build_future_change_target(
        feature_df=raw_df,
        horizon=horizon,
        time_col=TIME_COL,
        price_col=PRICE_COL,
        drop_unlabeled=True,
    )
    ds_reg = add_time_split_column(
        ds_reg,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )

    ds_clf = filter_long_flat_runs(
        ds_reg.copy(),
        price_col=PRICE_COL,
        time_col=TIME_COL,
        max_flat_seconds=active_max_flat_seconds,
    )
    ds_clf = add_direction_target(
        ds_clf,
        horizon=horizon,
        deadband=classification_deadband,
    )
    ds_clf = add_time_split_column(
        ds_clf,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )

    reg_target_col, clf_target_col = infer_task_target_cols(horizon)

    feature_cols = get_feature_columns(
        ds_reg,
        target_cols=[reg_target_col, clf_target_col],
        time_col=TIME_COL,
    )

    # regression
    train_df_reg = ds_reg[ds_reg["split"] == "train"].copy()
    val_df_reg = ds_reg[ds_reg["split"] == "val"].copy()
    test_df_reg = ds_reg[ds_reg["split"] == "test"].copy()

    X_train_reg = train_df_reg[feature_cols]
    X_val_reg = val_df_reg[feature_cols]
    X_test_reg = test_df_reg[feature_cols]
    X_train_reg, X_val_reg, X_test_reg, dropped_reg = drop_constant_columns(
        X_train_reg, X_val_reg, X_test_reg
    )

    y_train_reg = train_df_reg[reg_target_col]
    y_val_reg = val_df_reg[reg_target_col]
    y_test_reg = test_df_reg[reg_target_col]

    # classification
    train_df_clf = ds_clf[ds_clf["split"] == "train"].copy()
    val_df_clf = ds_clf[ds_clf["split"] == "val"].copy()
    test_df_clf = ds_clf[ds_clf["split"] == "test"].copy()

    X_train_clf = train_df_clf[feature_cols]
    X_val_clf = val_df_clf[feature_cols]
    X_test_clf = test_df_clf[feature_cols]
    X_train_clf, X_val_clf, X_test_clf, dropped_clf = drop_constant_columns(
        X_train_clf, X_val_clf, X_test_clf
    )

    train_mask = train_df_clf[clf_target_col].notna()
    val_mask = val_df_clf[clf_target_col].notna()
    test_mask = test_df_clf[clf_target_col].notna()

    X_train_clf = X_train_clf.loc[train_mask]
    X_val_clf = X_val_clf.loc[val_mask]
    X_test_clf = X_test_clf.loc[test_mask]

    y_train_clf = train_df_clf.loc[train_mask, clf_target_col].astype(int)
    y_val_clf = val_df_clf.loc[val_mask, clf_target_col].astype(int)
    y_test_clf = test_df_clf.loc[test_mask, clf_target_col].astype(int)

    return {
        "regression": {
            "X_train": X_train_reg,
            "X_val": X_val_reg,
            "X_test": X_test_reg,
            "y_train": y_train_reg,
            "y_val": y_val_reg,
            "y_test": y_test_reg,
            "dropped_constant_cols": dropped_reg,
        },
        "classification": {
            "X_train": X_train_clf,
            "X_val": X_val_clf,
            "X_test": X_test_clf,
            "y_train": y_train_clf,
            "y_val": y_val_clf,
            "y_test": y_test_clf,
            "dropped_constant_cols": dropped_clf,
        },
    }


# =========================================================
# Model / importance helpers
# =========================================================
def get_model_from_pipeline(pipeline):
    if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
        return pipeline.named_steps["model"]
    raise ValueError("Pipeline does not contain a 'model' step.")


def get_importance_dataframe(
    pipeline,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    task: str,
    model_name: str,
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    model = get_model_from_pipeline(pipeline)

    if X_eval.empty:
        raise ValueError(f"{task}: X_eval is empty.")
    if len(X_eval.columns) == 0:
        raise ValueError(f"{task}: no feature columns after preprocessing.")

    if model_name in {"ridge_reg", "logreg_clf"}:
        if not hasattr(model, "coef_"):
            raise ValueError(f"{model_name} does not have coef_.")

        coef = model.coef_
        if np.ndim(coef) == 2:
            coef = coef[0]

        coef = np.asarray(coef).ravel()

        if len(coef) != len(X_eval.columns):
            raise ValueError(
                f"Coefficient length ({len(coef)}) does not match "
                f"number of columns ({len(X_eval.columns)})."
            )

        out = pd.DataFrame(
            {
                "feature": X_eval.columns,
                "importance": np.abs(coef),
                "importance_std": np.nan,
                "method": "Absolute coefficient",
            }
        )

    elif model_name in {"hist_gbrt_reg", "hist_gb_clf"}:
        scoring = "neg_mean_absolute_error" if task == "regression" else "average_precision"

        perm = permutation_importance(
            estimator=pipeline,
            X=X_eval,
            y=y_eval,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
            n_jobs=-1,
        )

        out = pd.DataFrame(
            {
                "feature": X_eval.columns,
                "importance": perm.importances_mean,
                "importance_std": perm.importances_std,
                "method": "Permutation importance",
            }
        )

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    out = out.sort_values("importance", ascending=False).reset_index(drop=True)
    return out


# =========================================================
# Plotting
# =========================================================
def apply_publication_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.titlesize": 18,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def short_feature_name(name: str) -> str:
    mapping = {
        "best_yes_bid": "Best yes bid",
        "best_no_bid": "Best no bid",
        "best_yes_ask": "Best yes ask",
        "best_no_ask": "Best no ask",
        "midprice": "Midprice",
        "spread": "Spread",
        "yes_depth_top1": "Yes depth top1",
        "no_depth_top1": "No depth top1",
        "yes_total_depth": "Yes total depth",
        "no_total_depth": "No total depth",
        "total_depth": "Total depth",
        "yes_depth_share": "Yes depth share",
        "imbalance_top1": "Imbalance top1",
        "imbalance_total": "Imbalance total",
        "midprice_lag_1s": "Midprice lag 1s",
        "midprice_change_1s": "Midprice change 1s",
        "spread_lag_1s": "Spread lag 1s",
        "spread_change_1s": "Spread change 1s",
        "total_depth_lag_1s": "Depth lag 1s",
        "total_depth_change_1s": "Depth change 1s",
        "imbalance_top1_lag_1s": "Imbalance lag 1s",
        "imbalance_top1_change_1s": "Imbalance change 1s",
        "midprice_mean_3s": "Midprice mean 3s",
        "midprice_std_3s": "Midprice std 3s",
        "spread_mean_3s": "Spread mean 3s",
        "imbalance_top1_mean_3s": "Imbalance mean 3s",
    }
    return mapping.get(name, name.replace("_", " "))


def plot_importance(
    imp_df: pd.DataFrame,
    *,
    task: str,
    dataset_name: str,
    model_name: str,
    horizon: str,
    top_n: int,
    out_path: Path,
):
    apply_publication_style()

    df_plot = imp_df.head(top_n).copy()
    df_plot = df_plot.iloc[::-1].reset_index(drop=True)
    df_plot["feature_label"] = df_plot["feature"].map(short_feature_name)

    fig, ax = plt.subplots(figsize=(9.8, 5.9))

    values = df_plot["importance"].to_numpy()
    labels = df_plot["feature_label"].tolist()

    bars = ax.barh(labels, values, alpha=0.9)

    if df_plot["importance_std"].notna().any():
        xerr = df_plot["importance_std"].fillna(0).to_numpy()
        ax.errorbar(
            values,
            labels,
            xerr=xerr,
            fmt="none",
            capsize=3,
            linewidth=1.0,
        )

    pretty_task = "Regression" if task == "regression" else "Classification"
    pretty_model = {
        "ridge_reg": "Ridge Regression",
        "logreg_clf": "Logistic Regression",
        "hist_gbrt_reg": "HistGradientBoosting Regressor",
        "hist_gb_clf": "HistGradientBoosting Classifier",
    }.get(model_name, model_name)

    method = df_plot["method"].iloc[0]

    ax.set_title(
        f"Top {top_n} Feature Importance ({pretty_task})\n"
        f"{dataset_name} | {pretty_model} | Horizon = {horizon}",
        pad=14,
    )
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    max_val = float(np.nanmax(values)) if len(values) else 1.0
    if max_val <= 0:
        max_val = 1.0
    ax.set_xlim(0, max_val * 1.18)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(max_val * 0.015, 1e-6),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.text(
        0.99,
        0.02,
        method,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        style="italic",
        alpha=0.9,
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Best-row selection on FIXED dataset
# =========================================================
def choose_best_row_on_fixed_dataset(best_df: pd.DataFrame, task: str, dataset_name: str) -> pd.Series:
    sub = best_df[
        (best_df["task"].astype(str) == task) &
        (best_df["dataset_name"].astype(str) == dataset_name)
    ].copy()

    if sub.empty:
        raise ValueError(f"No rows found for task={task}, dataset_name={dataset_name}")

    if task == "regression":
        sub = sub.dropna(subset=["val_mae"])
        if sub.empty:
            raise ValueError(f"No regression rows with val_mae for dataset_name={dataset_name}")
        sub = sub.sort_values("val_mae", ascending=True)
    elif task == "classification":
        sub = sub.dropna(subset=["val_pr_auc"])
        if sub.empty:
            raise ValueError(f"No classification rows with val_pr_auc for dataset_name={dataset_name}")
        sub = sub.sort_values("val_pr_auc", ascending=False)
    else:
        raise ValueError(f"Unsupported task={task}")

    return sub.iloc[0]


def choose_best_classification_same_source_and_horizon(best_df: pd.DataFrame, reg_row: pd.Series) -> pd.Series:
    clf_df = best_df[best_df["task"].astype(str) == "classification"].copy()
    if clf_df.empty:
        raise ValueError("No classification rows found.")

    clf_df = clf_df.dropna(subset=["val_pr_auc"])
    if clf_df.empty:
        raise ValueError("No classification rows with val_pr_auc found.")

    mask = (
        (clf_df["dataset_name"].astype(str) == str(reg_row["dataset_name"])) &
        (clf_df["source_file"].astype(str) == str(reg_row["source_file"])) &
        (clf_df["horizon"].astype(str) == str(reg_row["horizon"]))
    )
    sub = clf_df.loc[mask].copy()

    if sub.empty:
        raise ValueError(
            "Found regression row on the fixed dataset, but could not find a classification row "
            "with the same dataset_name + source_file + horizon."
        )

    sub = sub.sort_values("val_pr_auc", ascending=False)
    return sub.iloc[0]


def resolve_model_path(row: pd.Series, task: str) -> Path:
    col = f"saved_model_path_{task}"
    if col not in row:
        raise ValueError(f"Missing column {col}")

    model_path = str(row[col]).strip()
    if model_path == "":
        raise ValueError(f"Empty model path in column {col}")

    return Path(model_path)


# =========================================================
# Main processing
# =========================================================
def process_task(
    row: pd.Series,
    split_bundle: dict,
    task: str,
    dataset_name_for_title: str,
    horizon_for_title: str,
    top_n: int = 10,
):
    model_name = str(row["model"])
    model_path = resolve_model_path(row, task)
    pipeline = joblib.load(model_path)

    X_test = split_bundle["X_test"]
    y_test = split_bundle["y_test"]

    imp_df = get_importance_dataframe(
        pipeline=pipeline,
        X_eval=X_test,
        y_eval=y_test,
        task=task,
        model_name=model_name,
        n_repeats=10,
        random_state=RANDOM_STATE,
    )

    task_tag = "regression" if task == "regression" else "classification"

    stem = f"{dataset_name_for_title}-{sanitize_name(horizon_for_title)}-{task_tag}"
    csv_out = PLOT_DIR / f"{stem}-importance.csv"
    fig_out = PLOT_DIR / f"{stem}-importance.png"

    imp_df.to_csv(csv_out, index=False)

    plot_importance(
        imp_df=imp_df,
        task=task,
        dataset_name=dataset_name_for_title,
        model_name=model_name,
        horizon=horizon_for_title,
        top_n=top_n,
        out_path=fig_out,
    )

    print(f"[{task}] model = {model_name}")
    print(f"[{task}] model path = {model_path}")
    print(f"[{task}] saved csv = {csv_out}")
    print(f"[{task}] saved fig = {fig_out}")


def main():
    if not BEST_CSV.exists():
        raise FileNotFoundError(f"Could not find: {BEST_CSV}")

    best_df = pd.read_csv(BEST_CSV)

    # 1) regression fixed to exact dataset_name user specified
    reg_row = choose_best_row_on_fixed_dataset(
        best_df=best_df,
        task="regression",
        dataset_name=ANCHOR_DATASET_NAME,
    )

    # 2) classification forced to same dataset + source_file + horizon
    clf_row = choose_best_classification_same_source_and_horizon(best_df, reg_row)

    anchor_dataset_name = str(reg_row["dataset_name"])
    anchor_source_file = str(reg_row["source_file"])
    anchor_horizon = str(reg_row["horizon"])

    print("=" * 100)
    print("Using user-specified fixed dataset")
    print(f"Dataset name : {anchor_dataset_name}")
    print(f"Source file  : {anchor_source_file}")
    print(f"Horizon      : {anchor_horizon}")
    print("=" * 100)

    feature_path = FEATURE_DIR / anchor_source_file
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature parquet not found: {feature_path}")

    split_data = build_all_splits_for_same_dataset(
        feature_path=feature_path,
        horizon=anchor_horizon,
        train_frac=DEFAULT_TRAIN_FRAC,
        val_frac=DEFAULT_VAL_FRAC,
        test_frac=DEFAULT_TEST_FRAC,
        classification_deadband=CLASSIFICATION_DEADBAND,
        active_max_flat_seconds=ACTIVE_MAX_FLAT_SECONDS,
    )

    process_task(
        row=reg_row,
        split_bundle=split_data["regression"],
        task="regression",
        dataset_name_for_title=anchor_dataset_name,
        horizon_for_title=anchor_horizon,
        top_n=10,
    )

    process_task(
        row=clf_row,
        split_bundle=split_data["classification"],
        task="classification",
        dataset_name_for_title=anchor_dataset_name,
        horizon_for_title=anchor_horizon,
        top_n=10,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()