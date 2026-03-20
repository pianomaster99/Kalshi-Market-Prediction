from __future__ import annotations

from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
RESULT_DIR = PROJECT_ROOT / "data" / "results"
MODEL_DIR = PROJECT_ROOT / "data" / "models"

TIME_COL = "ts"
PRICE_COL = "midprice"

DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15
DEFAULT_TEST_FRAC = 0.15

RANDOM_STATE = 42
CV_SPLITS = 3
CLASSIFICATION_DEADBAND = 0.0
ACTIVE_MAX_FLAT_SECONDS = 30.0
MIN_CLASSIFICATION_TRAIN_ROWS = 50
FIXED_HORIZON = "5s"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def horizon_str_to_seconds(horizon: str) -> float:
    return pd.to_timedelta(horizon).total_seconds()


def sanitize_name(text: str) -> str:
    return (
        str(text)
        .replace(" ", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


# ---------------------------------------------------------------------
# Targets and active-period filter
# ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------
# Feature prep
# ---------------------------------------------------------------------


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


def make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    transformers = []

    if numeric_cols:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_onehot_encoder()),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No numeric or categorical features available for training.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_cols, categorical_cols


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


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------


def get_model_specs() -> dict[str, dict]:
    return {
        "ridge_reg": {
            "task": "regression",
            "estimator": Ridge(),
            "param_grid": {
                "model__alpha": [0.1, 1.0, 10.0],
            },
            "scoring": "neg_mean_absolute_error",
            "selection_metric": "val_mae",
            "selection_mode": "min",
        },
        "hist_gbrt_reg": {
            "task": "regression",
            "estimator": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            "param_grid": {
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [15, 31],
                "model__max_depth": [6],
                "model__min_samples_leaf": [20],
            },
            "scoring": "neg_mean_absolute_error",
            "selection_metric": "val_mae",
            "selection_mode": "min",
        },
        "logreg_clf": {
            "task": "classification",
            "estimator": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "model__C": [0.1, 1.0, 10.0],
            },
            "scoring": "average_precision",
            "selection_metric": "val_pr_auc",
            "selection_mode": "max",
        },
        "hist_gb_clf": {
            "task": "classification",
            "estimator": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [15, 31],
                "model__max_depth": [6],
                "model__min_samples_leaf": [20],
            },
            "scoring": "average_precision",
            "selection_metric": "val_pr_auc",
            "selection_mode": "max",
        },
    }


def build_pipeline(X_train: pd.DataFrame, model) -> Pipeline:
    preprocessor, _, _ = make_preprocessor(X_train)
    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", clone(model)),
        ]
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    return float((true_sign == pred_sign).mean())


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray | None) -> float:
    if y_prob is None:
        return np.nan
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_prob))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "sign_acc": sign_accuracy(y_true, y_pred),
        "corr": safe_corr(y_true, y_pred),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": safe_pr_auc(y_true, y_prob),
    }


# ---------------------------------------------------------------------
# Grid search helpers
# ---------------------------------------------------------------------


def pick_safe_time_series_splits_for_classification(
    y_train: pd.Series,
    max_splits: int = CV_SPLITS,
) -> int | None:
    y = pd.Series(y_train).reset_index(drop=True)

    if y.nunique(dropna=True) < 2:
        return None

    max_allowed = min(max_splits, len(y) - 1)
    if max_allowed < 2:
        return None

    for n_splits in range(max_allowed, 1, -1):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        ok = True
        for train_idx, _ in tscv.split(y):
            y_fold = y.iloc[train_idx]
            if y_fold.nunique(dropna=True) < 2:
                ok = False
                break
        if ok:
            return n_splits

    return None


def fit_one_model_with_gridsearch(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    cv_splits: int = CV_SPLITS,
    verbose: int = 0,
) -> GridSearchCV:
    model_specs = get_model_specs()
    if model_name not in model_specs:
        raise ValueError(f"Unknown model_name: {model_name}")

    spec = model_specs[model_name]
    pipe = build_pipeline(X_train, spec["estimator"])

    if spec["task"] == "classification":
        n_splits = pick_safe_time_series_splits_for_classification(y_train=y_train, max_splits=cv_splits)
        if n_splits is None:
            raise ValueError(
                "Classification target is not usable for TimeSeriesSplit CV "
                "(need both classes in every training fold)."
            )
    else:
        n_splits = min(cv_splits, len(X_train) - 1)
        if n_splits < 2:
            raise ValueError("Not enough train rows for TimeSeriesSplit.")

    cv = TimeSeriesSplit(n_splits=n_splits)

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=spec["param_grid"],
        scoring=spec["scoring"],
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=verbose,
        error_score="raise",
    )

    gs.fit(X_train, y_train)
    return gs


# ---------------------------------------------------------------------
# One file
# ---------------------------------------------------------------------


def choose_best_bundle(bundles: list[dict], selection_metric: str, selection_mode: str) -> dict:
    if selection_mode == "min":
        return min(bundles, key=lambda x: x["row"][selection_metric])
    if selection_mode == "max":
        return max(bundles, key=lambda x: x["row"][selection_metric])
    raise ValueError(f"Unknown selection_mode: {selection_mode}")


def make_skip_row(
    *,
    feature_path: Path,
    base_name: str,
    task: str,
    model_name: str,
    horizon: str,
    requested_horizon_seconds: float,
    realized_horizon_mean: float,
    panel_seconds: float | None,
    classification_deadband: float,
    active_max_flat_seconds: float,
    n_rows_total: int,
    n_train_task: int,
    n_val_task: int,
    n_test_task: int,
    n_features: int,
    constant_cols: list[str],
    error: str,
) -> dict:
    return {
        "source_file": feature_path.name,
        "dataset_name": base_name,
        "task": task,
        "model": model_name,
        "horizon": horizon,
        "requested_horizon_seconds": requested_horizon_seconds,
        "realized_horizon_mean_seconds": realized_horizon_mean,
        "panel_median_seconds": panel_seconds,
        "classification_deadband": classification_deadband,
        "active_max_flat_seconds": active_max_flat_seconds,
        "n_rows_total": n_rows_total,
        "n_train_task": n_train_task,
        "n_val_task": n_val_task,
        "n_test_task": n_test_task,
        "n_features": n_features,
        "n_constant_dropped": len(constant_cols),
        "constant_cols": json.dumps(constant_cols),
        "error": error,
    }


def run_one_horizon_on_one_file(
    feature_path: str | Path,
    horizon: str,
    models_to_try: list[str],
    output_result_dir: str | Path,
    output_model_dir: str | Path,
    time_col: str = TIME_COL,
    price_col: str = PRICE_COL,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    classification_deadband: float = CLASSIFICATION_DEADBAND,
    active_max_flat_seconds: float = ACTIVE_MAX_FLAT_SECONDS,
    verbose: int = 0,
) -> list[dict]:
    feature_path = Path(feature_path)
    output_result_dir = Path(output_result_dir)
    output_model_dir = Path(output_model_dir)

    output_result_dir.mkdir(parents=True, exist_ok=True)
    output_model_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_parquet(feature_path)
    panel_seconds = infer_panel_seconds(raw_df, time_col=time_col)

    ds_reg = build_future_change_target(
        feature_df=raw_df,
        horizon=horizon,
        time_col=time_col,
        price_col=price_col,
        drop_unlabeled=True,
    )
    ds_reg = add_time_split_column(ds_reg, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac)

    ds_clf = filter_long_flat_runs(
        ds_reg.copy(),
        price_col=price_col,
        time_col=time_col,
        max_flat_seconds=active_max_flat_seconds,
    )
    ds_clf = add_direction_target(ds_clf, horizon=horizon, deadband=classification_deadband)
    ds_clf = add_time_split_column(ds_clf, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac)

    horizon_suffix = sanitize_name(horizon)
    reg_target_col = f"target_{horizon_suffix}"
    clf_target_col = f"direction_target_{horizon_suffix}"
    realized_col = f"realized_horizon_seconds_{horizon_suffix}"

    feature_cols = get_feature_columns(
        ds_reg,
        target_cols=[reg_target_col, clf_target_col],
        time_col=time_col,
    )

    train_df_reg = ds_reg[ds_reg["split"] == "train"].copy()
    val_df_reg = ds_reg[ds_reg["split"] == "val"].copy()
    test_df_reg = ds_reg[ds_reg["split"] == "test"].copy()

    train_df_clf = ds_clf[ds_clf["split"] == "train"].copy()
    val_df_clf = ds_clf[ds_clf["split"] == "val"].copy()
    test_df_clf = ds_clf[ds_clf["split"] == "test"].copy()

    if len(train_df_reg) < 50:
        raise ValueError(f"Too few regression train rows after target build: {len(train_df_reg)}")

    X_train_reg_raw = train_df_reg[feature_cols]
    X_val_reg_raw = val_df_reg[feature_cols]
    X_test_reg_raw = test_df_reg[feature_cols]
    X_train_reg, X_val_reg, X_test_reg, constant_cols_reg = drop_constant_columns(
        X_train_reg_raw, X_val_reg_raw, X_test_reg_raw
    )
    if X_train_reg.shape[1] == 0:
        raise ValueError("All regression feature columns were constant in the train split.")

    X_train_clf_raw = train_df_clf[feature_cols]
    X_val_clf_raw = val_df_clf[feature_cols]
    X_test_clf_raw = test_df_clf[feature_cols]
    X_train_clf, X_val_clf, X_test_clf, constant_cols_clf = drop_constant_columns(
        X_train_clf_raw, X_val_clf_raw, X_test_clf_raw
    )

    y_train_reg = train_df_reg[reg_target_col]
    y_val_reg = val_df_reg[reg_target_col]
    y_test_reg = test_df_reg[reg_target_col]

    base_name = feature_path.stem.replace("-features", "")
    requested_horizon_seconds = horizon_str_to_seconds(horizon)
    realized_horizon_mean_reg = float(ds_reg[realized_col].mean()) if realized_col in ds_reg.columns else np.nan
    realized_horizon_mean_clf = float(ds_clf[realized_col].mean()) if realized_col in ds_clf.columns else np.nan

    rows: list[dict] = []
    fitted_bundles: list[dict] = []

    model_specs = get_model_specs()

    for model_name in models_to_try:
        spec = model_specs[model_name]
        task = spec["task"]
        print(f"    -> fitting {model_name} ...")
        start = time.time()

        if task == "regression":
            X_train_task, y_train_task = X_train_reg, y_train_reg
            X_val_task, y_val_task = X_val_reg, y_val_reg
            X_test_task, y_test_task = X_test_reg, y_test_reg
            ds_task = ds_reg
            realized_horizon_mean = realized_horizon_mean_reg
            constant_cols = constant_cols_reg
        else:
            train_mask = train_df_clf[clf_target_col].notna()
            val_mask = val_df_clf[clf_target_col].notna()
            test_mask = test_df_clf[clf_target_col].notna()

            X_train_task = X_train_clf.loc[train_mask]
            y_train_task = train_df_clf.loc[train_mask, clf_target_col].astype(int)
            X_val_task = X_val_clf.loc[val_mask]
            y_val_task = val_df_clf.loc[val_mask, clf_target_col].astype(int)
            X_test_task = X_test_clf.loc[test_mask]
            y_test_task = test_df_clf.loc[test_mask, clf_target_col].astype(int)

            ds_task = ds_clf
            realized_horizon_mean = realized_horizon_mean_clf
            constant_cols = constant_cols_clf

            if X_train_clf.shape[1] == 0:
                msg = "Skipped classification: all feature columns were constant in the filtered train split."
                rows.append(
                    make_skip_row(
                        feature_path=feature_path,
                        base_name=base_name,
                        task=task,
                        model_name=model_name,
                        horizon=horizon,
                        requested_horizon_seconds=requested_horizon_seconds,
                        realized_horizon_mean=realized_horizon_mean,
                        panel_seconds=panel_seconds,
                        classification_deadband=classification_deadband,
                        active_max_flat_seconds=active_max_flat_seconds,
                        n_rows_total=len(ds_clf),
                        n_train_task=len(X_train_task),
                        n_val_task=len(X_val_task),
                        n_test_task=len(X_test_task),
                        n_features=0,
                        constant_cols=constant_cols,
                        error=msg,
                    )
                )
                continue

            if len(X_train_task) < MIN_CLASSIFICATION_TRAIN_ROWS:
                msg = f"Skipped classification: too few train rows after filtering ({len(X_train_task)})."
                rows.append(
                    make_skip_row(
                        feature_path=feature_path,
                        base_name=base_name,
                        task=task,
                        model_name=model_name,
                        horizon=horizon,
                        requested_horizon_seconds=requested_horizon_seconds,
                        realized_horizon_mean=realized_horizon_mean,
                        panel_seconds=panel_seconds,
                        classification_deadband=classification_deadband,
                        active_max_flat_seconds=active_max_flat_seconds,
                        n_rows_total=len(ds_clf),
                        n_train_task=len(X_train_task),
                        n_val_task=len(X_val_task),
                        n_test_task=len(X_test_task),
                        n_features=X_train_task.shape[1],
                        constant_cols=constant_cols,
                        error=msg,
                    )
                )
                continue

            if y_train_task.nunique(dropna=True) < 2:
                msg = "Skipped classification: train target has only one class."
                rows.append(
                    make_skip_row(
                        feature_path=feature_path,
                        base_name=base_name,
                        task=task,
                        model_name=model_name,
                        horizon=horizon,
                        requested_horizon_seconds=requested_horizon_seconds,
                        realized_horizon_mean=realized_horizon_mean,
                        panel_seconds=panel_seconds,
                        classification_deadband=classification_deadband,
                        active_max_flat_seconds=active_max_flat_seconds,
                        n_rows_total=len(ds_clf),
                        n_train_task=len(X_train_task),
                        n_val_task=len(X_val_task),
                        n_test_task=len(X_test_task),
                        n_features=X_train_task.shape[1],
                        constant_cols=constant_cols,
                        error=msg,
                    )
                )
                continue

        try:
            gs = fit_one_model_with_gridsearch(
                X_train=X_train_task,
                y_train=y_train_task,
                model_name=model_name,
                verbose=verbose,
            )
        except Exception as e:
            msg = f"Fit skipped/failed: {e}"
            rows.append(
                make_skip_row(
                    feature_path=feature_path,
                    base_name=base_name,
                    task=task,
                    model_name=model_name,
                    horizon=horizon,
                    requested_horizon_seconds=requested_horizon_seconds,
                    realized_horizon_mean=realized_horizon_mean,
                    panel_seconds=panel_seconds,
                    classification_deadband=classification_deadband,
                    active_max_flat_seconds=active_max_flat_seconds,
                    n_rows_total=len(ds_task),
                    n_train_task=len(X_train_task),
                    n_val_task=len(X_val_task),
                    n_test_task=len(X_test_task),
                    n_features=X_train_task.shape[1] if hasattr(X_train_task, "shape") else 0,
                    constant_cols=constant_cols,
                    error=msg,
                )
            )
            continue

        best_estimator = gs.best_estimator_

        if task == "regression":
            train_pred = best_estimator.predict(X_train_task)
            val_pred = best_estimator.predict(X_val_task)
            test_pred = best_estimator.predict(X_test_task)

            train_metrics = regression_metrics(y_train_task.to_numpy(), train_pred)
            val_metrics = regression_metrics(y_val_task.to_numpy(), val_pred)
            test_metrics = regression_metrics(y_test_task.to_numpy(), test_pred)

            row = {
                "source_file": feature_path.name,
                "dataset_name": base_name,
                "task": task,
                "model": model_name,
                "horizon": horizon,
                "requested_horizon_seconds": requested_horizon_seconds,
                "realized_horizon_mean_seconds": realized_horizon_mean,
                "panel_median_seconds": panel_seconds,
                "classification_deadband": classification_deadband,
                "active_max_flat_seconds": active_max_flat_seconds,
                "n_rows_total": len(ds_reg),
                "n_train_task": len(X_train_task),
                "n_val_task": len(X_val_task),
                "n_test_task": len(X_test_task),
                "n_features": X_train_task.shape[1],
                "n_constant_dropped": len(constant_cols),
                "constant_cols": json.dumps(constant_cols),
                "cv_best_mae": float(-gs.best_score_),
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "train_sign_acc": train_metrics["sign_acc"],
                "train_corr": train_metrics["corr"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "val_sign_acc": val_metrics["sign_acc"],
                "val_corr": val_metrics["corr"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "test_sign_acc": test_metrics["sign_acc"],
                "test_corr": test_metrics["corr"],
                "best_params": json.dumps(gs.best_params_, default=str),
                "fit_seconds": time.time() - start,
                "error": "",
            }
            rows.append(row)
            fitted_bundles.append(
                {
                    "task": task,
                    "row": row,
                    "model_name": model_name,
                    "best_estimator": best_estimator,
                    "val_pred": val_pred,
                    "test_pred": test_pred,
                    "val_index": y_val_task.index,
                    "test_index": y_test_task.index,
                    "ds_task": ds_reg,
                }
            )
        else:
            train_pred = best_estimator.predict(X_train_task)
            val_pred = best_estimator.predict(X_val_task)
            test_pred = best_estimator.predict(X_test_task)

            train_prob = best_estimator.predict_proba(X_train_task)[:, 1]
            val_prob = best_estimator.predict_proba(X_val_task)[:, 1]
            test_prob = best_estimator.predict_proba(X_test_task)[:, 1]

            train_metrics = classification_metrics(y_train_task.to_numpy(), train_pred, train_prob)
            val_metrics = classification_metrics(y_val_task.to_numpy(), val_pred, val_prob)
            test_metrics = classification_metrics(y_test_task.to_numpy(), test_pred, test_prob)

            row = {
                "source_file": feature_path.name,
                "dataset_name": base_name,
                "task": task,
                "model": model_name,
                "horizon": horizon,
                "requested_horizon_seconds": requested_horizon_seconds,
                "realized_horizon_mean_seconds": realized_horizon_mean,
                "panel_median_seconds": panel_seconds,
                "classification_deadband": classification_deadband,
                "active_max_flat_seconds": active_max_flat_seconds,
                "n_rows_total": len(ds_clf),
                "n_train_task": len(X_train_task),
                "n_val_task": len(X_val_task),
                "n_test_task": len(X_test_task),
                "n_features": X_train_task.shape[1],
                "n_constant_dropped": len(constant_cols),
                "constant_cols": json.dumps(constant_cols),
                "cv_best_pr_auc": float(gs.best_score_),
                "train_acc": train_metrics["acc"],
                "train_bal_acc": train_metrics["bal_acc"],
                "train_f1": train_metrics["f1"],
                "train_pr_auc": train_metrics["pr_auc"],
                "val_acc": val_metrics["acc"],
                "val_bal_acc": val_metrics["bal_acc"],
                "val_f1": val_metrics["f1"],
                "val_pr_auc": val_metrics["pr_auc"],
                "test_acc": test_metrics["acc"],
                "test_bal_acc": test_metrics["bal_acc"],
                "test_f1": test_metrics["f1"],
                "test_pr_auc": test_metrics["pr_auc"],
                "best_params": json.dumps(gs.best_params_, default=str),
                "fit_seconds": time.time() - start,
                "error": "",
            }
            rows.append(row)
            fitted_bundles.append(
                {
                    "task": task,
                    "row": row,
                    "model_name": model_name,
                    "best_estimator": best_estimator,
                    "val_pred": val_pred,
                    "test_pred": test_pred,
                    "val_prob": val_prob,
                    "test_prob": test_prob,
                    "val_index": y_val_task.index,
                    "test_index": y_test_task.index,
                    "ds_task": ds_clf,
                }
            )

    for task in ["regression", "classification"]:
        bundles = [b for b in fitted_bundles if b["task"] == task]
        if not bundles:
            continue

        first_spec = model_specs[bundles[0]["model_name"]]
        best_bundle = choose_best_bundle(
            bundles,
            selection_metric=first_spec["selection_metric"],
            selection_mode=first_spec["selection_mode"],
        )
        best_model_name = best_bundle["model_name"]
        ds_best = best_bundle["ds_task"]

        model_path = output_model_dir / f"{base_name}-{horizon_suffix}-{task}-{best_model_name}.joblib"
        joblib.dump(best_bundle["best_estimator"], model_path)

        if task == "regression":
            pred_df = pd.concat(
                [
                    ds_best.loc[best_bundle["val_index"], [time_col, price_col, reg_target_col, "future_midprice", "future_ts"]]
                    .assign(
                        split="val",
                        prediction=best_bundle["val_pred"],
                        best_model=best_model_name,
                        task=task,
                        horizon=horizon,
                    ),
                    ds_best.loc[best_bundle["test_index"], [time_col, price_col, reg_target_col, "future_midprice", "future_ts"]]
                    .assign(
                        split="test",
                        prediction=best_bundle["test_pred"],
                        best_model=best_model_name,
                        task=task,
                        horizon=horizon,
                    ),
                ],
                axis=0,
            ).reset_index(drop=True)
            pred_df["prediction_error"] = pred_df["prediction"] - pred_df[reg_target_col]
        else:
            pred_df = pd.concat(
                [
                    ds_best.loc[best_bundle["val_index"], [time_col, price_col, clf_target_col, "future_midprice", "future_ts"]]
                    .assign(
                        split="val",
                        prediction=best_bundle["val_pred"],
                        prediction_proba=best_bundle["val_prob"],
                        best_model=best_model_name,
                        task=task,
                        horizon=horizon,
                    ),
                    ds_best.loc[best_bundle["test_index"], [time_col, price_col, clf_target_col, "future_midprice", "future_ts"]]
                    .assign(
                        split="test",
                        prediction=best_bundle["test_pred"],
                        prediction_proba=best_bundle["test_prob"],
                        best_model=best_model_name,
                        task=task,
                        horizon=horizon,
                    ),
                ],
                axis=0,
            ).reset_index(drop=True)

        pred_path = output_result_dir / f"{base_name}-{horizon_suffix}-{task}-best-preds.parquet"
        pred_df.to_parquet(pred_path, engine="pyarrow", compression="zstd")

        for row in rows:
            if row.get("task") != task:
                continue
            row[f"is_best_for_horizon_{task}"] = int(row.get("model") == best_model_name and not row.get("error"))
            row[f"saved_model_path_{task}"] = str(model_path) if row.get("model") == best_model_name and not row.get("error") else ""
            row[f"saved_pred_path_{task}"] = str(pred_path) if row.get("model") == best_model_name and not row.get("error") else ""

    return rows


# ---------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------


def run_full_experiment(
    feature_dir: str | Path = FEATURE_DIR,
    result_dir: str | Path = RESULT_DIR / "mixed_tasks_active_5s",
    model_dir: str | Path = MODEL_DIR / "mixed_tasks_active_5s",
    horizon: str = FIXED_HORIZON,
    models_to_try: list[str] | None = None,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    classification_deadband: float = CLASSIFICATION_DEADBAND,
    active_max_flat_seconds: float = ACTIVE_MAX_FLAT_SECONDS,
    verbose: int = 0,
) -> pd.DataFrame:
    feature_dir = Path(feature_dir)
    result_dir = Path(result_dir)
    model_dir = Path(model_dir)

    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if models_to_try is None:
        models_to_try = [
            "ridge_reg",
            "logreg_clf",
        ]

    feature_files = sorted(feature_dir.glob("*-features.parquet"))
    feature_files = feature_files[:10]
    if not feature_files:
        raise FileNotFoundError(f"No *-features.parquet files found in {feature_dir}")

    all_rows: list[dict] = []

    print(f"Found {len(feature_files)} feature file(s).")
    print(f"Running fixed horizon: {horizon}")
    print(f"Models: {models_to_try}")
    print(f"Classification deadband: {classification_deadband}")
    print(f"Active max flat seconds: {active_max_flat_seconds}")
    print()

    for feature_idx, feature_path in enumerate(feature_files, start=1):
        print("=" * 100)
        print(f"[FILE {feature_idx}/{len(feature_files)}] {feature_path.name}")
        start = time.time()

        try:
            rows = run_one_horizon_on_one_file(
                feature_path=feature_path,
                horizon=horizon,
                models_to_try=models_to_try,
                output_result_dir=result_dir / "predictions",
                output_model_dir=model_dir,
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac,
                classification_deadband=classification_deadband,
                active_max_flat_seconds=active_max_flat_seconds,
                verbose=verbose,
            )
            all_rows.extend(rows)

            elapsed = time.time() - start
            rows_df = pd.DataFrame(rows)
            good_rows = rows_df[rows_df.get("error", "") == ""].copy() if "error" in rows_df.columns else rows_df.copy()
            reg_rows = good_rows[good_rows["task"] == "regression"] if not good_rows.empty else pd.DataFrame()
            clf_rows = good_rows[good_rows["task"] == "classification"] if not good_rows.empty else pd.DataFrame()

            msg_parts = []
            if not reg_rows.empty:
                best_reg = reg_rows.sort_values("val_mae").iloc[0]
                msg_parts.append(
                    f"reg best={best_reg['model']} val_MAE={best_reg['val_mae']:.6f} test_sign_acc={best_reg['test_sign_acc']:.4f}"
                )
            if not clf_rows.empty:
                best_clf = clf_rows.sort_values("val_pr_auc", ascending=False).iloc[0]
                msg_parts.append(
                    f"clf best={best_clf['model']} val_PR_AUC={best_clf['val_pr_auc']:.4f} test_PR_AUC={best_clf['test_pr_auc']:.4f}"
                )
            if not msg_parts:
                msg_parts.append("no successful models")

            print(f"  ✔ {' | '.join(msg_parts)} | time={elapsed:.2f}s")

        except Exception as e:
            print(f"  ✘ failed for file={feature_path.name}: {e}")
            all_rows.append(
                {
                    "source_file": feature_path.name,
                    "dataset_name": feature_path.stem.replace("-features", ""),
                    "task": "FAILED",
                    "model": "FAILED",
                    "horizon": horizon,
                    "requested_horizon_seconds": horizon_str_to_seconds(horizon),
                    "error": str(e),
                }
            )

        summary_df = pd.DataFrame(all_rows)
        summary_df.to_csv(result_dir / "grid_summary.csv", index=False)

    summary_df = pd.DataFrame(all_rows)
    summary_path = result_dir / "grid_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if not summary_df.empty:
        good_df = summary_df[(summary_df["model"] != "FAILED") & (summary_df.get("error", "") == "")].copy()
        if not good_df.empty:
            best_df_parts = []
            reg_df = good_df[good_df["task"] == "regression"]
            clf_df = good_df[good_df["task"] == "classification"]

            if not reg_df.empty:
                best_df_parts.append(
                    reg_df.sort_values(["source_file", "val_mae"])
                    .groupby(["source_file"], as_index=False)
                    .head(1)
                )
            if not clf_df.empty:
                best_df_parts.append(
                    clf_df.sort_values(["source_file", "val_pr_auc"], ascending=[True, False])
                    .groupby(["source_file"], as_index=False)
                    .head(1)
                )

            if best_df_parts:
                best_df = pd.concat(best_df_parts, axis=0).reset_index(drop=True)
                best_path = result_dir / "best_by_file_and_task.csv"
                best_df.to_csv(best_path, index=False)
                print(f"\nSaved summary to: {summary_path}")
                print(f"Saved best-by-file-and-task to: {best_path}")
            else:
                print(f"\nSaved summary to: {summary_path}")
        else:
            print(f"\nSaved summary to: {summary_path}")
    else:
        print(f"\nSaved summary to: {summary_path}")

    return summary_df


if __name__ == "__main__":
    MODELS_TO_TRY = [
        "ridge_reg",
        "logreg_clf",
        "hist_gbrt_reg",
        "hist_gb_clf"
    ]

    run_full_experiment(
        feature_dir=FEATURE_DIR,
        result_dir=RESULT_DIR / "mixed_tasks_active_5s",
        model_dir=MODEL_DIR / "mixed_tasks_active_5s",
        horizon=FIXED_HORIZON,
        models_to_try=MODELS_TO_TRY,
        train_frac=0.70,
        val_frac=0.15,
        test_frac=0.15,
        classification_deadband=0.0,
        active_max_flat_seconds=30.0,
        verbose=0,
    )
