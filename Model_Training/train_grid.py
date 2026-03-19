from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

LABEL_DIR = Path("data/labeled")
RESULT_DIR = Path("data/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Feature groups
# ----------------------------------------------------------------------
FEATURE_GROUPS = {
    "price": [
        "best_yes_bid",
        "best_no_bid",
        "best_yes_ask",
        "best_no_ask",
        "midprice",
        "spread",
    ],
    "local_depth": [
        "yes_depth_top1",
        "yes_depth_top3",
        "yes_depth_top5",
        "no_depth_top1",
        "no_depth_top3",
        "no_depth_top5",
    ],
    "total_depth": [
        "yes_total_depth",
        "no_total_depth",
        "total_depth",
    ],
    "imbalance": [
        "imbalance_top1",
        "imbalance_top3",
        "imbalance_top5",
    ],
}

def make_horizons(start_seconds: int = 60, end_seconds: int = 600, step_seconds: int = 30) -> list[str]:
    """
    Build horizons from 1min to 10min in 30-second steps.

    Examples:
    1min, 1min30s, 2min, 2min30s, ..., 10min
    """
    horizons = []

    for secs in range(start_seconds, end_seconds + 1, step_seconds):
        minutes = secs // 60
        seconds = secs % 60

        if seconds == 0:
            horizons.append(f"{minutes}min")
        else:
            horizons.append(f"{minutes}min{seconds}s")

    return horizons

# ----------------------------------------------------------------------
# Experiment settings
# ----------------------------------------------------------------------
HORIZONS = make_horizons(start_seconds=60, end_seconds=600, step_seconds=30)
MOVE_THRESHOLD = 0.02  # 2 cents for label construction
DECISION_THRESHOLDS = [round(x, 1) for x in np.arange(0.1, 1.01, 0.1)]

LOGREG_CONFIG = {
    "class_weight": "balanced",
    "C": 1.0,
    "max_iter": 3000,
    "random_state": 42,
}

TRAIN_SIZE = 0.60
VAL_SIZE = 0.20
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def horizon_to_seconds(horizon: str) -> int:
    """
    Convert strings like:
    1min -> 60
    5min30s -> 330
    10min -> 600
    """
    return int(pd.to_timedelta(horizon).total_seconds())

def threshold_suffix(move_threshold: float) -> str:
    """
    0.02 -> '2c'
    """
    return f"{int(round(move_threshold * 100))}c"


def make_label_col(horizon: str, move_threshold: float = MOVE_THRESHOLD) -> str:
    """
    Example:
    horizon='5min', move_threshold=0.02 -> 'label_up_5min_2c'
    """
    return f"label_up_{horizon}_{threshold_suffix(move_threshold)}"


def make_label_pattern(horizon: str, move_threshold: float = MOVE_THRESHOLD) -> str:
    """
    Example:
    horizon='5min', move_threshold=0.02 -> '*-labeled-5min-2c.parquet'
    """
    return f"*-labeled-{horizon}-{threshold_suffix(move_threshold)}.parquet"


def load_all_labeled_data(
    label_dir: str | Path = LABEL_DIR,
    pattern: str = "*-labeled.parquet",
) -> pd.DataFrame:
    """
    Load all labeled parquet files matching the pattern and concatenate them.
    """
    label_dir = Path(label_dir)
    files = sorted(label_dir.glob(pattern))

    if not files:
        raise ValueError(f"No labeled parquet files found in {label_dir} with pattern={pattern}")

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        df["source_file"] = f.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def generate_feature_set_combinations(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Generate all non-empty combinations of feature groups.

    4 groups -> 15 combinations.
    """
    group_names = list(FEATURE_GROUPS.keys())
    feature_sets = {}

    for r in range(1, len(group_names) + 1):
        for combo in combinations(group_names, r):
            name = "+".join(combo)
            cols = []

            for g in combo:
                cols.extend(FEATURE_GROUPS[g])

            # Keep only columns that actually exist in df
            cols = [c for c in cols if c in df.columns]
            cols = list(dict.fromkeys(cols))  # remove duplicates while preserving order

            if cols:
                feature_sets[name] = cols

    return feature_sets


def split_unique_groups(
    unique_groups: np.ndarray,
    second_part_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a list of unique market IDs into two disjoint parts.

    second_part_fraction = fraction assigned to the second output.
    """
    rng = np.random.default_rng(random_state)
    groups = np.array(unique_groups, copy=True)
    rng.shuffle(groups)

    n_second = max(1, int(round(len(groups) * second_part_fraction)))
    n_second = min(n_second, len(groups) - 1)  # ensure first part non-empty

    first_groups = groups[:-n_second]
    second_groups = groups[-n_second:]

    return first_groups, second_groups


def train_val_test_split_by_market(
    df: pd.DataFrame,
    label_col: str,
    train_size: float = TRAIN_SIZE,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by market_id, not by rows, to avoid leakage.
    """
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_size + val_size + test_size must sum to 1.")

    work_df = df.dropna(subset=["market_id", label_col]).copy()
    work_df[label_col] = work_df[label_col].astype(int)

    unique_markets = work_df["market_id"].dropna().unique()
    if len(unique_markets) < 3:
        raise ValueError("Need at least 3 distinct markets for train/val/test split.")

    # First: train vs temp(val+test)
    train_markets, temp_markets = split_unique_groups(
        unique_groups=unique_markets,
        second_part_fraction=(val_size + test_size),
        random_state=random_state,
    )

    # Then: val vs test inside temp
    test_fraction_inside_temp = test_size / (val_size + test_size)
    val_markets, test_markets = split_unique_groups(
        unique_groups=temp_markets,
        second_part_fraction=test_fraction_inside_temp,
        random_state=random_state + 1,
    )

    train_df = work_df[work_df["market_id"].isin(train_markets)].copy()
    val_df = work_df[work_df["market_id"].isin(val_markets)].copy()
    test_df = work_df[work_df["market_id"].isin(test_markets)].copy()

    return train_df, val_df, test_df


def build_logreg_model() -> Pipeline:
    """
    Single Logistic Regression family model.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**LOGREG_CONFIG)),
    ])


def safe_roc_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """
    ROC-AUC requires both classes to be present.
    """
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def safe_pr_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """
    PR-AUC requires at least one positive in y_true.
    """
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    return average_precision_score(y_true, y_prob)


def probability_metrics(y_true: pd.Series, y_prob: np.ndarray) -> dict:
    """
    Metrics that do NOT depend on the decision threshold.
    """
    return {
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "pr_auc": safe_pr_auc(y_true, y_prob),
    }


def threshold_metrics(y_true: pd.Series, y_prob: np.ndarray, decision_threshold: float) -> dict:
    """
    Metrics that DO depend on the decision threshold.
    """
    y_pred = (y_prob >= decision_threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def coefficient_table(model: Pipeline, feature_cols: list[str], horizon: str, feature_set_name: str) -> pd.DataFrame:
    """
    Return logistic regression coefficients.
    """
    clf = model.named_steps["clf"]
    coefs = clf.coef_[0]

    coef_df = pd.DataFrame({
        "horizon": horizon,
        "feature_set": feature_set_name,
        "feature": feature_cols,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False)

    return coef_df


def print_split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str) -> None:
    """
    Print split summary for debugging / sanity checks.
    """
    print("\n===== MARKET SPLIT =====")
    print(f"Train markets ({train_df['market_id'].nunique()}): {sorted(train_df['market_id'].unique())}")
    print(f"Val markets   ({val_df['market_id'].nunique()}): {sorted(val_df['market_id'].unique())}")
    print(f"Test markets  ({test_df['market_id'].nunique()}): {sorted(test_df['market_id'].unique())}")

    print("\n===== LABEL DISTRIBUTION =====")
    print("Train:")
    print(train_df[label_col].value_counts(normalize=True).sort_index())
    print("\nVal:")
    print(val_df[label_col].value_counts(normalize=True).sort_index())
    print("\nTest:")
    print(test_df[label_col].value_counts(normalize=True).sort_index())


# ----------------------------------------------------------------------
# Main experiment loop
# ----------------------------------------------------------------------
def run_all_experiments() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    threshold_rows = []
    coef_tables = []

    for horizon in HORIZONS:
        pattern = make_label_pattern(horizon=horizon, move_threshold=MOVE_THRESHOLD)
        label_col = make_label_col(horizon=horizon, move_threshold=MOVE_THRESHOLD)

        print("\n" + "=" * 100)
        print(f"RUNNING HORIZON = {horizon}")
        print(f"Pattern = {pattern}")
        print(f"Label   = {label_col}")

        df = load_all_labeled_data(label_dir=LABEL_DIR, pattern=pattern)

        if label_col not in df.columns:
            raise ValueError(f"Expected label column '{label_col}' not found for horizon={horizon}")

        feature_sets = generate_feature_set_combinations(df)

        train_df, val_df, test_df = train_val_test_split_by_market(
            df=df,
            label_col=label_col,
            train_size=TRAIN_SIZE,
            val_size=VAL_SIZE,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        print(f"Total rows: {len(df)}")
        print(f"Train rows: {len(train_df)}")
        print(f"Val rows:   {len(val_df)}")
        print(f"Test rows:  {len(test_df)}")
        print_split_summary(train_df, val_df, test_df, label_col)

        for feature_set_name, feature_cols in feature_sets.items():
            print("-" * 100)
            print(f"Horizon={horizon} | Feature set={feature_set_name} | n_features={len(feature_cols)}")

            X_train = train_df[feature_cols]
            y_train = train_df[label_col].astype(int)

            X_val = val_df[feature_cols]
            y_val = val_df[label_col].astype(int)

            X_test = test_df[feature_cols]
            y_test = test_df[label_col].astype(int)

            model = build_logreg_model()
            model.fit(X_train, y_train)

            val_prob = model.predict_proba(X_val)[:, 1]
            test_prob = model.predict_proba(X_test)[:, 1]

            val_prob_metrics = probability_metrics(y_val, val_prob)
            test_prob_metrics = probability_metrics(y_test, test_prob)

            # Save coefficients
            coef_tables.append(
                coefficient_table(
                    model=model,
                    feature_cols=feature_cols,
                    horizon=horizon,
                    feature_set_name=feature_set_name,
                )
            )

            # Sweep decision thresholds
            val_threshold_results = []
            for th in DECISION_THRESHOLDS:
                val_tm = threshold_metrics(y_val, val_prob, th)
                test_tm = threshold_metrics(y_test, test_prob, th)

                threshold_rows.append({
                    "horizon": horizon,
                    "horizon_seconds": horizon_to_seconds(horizon),
                    "feature_set": feature_set_name,
                    "n_features": len(feature_cols),
                    "decision_threshold": th,

                    "val_accuracy": val_tm["accuracy"],
                    "val_precision": val_tm["precision"],
                    "val_recall": val_tm["recall"],
                    "val_f1": val_tm["f1"],

                    "test_accuracy": test_tm["accuracy"],
                    "test_precision": test_tm["precision"],
                    "test_recall": test_tm["recall"],
                    "test_f1": test_tm["f1"],
                })

                val_threshold_results.append({
                    "decision_threshold": th,
                    "val_accuracy": val_tm["accuracy"],
                    "val_precision": val_tm["precision"],
                    "val_recall": val_tm["recall"],
                    "val_f1": val_tm["f1"],
                })

            # Pick best threshold by validation F1
            val_threshold_df = pd.DataFrame(val_threshold_results)
            val_threshold_df = val_threshold_df.sort_values(
                ["val_f1", "val_precision", "val_recall"],
                ascending=False
            ).reset_index(drop=True)

            best_threshold = float(val_threshold_df.iloc[0]["decision_threshold"])

            best_val_tm = threshold_metrics(y_val, val_prob, best_threshold)
            best_test_tm = threshold_metrics(y_test, test_prob, best_threshold)

            summary_rows.append({
                "horizon": horizon,
                "horizon_seconds": horizon_to_seconds(horizon),
                "feature_set": feature_set_name,
                "n_features": len(feature_cols),

                "roc_auc_val": val_prob_metrics["roc_auc"],
                "pr_auc_val": val_prob_metrics["pr_auc"],
                "roc_auc_test": test_prob_metrics["roc_auc"],
                "pr_auc_test": test_prob_metrics["pr_auc"],

                "best_decision_threshold_by_val_f1": best_threshold,

                "val_accuracy_at_best_threshold": best_val_tm["accuracy"],
                "val_precision_at_best_threshold": best_val_tm["precision"],
                "val_recall_at_best_threshold": best_val_tm["recall"],
                "val_f1_at_best_threshold": best_val_tm["f1"],

                "test_accuracy_at_best_threshold": best_test_tm["accuracy"],
                "test_precision_at_best_threshold": best_test_tm["precision"],
                "test_recall_at_best_threshold": best_test_tm["recall"],
                "test_f1_at_best_threshold": best_test_tm["f1"],
            })

            print(f"Best threshold by validation F1: {best_threshold:.2f}")
            print(f"Validation PR-AUC: {val_prob_metrics['pr_auc']:.4f} | Test PR-AUC: {test_prob_metrics['pr_auc']:.4f}")
            print(f"Validation F1 @ best threshold: {best_val_tm['f1']:.4f} | Test F1 @ best threshold: {best_test_tm['f1']:.4f}")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["pr_auc_test", "test_f1_at_best_threshold"],
        ascending=False
    ).reset_index(drop=True)

    threshold_df = pd.DataFrame(threshold_rows).sort_values(
        ["horizon", "feature_set", "decision_threshold"]
    ).reset_index(drop=True)

    coef_df = pd.concat(coef_tables, ignore_index=True)

    return summary_df, threshold_df, coef_df


if __name__ == "__main__":
    summary_df, threshold_df, coef_df = run_all_experiments()

    summary_path = RESULT_DIR / "logreg_summary.csv"
    threshold_path = RESULT_DIR / "logreg_threshold_sweep.csv"
    coef_path = RESULT_DIR / "logreg_coefficients.csv"

    summary_df.to_csv(summary_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)
    coef_df.to_csv(coef_path, index=False)

    print("\n" + "=" * 100)
    print("TOP RESULTS BY TEST PR-AUC")
    print(summary_df.head(20).to_string(index=False))

    print(f"\nSaved summary to:   {summary_path}")
    print(f"Saved threshold to: {threshold_path}")
    print(f"Saved coefficients: {coef_path}")