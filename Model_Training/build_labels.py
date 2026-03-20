from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
LABEL_DIR = PROJECT_ROOT / "data" / "labeled"

DEFAULT_PANEL_INTERVAL = "15s"
DEFAULT_HORIZON = "5min"

DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15
DEFAULT_TEST_FRAC = 0.15


def seconds_to_horizon_str(total_seconds: float) -> str:
    """
    Examples:
    0.1  -> '0.1s'
    0.5  -> '0.5s'
    30   -> '30s'
    45   -> '45s'
    60   -> '1min'
    75   -> '1min15s'
    300  -> '5min'
    """
    if total_seconds <= 0:
        raise ValueError("total_seconds must be positive.")

    if total_seconds < 60:
        return f"{total_seconds:g}s"

    minutes = int(total_seconds // 60)
    seconds = total_seconds - minutes * 60

    if abs(seconds) < 1e-12:
        return f"{minutes}min"

    if float(seconds).is_integer():
        return f"{minutes}min{int(seconds)}s"

    return f"{minutes}min{seconds:g}s"


def make_horizons(
    start_seconds: float = 30,
    end_seconds: float = 300,
    step_seconds: float = 15,
) -> list[str]:
    """Build horizons from start_seconds to end_seconds in step_seconds increments."""
    if start_seconds <= 0 or end_seconds <= 0 or step_seconds <= 0:
        raise ValueError("start_seconds, end_seconds, and step_seconds must all be positive.")
    if start_seconds > end_seconds:
        raise ValueError("start_seconds must be <= end_seconds.")

    span = end_seconds - start_seconds
    n_steps_float = span / step_seconds
    n_steps = int(round(n_steps_float))

    if abs(n_steps_float - n_steps) > 1e-9:
        raise ValueError("The range must be an exact multiple of step_seconds.")

    horizons = []
    for i in range(n_steps + 1):
        secs = round(start_seconds + i * step_seconds, 10)
        horizons.append(seconds_to_horizon_str(secs))

    return horizons


def _steps_from_horizon(panel_interval: str, horizon: str) -> int:
    """Convert a time horizon into number of panel rows ahead."""
    interval_td = pd.to_timedelta(panel_interval)
    horizon_td = pd.to_timedelta(horizon)

    if interval_td <= pd.Timedelta(0):
        raise ValueError("panel_interval must be positive.")
    if horizon_td <= pd.Timedelta(0):
        raise ValueError("horizon must be positive.")

    ratio = horizon_td / interval_td
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            f"Horizon {horizon} is not an exact multiple of panel interval {panel_interval}."
        )

    return int(round(ratio))


def add_time_split_column(
    df: pd.DataFrame,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> pd.DataFrame:
    """
    Add a chronological split column: train / val / test.
    Assumes df is already sorted by time.
    """
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


def build_labels_from_feature_df(
    feature_df: pd.DataFrame,
    panel_interval: str = DEFAULT_PANEL_INTERVAL,
    horizon: str = DEFAULT_HORIZON,
    drop_unlabeled: bool = True,
    add_split_col: bool = True,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> pd.DataFrame:
    """
    Add future-based regression target columns to one feature dataframe.

    New columns:
    - future_midprice:
        Midprice at time t + horizon
    - midprice_change_<horizon>:
        future_midprice - current midprice
    - target_<horizon>:
        same as midprice_change_<horizon>, explicit target column for modeling
    - split:
        train / val / test (time-based split)
    """
    df = feature_df.copy()

    if "ts" not in df.columns:
        raise ValueError("feature_df must contain a 'ts' column.")
    if "midprice" not in df.columns:
        raise ValueError("feature_df must contain a 'midprice' column.")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    steps_ahead = _steps_from_horizon(panel_interval=panel_interval, horizon=horizon)

    df["future_midprice"] = df["midprice"].shift(-steps_ahead)

    horizon_suffix = horizon.replace(" ", "")
    change_col = f"midprice_change_{horizon_suffix}"
    target_col = f"target_{horizon_suffix}"

    df[change_col] = df["future_midprice"] - df["midprice"]
    df[target_col] = df[change_col]

    if drop_unlabeled:
        df = df.dropna(subset=["future_midprice", change_col]).reset_index(drop=True)

    if add_split_col:
        df = add_time_split_column(
            df,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
        )

    return df


def build_labels_from_parquet(
    feature_path: str | Path,
    panel_interval: str = DEFAULT_PANEL_INTERVAL,
    horizon: str = DEFAULT_HORIZON,
    drop_unlabeled: bool = True,
    add_split_col: bool = True,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> pd.DataFrame:
    """Read one features parquet and add regression targets."""
    feature_path = Path(feature_path)
    df = pd.read_parquet(feature_path)

    return build_labels_from_feature_df(
        feature_df=df,
        panel_interval=panel_interval,
        horizon=horizon,
        drop_unlabeled=drop_unlabeled,
        add_split_col=add_split_col,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )


def save_labeled_table(
    feature_path: str | Path,
    output_path: str | Path,
    panel_interval: str = DEFAULT_PANEL_INTERVAL,
    horizon: str = DEFAULT_HORIZON,
    drop_unlabeled: bool = True,
    add_split_col: bool = True,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> None:
    """Build a labeled dataset from one features parquet and save it."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labeled_df = build_labels_from_parquet(
        feature_path=feature_path,
        panel_interval=panel_interval,
        horizon=horizon,
        drop_unlabeled=drop_unlabeled,
        add_split_col=add_split_col,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )

    labeled_df.to_parquet(output_path, engine="pyarrow", compression="zstd")
    print(f"Saved labeled table to {output_path}")
    print(labeled_df.head())


def build_all_labeled_tables(
    feature_dir: str | Path = FEATURE_DIR,
    output_dir: str | Path = LABEL_DIR,
    panel_interval: str = DEFAULT_PANEL_INTERVAL,
    horizon: str = DEFAULT_HORIZON,
    drop_unlabeled: bool = True,
    add_split_col: bool = True,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    force: bool = False,
) -> None:
    """Batch process all *-features.parquet files into labeled tables."""
    feature_dir = Path(feature_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_files = sorted(feature_dir.glob("*-features.parquet"))

    if not feature_files:
        print(f"No feature parquet files found in {feature_dir}")
        return

    print(f"Found {len(feature_files)} feature file(s). Panel interval = {panel_interval}\n")

    processed = 0
    skipped = 0
    failed = 0

    for i, feature_path in enumerate(feature_files, start=1):
        stem = feature_path.name.replace("-features.parquet", "")
        safe_horizon = horizon.replace(" ", "").replace("/", "_")
        output_path = output_dir / f"{stem}-labeled-{safe_horizon}.parquet"

        if output_path.exists() and not force:
            print(f"[{i}/{len(feature_files)}] Skipping {stem} (already exists)")
            skipped += 1
            continue

        print(f"[{i}/{len(feature_files)}] Building labels for {stem} | horizon={horizon}...")
        start = time.time()

        try:
            save_labeled_table(
                feature_path=feature_path,
                output_path=output_path,
                panel_interval=panel_interval,
                horizon=horizon,
                drop_unlabeled=drop_unlabeled,
                add_split_col=add_split_col,
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac,
            )
            elapsed = time.time() - start
            print(f"✅ Done {stem} ({elapsed:.2f}s)")
            processed += 1
        except Exception as e:
            print(f"❌ Failed {stem}: {e}")
            failed += 1

    print("\n===== SUMMARY =====")
    print(f"Panel interval: {panel_interval}")
    print(f"Horizon:        {horizon}")
    print(f"Train frac:     {train_frac}")
    print(f"Val frac:       {val_frac}")
    print(f"Test frac:      {test_frac}")
    print(f"Processed:      {processed}")
    print(f"Skipped:        {skipped}")
    print(f"Failed:         {failed}")
    print("===================")


def build_labels_for_horizons(
    horizons: list[str],
    feature_dir: str | Path = FEATURE_DIR,
    output_dir: str | Path = LABEL_DIR,
    panel_interval: str = DEFAULT_PANEL_INTERVAL,
    drop_unlabeled: bool = True,
    add_split_col: bool = True,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    force: bool = False,
) -> None:
    for horizon in horizons:
        print(f"\n{'=' * 80}")
        print(f"Building labels for horizon = {horizon}")

        build_all_labeled_tables(
            feature_dir=feature_dir,
            output_dir=output_dir,
            panel_interval=panel_interval,
            horizon=horizon,
            drop_unlabeled=drop_unlabeled,
            add_split_col=add_split_col,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            force=force,
        )


if __name__ == "__main__":
    horizons = make_horizons(start_seconds=30, end_seconds=300, step_seconds=15)

    build_labels_for_horizons(
        horizons=horizons,
        feature_dir=FEATURE_DIR,
        output_dir=LABEL_DIR,
        panel_interval=DEFAULT_PANEL_INTERVAL,
        drop_unlabeled=True,
        add_split_col=True,
        train_frac=0.70,
        val_frac=0.15,
        test_frac=0.15,
        force=True,
    )