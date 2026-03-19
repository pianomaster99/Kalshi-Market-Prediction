from pathlib import Path
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
LABEL_DIR = PROJECT_ROOT / "data" / "labeled"
FEATURE_DIR = Path("data/features")



def _steps_from_horizon(panel_interval: str, horizon: str) -> int:
    """
    Convert a time horizon into number of rows ahead.

    Example:
    - panel_interval = "30s"
    - horizon = "5min"
    Then:
    5 minutes / 30 seconds = 10 steps
    """
    interval_td = pd.to_timedelta(panel_interval)
    horizon_td = pd.to_timedelta(horizon)

    if interval_td <= pd.Timedelta(0):
        raise ValueError("panel_interval must be positive.")
    if horizon_td <= pd.Timedelta(0):
        raise ValueError("horizon must be positive.")

    ratio = horizon_td / interval_td

    # We want exact integer step alignment
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            f"Horizon {horizon} is not an exact multiple of panel interval {panel_interval}."
        )

    return int(round(ratio))


def build_labels_from_feature_df(
    feature_df: pd.DataFrame,
    panel_interval: str = "30s",
    horizon: str = "5min",
    threshold: float = 0.02,
    drop_unlabeled: bool = True,
) -> pd.DataFrame:
    """
    Add future-based label columns to one feature dataframe.

    New columns:
    - future_midprice:
        Midprice at time t + horizon

    - midprice_change:
        future_midprice - current midprice

    - label_up:
        1 if future midprice increases by at least threshold
        0 otherwise
    """
    df = feature_df.copy()

    if "ts" not in df.columns:
        raise ValueError("feature_df must contain a 'ts' column.")
    if "midprice" not in df.columns:
        raise ValueError("feature_df must contain a 'midprice' column.")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    steps_ahead = _steps_from_horizon(panel_interval=panel_interval, horizon=horizon)

    # ------------------------------------------------------------------
    # future_midprice:
    # the midprice observed exactly horizon steps in the future
    # ------------------------------------------------------------------
    df["future_midprice"] = df["midprice"].shift(-steps_ahead)

    # ------------------------------------------------------------------
    # midprice_change:
    # how much the future midprice differs from the current midprice
    # ------------------------------------------------------------------
    horizon_suffix = horizon.replace(" ", "")
    threshold_suffix = str(int(round(threshold * 100))) + "c"

    change_col = f"midprice_change_{horizon_suffix}"
    label_col = f"label_up_{horizon_suffix}_{threshold_suffix}"

    df[change_col] = df["future_midprice"] - df["midprice"]

    # ------------------------------------------------------------------
    # label_up:
    # 1 if future price move is at least threshold
    # 0 otherwise
    # ------------------------------------------------------------------
    df[label_col] = np.where(
        df[change_col].notna(),
        (df[change_col] >= threshold).astype(int),
        np.nan,
    )

    # Optionally drop rows near market end where future label is unavailable
    if drop_unlabeled:
        df = df.dropna(subset=["future_midprice", label_col]).reset_index(drop=True)
        df[label_col] = df[label_col].astype(int)

    return df


def build_labels_from_parquet(
    feature_path: str | Path,
    panel_interval: str = "30s",
    horizon: str = "5min",
    threshold: float = 0.02,
    drop_unlabeled: bool = True,
) -> pd.DataFrame:
    """
    Read one features parquet and add labels.
    """
    feature_path = Path(feature_path)
    df = pd.read_parquet(feature_path)

    return build_labels_from_feature_df(
        feature_df=df,
        panel_interval=panel_interval,
        horizon=horizon,
        threshold=threshold,
        drop_unlabeled=drop_unlabeled,
    )


def save_labeled_table(
    feature_path: str | Path,
    output_path: str | Path,
    panel_interval: str = "30s",
    horizon: str = "5min",
    threshold: float = 0.02,
    drop_unlabeled: bool = True,
) -> None:
    """
    Build labeled dataset from one feature parquet and save it.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labeled_df = build_labels_from_parquet(
        feature_path=feature_path,
        panel_interval=panel_interval,
        horizon=horizon,
        threshold=threshold,
        drop_unlabeled=drop_unlabeled,
    )

    labeled_df.to_parquet(output_path, engine="pyarrow", compression="zstd")
    print(f"Saved labeled table to {output_path}")
    print(labeled_df.head())
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

def build_labels_for_horizons(
    horizons: list[str],
    feature_dir: str | Path = FEATURE_DIR,
    output_dir: str | Path = LABEL_DIR,
    panel_interval: str = "30s",
    threshold: float = 0.02,
    drop_unlabeled: bool = True,
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
            threshold=threshold,
            drop_unlabeled=drop_unlabeled,
            force=force,
        )

def build_all_labeled_tables(
    feature_dir: str | Path = FEATURE_DIR,
    output_dir: str | Path = LABEL_DIR,
    panel_interval: str = "30s",
    horizon: str = "5min",
    threshold: float = 0.02,
    drop_unlabeled: bool = True,
    force: bool = False,
) -> None:
    """
    Batch process all *-features.parquet files into labeled tables.
    """
    feature_dir = Path(feature_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_files = sorted(feature_dir.glob("*-features.parquet"))

    if not feature_files:
        print(f"No feature parquet files found in {feature_dir}")
        return

    print(f"Found {len(feature_files)} feature file(s).\n")

    processed = 0
    skipped = 0
    failed = 0

    for i, feature_path in enumerate(feature_files, start=1):
        stem = feature_path.name.replace("-features.parquet", "")
        safe_horizon = horizon.replace(" ", "").replace("/", "_")
        safe_threshold = str(int(round(threshold * 100))) + "c"
        output_path = output_dir / f"{stem}-labeled-{safe_horizon}-{safe_threshold}.parquet"

        if output_path.exists() and not force:
            print(f"[{i}/{len(feature_files)}] Skipping {stem} (already exists)")
            skipped += 1
            continue

        print(f"[{i}/{len(feature_files)}] Building labels for {stem}...")
        start = time.time()

        try:
            save_labeled_table(
                feature_path=feature_path,
                output_path=output_path,
                panel_interval=panel_interval,
                horizon=horizon,
                threshold=threshold,
                drop_unlabeled=drop_unlabeled,
            )
            elapsed = time.time() - start
            print(f"✅ Done {stem} ({elapsed:.2f}s)")
            processed += 1

        except Exception as e:
            print(f"❌ Failed {stem}: {e}")
            failed += 1

    print("\n===== SUMMARY =====")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print("===================")

def make_horizons(start_seconds: int = 60, end_seconds: int = 600, step_seconds: int = 30) -> list[str]:
    """
    Build horizons from 1min to 10min in 30-second steps.
    Example outputs:
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

if __name__ == "__main__":
    horizons = make_horizons(start_seconds=60, end_seconds=600, step_seconds=30)

    build_labels_for_horizons(
        horizons=horizons,
        feature_dir=FEATURE_DIR,
        output_dir=LABEL_DIR,
        panel_interval="30s",
        threshold=0.02,
        drop_unlabeled=True,
        force=True,
    )