from pathlib import Path
import time
import argparse

import numpy as np
import pandas as pd


TIME_PANEL_DIR = Path("data/time_panel")
FEATURE_DIR = Path("data/features")
DEFAULT_INTERVAL = "1s"


def interval_to_filename_suffix(interval: str) -> str:
    return (
        interval.strip()
        .replace(" ", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "p")
    )


def seconds_to_feature_suffix(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    if total_ms <= 0:
        raise ValueError("seconds must be positive")

    if total_ms % 1000 == 0:
        return f"{total_ms // 1000}s"
    return f"{total_ms}ms"


def parse_float_tuple(text: str) -> tuple[float, ...]:
    if not text.strip():
        return tuple()
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def _extract_price_levels(columns: list[str], prefix: str) -> list[int]:
    levels = []
    for col in columns:
        if col.startswith(prefix):
            try:
                levels.append(int(col.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return sorted(levels)


def _best_bid_vectorized(book_array: np.ndarray, price_levels: list[int]) -> np.ndarray:
    reversed_book = book_array[:, ::-1]
    positive = reversed_book > 0
    has_any = positive.any(axis=1)
    first_idx = positive.argmax(axis=1)

    reversed_levels = np.array(price_levels[::-1], dtype=float) / 100.0
    best = np.full(book_array.shape[0], np.nan, dtype=float)
    best[has_any] = reversed_levels[first_idx[has_any]]
    return best


def _top_k_depth_vectorized(book_array: np.ndarray, k: int) -> np.ndarray:
    reversed_book = book_array[:, ::-1]
    positive = reversed_book > 0
    positive_rank = positive.cumsum(axis=1)
    include_mask = positive & (positive_rank <= k)
    return np.where(include_mask, reversed_book, 0.0).sum(axis=1)


def _safe_imbalance_series(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = a + b
    out = (a - b) / denom
    out = out.where((denom != 0) & a.notna() & b.notna())
    return out


def _derive_market_id_from_panel_name(panel_path: Path, interval: str) -> str:
    interval_suffix = interval_to_filename_suffix(interval)
    target = f"-base-panel-{interval_suffix}.parquet"
    name = panel_path.name

    if name.endswith(target):
        return name[: -len(target)]

    return name.replace("-base-panel.parquet", "")


def _seconds_to_steps(panel_interval: str, seconds: float) -> int:
    panel_td = pd.to_timedelta(panel_interval)
    target_td = pd.to_timedelta(f"{seconds}s")

    if panel_td <= pd.Timedelta(0):
        raise ValueError("panel_interval must be positive")
    if target_td <= pd.Timedelta(0):
        raise ValueError("seconds must be positive")

    ratio = target_td / panel_td
    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            f"Window {seconds}s is not an exact multiple of panel interval {panel_interval}"
        )
    return int(round(ratio))


def build_features_from_panel(
    panel_df: pd.DataFrame,
    market_id: str | None = None,
    panel_interval: str = DEFAULT_INTERVAL,
    include_imbalance: bool = True,
    add_recent_dynamics: bool = True,
    lag_seconds: tuple[float, ...] = (1.0,),
    rolling_window_seconds: tuple[float, ...] = (3.0,),
) -> pd.DataFrame:
    """
    Lean feature builder for 5-second prediction.
    Keeps the most interpretable microstructure features and a small number
    of recent-dynamics features, without changing the original time panel.
    """
    df = panel_df.copy()

    if "ts" not in df.columns:
        raise ValueError("panel_df must contain a 'ts' column.")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    if df.empty:
        raise ValueError("panel_df is empty after timestamp cleaning.")

    if market_id is None and "market_id" in df.columns:
        unique_ids = df["market_id"].dropna().unique()
        if len(unique_ids) == 1:
            market_id = unique_ids[0]

    yes_cols = sorted([c for c in df.columns if c.startswith("yes_")])
    no_cols = sorted([c for c in df.columns if c.startswith("no_")])
    if not yes_cols or not no_cols:
        raise ValueError("Expected yes_* and no_* columns in panel_df.")

    yes_levels = _extract_price_levels(yes_cols, "yes_")
    no_levels = _extract_price_levels(no_cols, "no_")

    yes_arr = df[yes_cols].to_numpy(dtype=float)
    no_arr = df[no_cols].to_numpy(dtype=float)

    feature_df = pd.DataFrame(index=df.index)
    feature_df["ts"] = df["ts"]
    if market_id is not None:
        feature_df["market_id"] = market_id

    best_yes_bid = _best_bid_vectorized(yes_arr, yes_levels)
    best_no_bid = _best_bid_vectorized(no_arr, no_levels)

    best_yes_ask = np.where(np.isfinite(best_no_bid), 1.0 - best_no_bid, np.nan)
    best_no_ask = np.where(np.isfinite(best_yes_bid), 1.0 - best_yes_bid, np.nan)

    midprice = np.where(
        np.isfinite(best_yes_bid) & np.isfinite(best_yes_ask),
        (best_yes_bid + best_yes_ask) / 2.0,
        np.nan,
    )
    spread = np.where(
        np.isfinite(best_yes_bid) & np.isfinite(best_yes_ask),
        best_yes_ask - best_yes_bid,
        np.nan,
    )

    feature_df["best_yes_bid"] = best_yes_bid
    feature_df["best_no_bid"] = best_no_bid
    feature_df["best_yes_ask"] = best_yes_ask
    feature_df["best_no_ask"] = best_no_ask
    feature_df["midprice"] = midprice
    feature_df["spread"] = spread

    feature_df["yes_depth_top1"] = _top_k_depth_vectorized(yes_arr, k=1)
    feature_df["no_depth_top1"] = _top_k_depth_vectorized(no_arr, k=1)

    feature_df["yes_total_depth"] = np.nan_to_num(yes_arr, nan=0.0).sum(axis=1)
    feature_df["no_total_depth"] = np.nan_to_num(no_arr, nan=0.0).sum(axis=1)
    feature_df["total_depth"] = feature_df["yes_total_depth"] + feature_df["no_total_depth"]
    feature_df["yes_depth_share"] = feature_df["yes_total_depth"] / feature_df["total_depth"].replace(0, np.nan)

    if include_imbalance:
        feature_df["imbalance_top1"] = _safe_imbalance_series(
            feature_df["yes_depth_top1"], feature_df["no_depth_top1"]
        )
        feature_df["imbalance_total"] = _safe_imbalance_series(
            feature_df["yes_total_depth"], feature_df["no_total_depth"]
        )

    if add_recent_dynamics:
        lag_base_cols = [
            "midprice",
            "spread",
            "total_depth",
        ]
        if include_imbalance:
            lag_base_cols.append("imbalance_top1")

        for seconds in lag_seconds:
            steps = _seconds_to_steps(panel_interval=panel_interval, seconds=seconds)
            suffix = seconds_to_feature_suffix(seconds)

            for col in lag_base_cols:
                lag_col = f"{col}_lag_{suffix}"
                diff_col = f"{col}_change_{suffix}"
                feature_df[lag_col] = feature_df[col].shift(steps)
                feature_df[diff_col] = feature_df[col] - feature_df[lag_col]

        for seconds in rolling_window_seconds:
            steps = _seconds_to_steps(panel_interval=panel_interval, seconds=seconds)
            suffix = seconds_to_feature_suffix(seconds)

            feature_df[f"midprice_mean_{suffix}"] = (
                feature_df["midprice"].rolling(steps, min_periods=steps).mean()
            )
            feature_df[f"midprice_std_{suffix}"] = (
                feature_df["midprice"].rolling(steps, min_periods=steps).std()
            )
            feature_df[f"spread_mean_{suffix}"] = (
                feature_df["spread"].rolling(steps, min_periods=steps).mean()
            )

            if include_imbalance:
                feature_df[f"imbalance_top1_mean_{suffix}"] = (
                    feature_df["imbalance_top1"].rolling(steps, min_periods=steps).mean()
                )

    desired_front = [c for c in ["market_id", "ts"] if c in feature_df.columns]
    other_cols = [c for c in feature_df.columns if c not in desired_front]
    feature_df = feature_df[desired_front + other_cols]

    return feature_df


def build_features_from_parquet(
    panel_path: str | Path,
    panel_interval: str = DEFAULT_INTERVAL,
    include_imbalance: bool = True,
    add_recent_dynamics: bool = True,
    lag_seconds: tuple[float, ...] = (1.0,),
    rolling_window_seconds: tuple[float, ...] = (3.0,),
) -> pd.DataFrame:
    panel_path = Path(panel_path)
    panel_df = pd.read_parquet(panel_path)

    market_id = None
    if "market_id" in panel_df.columns:
        vals = panel_df["market_id"].dropna().unique()
        if len(vals) == 1:
            market_id = vals[0]

    if market_id is None:
        market_id = _derive_market_id_from_panel_name(
            panel_path,
            interval=panel_interval,
        )

    return build_features_from_panel(
        panel_df=panel_df,
        market_id=market_id,
        panel_interval=panel_interval,
        include_imbalance=include_imbalance,
        add_recent_dynamics=add_recent_dynamics,
        lag_seconds=lag_seconds,
        rolling_window_seconds=rolling_window_seconds,
    )


def save_feature_table(
    panel_path: str | Path,
    output_path: str | Path,
    panel_interval: str = DEFAULT_INTERVAL,
    include_imbalance: bool = True,
    add_recent_dynamics: bool = True,
    lag_seconds: tuple[float, ...] = (1.0,),
    rolling_window_seconds: tuple[float, ...] = (3.0,),
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_df = build_features_from_parquet(
        panel_path=panel_path,
        panel_interval=panel_interval,
        include_imbalance=include_imbalance,
        add_recent_dynamics=add_recent_dynamics,
        lag_seconds=lag_seconds,
        rolling_window_seconds=rolling_window_seconds,
    )

    feature_df.to_parquet(output_path, engine="pyarrow", compression="zstd")
    print(f"Saved features to {output_path}")
    print(feature_df.head())


def build_all_feature_tables(
    panel_dir: str | Path = TIME_PANEL_DIR,
    output_dir: str | Path = FEATURE_DIR,
    panel_interval: str = DEFAULT_INTERVAL,
    include_imbalance: bool = True,
    add_recent_dynamics: bool = True,
    lag_seconds: tuple[float, ...] = (1.0,),
    rolling_window_seconds: tuple[float, ...] = (3.0,),
    force: bool = False,
) -> None:
    panel_dir = Path(panel_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interval_suffix = interval_to_filename_suffix(panel_interval)
    pattern = f"*-base-panel-{interval_suffix}.parquet"
    panel_files = sorted(panel_dir.glob(pattern))

    if not panel_files:
        print(f"No base-panel parquet files found in {panel_dir} with pattern {pattern}")
        return

    print(f"Found {len(panel_files)} base-panel file(s).")
    print(f"Panel interval: {panel_interval}")
    print()

    processed = 0
    skipped = 0
    failed = 0

    for i, panel_path in enumerate(panel_files, start=1):
        stem = _derive_market_id_from_panel_name(panel_path, interval=panel_interval)
        output_path = output_dir / f"{stem}-features.parquet"

        if output_path.exists() and not force:
            print(f"[{i}/{len(panel_files)}] Skipping {stem} (already exists)")
            skipped += 1
            continue

        print(f"[{i}/{len(panel_files)}] Building features for {stem}...")
        start = time.time()

        try:
            save_feature_table(
                panel_path=panel_path,
                output_path=output_path,
                panel_interval=panel_interval,
                include_imbalance=include_imbalance,
                add_recent_dynamics=add_recent_dynamics,
                lag_seconds=lag_seconds,
                rolling_window_seconds=rolling_window_seconds,
            )
            elapsed = time.time() - start
            print(f"✅ Done {stem} ({elapsed:.2f}s)")
            processed += 1
        except Exception as exc:
            print(f"❌ Failed {stem}: {exc}")
            failed += 1

    print("\n===== SUMMARY =====")
    print(f"Panel interval: {panel_interval}")
    print(f"Processed:      {processed}")
    print(f"Skipped:        {skipped}")
    print(f"Failed:         {failed}")
    print("===================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-dir", default=str(TIME_PANEL_DIR))
    parser.add_argument("--output-dir", default=str(FEATURE_DIR))
    parser.add_argument("--panel-interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-imbalance", action="store_true")
    parser.add_argument("--no-recent-dynamics", action="store_true")
    parser.add_argument("--lag-seconds", default="1.0")
    parser.add_argument("--rolling-window-seconds", default="3.0")
    args = parser.parse_args()

    build_all_feature_tables(
        panel_dir=args.panel_dir,
        output_dir=args.output_dir,
        panel_interval=args.panel_interval,
        include_imbalance=not args.no_imbalance,
        add_recent_dynamics=not args.no_recent_dynamics,
        lag_seconds=parse_float_tuple(args.lag_seconds),
        rolling_window_seconds=parse_float_tuple(args.rolling_window_seconds),
        force=args.force,
    )
