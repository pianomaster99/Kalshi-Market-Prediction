from pathlib import Path
import pandas as pd


def build_time_based_orderbook_panel(
    orderbook_df: pd.DataFrame,
    market_id: str | None = None,
    interval: str = "30s",
) -> pd.DataFrame:
    """
    Convert event-driven orderbook rows into a fixed-time panel.

    Parameters
    ----------
    orderbook_df : pd.DataFrame
        Must contain:
          - 'ts'
          - yes_01 ... yes_99
          - no_01 ... no_99
    market_id : str | None
        Optional market identifier to attach as a column.
    interval : str
        Pandas frequency string, e.g. '30s', '1min'.

    Returns
    -------
    pd.DataFrame
        Fixed-time sampled orderbook panel with columns:
          - market_id (if provided)
          - ts
          - yes_01 ... yes_99
          - no_01 ... no_99
    """

    if "ts" not in orderbook_df.columns:
        raise ValueError("orderbook_df must contain a 'ts' column.")

    df = orderbook_df.copy()

    # Ensure timestamp type
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    if df.empty:
        raise ValueError("orderbook_df has no valid timestamps after cleaning.")

    # Remove duplicate timestamps: keep the latest row at the same ts
    df = df.drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

    # Identify book columns
    yes_cols = sorted([c for c in df.columns if c.startswith("yes_")])
    no_cols = sorted([c for c in df.columns if c.startswith("no_")])
    book_cols = yes_cols + no_cols

    if not book_cols:
        raise ValueError("No orderbook columns found (expected yes_* / no_* columns).")

    # Build fixed time grid
    min_ts = df["ts"].min()
    max_ts = df["ts"].max()

    start_ts = min_ts.ceil(interval)
    end_ts = max_ts.floor(interval)

    # Edge case: if the market is shorter than one interval
    if start_ts > end_ts:
        start_ts = min_ts
        end_ts = max_ts

    grid = pd.DataFrame({
        "ts": pd.date_range(start=start_ts, end=end_ts, freq=interval, tz="UTC")
    })

    if grid.empty:
        raise ValueError("Generated time grid is empty. Check timestamps or interval.")

    # As-of merge:
    # for each grid timestamp, take the most recent known orderbook state at or before that time
    panel = pd.merge_asof(
        grid,
        df[["ts"] + book_cols],
        on="ts",
        direction="backward",
    )

    # Drop rows before the first known orderbook state
    panel = panel.dropna(subset=book_cols, how="all").reset_index(drop=True)

    # Optional market_id
    if market_id is not None:
        panel.insert(0, "market_id", market_id)

    return panel

def build_time_based_panel_from_parquet(
    orderbook_path: str | Path,
    interval: str = "30s",
    market_id: str | None = None,
) -> pd.DataFrame:
    """
    Read a cleaned orderbook parquet file and convert it to a fixed-time panel.
    """
    orderbook_path = Path(orderbook_path)
    df = pd.read_parquet(orderbook_path)

    if market_id is None:
        market_id = orderbook_path.name.replace("-orderbook.parquet", "")

    panel = build_time_based_orderbook_panel(
        orderbook_df=df,
        market_id=market_id,
        interval=interval,
    )
    return panel

def save_time_based_panel(
    orderbook_path: str | Path,
    output_path: str | Path,
    interval: str = "30s",
    market_id: str | None = None,
) -> None:
    """
    Build fixed-time panel from a cleaned orderbook parquet and save it.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panel = build_time_based_panel_from_parquet(
        orderbook_path=orderbook_path,
        interval=interval,
        market_id=market_id,
    )

    panel.to_parquet(output_path, engine="pyarrow", compression="zstd")
    print(f"Saved fixed-time panel to {output_path}")
    print(panel.head())


def sanity_check_single_timestamp(
    raw_orderbook_path: str | Path,
    panel_path: str | Path,
    check_ts: str,
    cols_to_check: list[str] | None = None,
) -> None:
    """
    Compare one fixed-time panel row against the most recent raw orderbook row
    at or before the given timestamp.

    Parameters
    ----------
    raw_orderbook_path : str | Path
        Path to original event-driven orderbook parquet.
    panel_path : str | Path
        Path to fixed-time base panel parquet.
    check_ts : str
        Timestamp to check, e.g. "2026-03-08 00:15:30+00:00"
    cols_to_check : list[str] | None
        Columns to compare. If None, use a default subset.
    """
    if cols_to_check is None:
        cols_to_check = ["yes_01", "yes_03", "yes_11", "no_01", "no_19"]

    raw_df = pd.read_parquet(raw_orderbook_path)
    panel_df = pd.read_parquet(panel_path)

    raw_df["ts"] = pd.to_datetime(raw_df["ts"], utc=True, errors="coerce")
    panel_df["ts"] = pd.to_datetime(panel_df["ts"], utc=True, errors="coerce")
    target_ts = pd.to_datetime(check_ts, utc=True)

    raw_df = raw_df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    panel_df = panel_df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    # 取 panel 中该时间点对应的行
    panel_row = panel_df.loc[panel_df["ts"] == target_ts]
    if panel_row.empty:
        raise ValueError(f"Timestamp {target_ts} not found in panel.")
    panel_row = panel_row.iloc[0]

    # 取 raw 中 <= target_ts 的最后一条
    raw_candidates = raw_df.loc[raw_df["ts"] <= target_ts]
    if raw_candidates.empty:
        raise ValueError(f"No raw orderbook row found at or before {target_ts}.")
    raw_row = raw_candidates.iloc[-1]

    print("=" * 80)
    print(f"Target panel ts: {target_ts}")
    print(f"Matched raw ts : {raw_row['ts']}")
    print("=" * 80)

    comparison_rows = []
    all_match = True

    for col in cols_to_check:
        panel_val = panel_row[col]
        raw_val = raw_row[col]
        is_match = panel_val == raw_val
        if not is_match:
            all_match = False

        comparison_rows.append({
            "column": col,
            "panel_value": panel_val,
            "raw_value": raw_val,
            "match": is_match,
        })

    comparison_df = pd.DataFrame(comparison_rows)
    print(comparison_df.to_string(index=False))

    print("\nResult:")
    if all_match:
        print("✅ All checked columns match.")
    else:
        print("❌ Some columns do NOT match.")

def build_all_time_panels(
    reconstructed_dir: str | Path = "data/reconstructed_data",
    output_dir: str | Path = "data/time_panel",
    interval: str = "30s",
    force: bool = False,
    ) -> None:
    """
    Read all *-orderbook.parquet files in reconstructed_dir
    and build fixed-time base panels into output_dir.
    """
    reconstructed_dir = Path(reconstructed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orderbook_files = sorted(reconstructed_dir.glob("*-orderbook.parquet"))

    if not orderbook_files:
        print(f"No orderbook parquet files found in {reconstructed_dir}")
        return

    print(f"Found {len(orderbook_files)} orderbook files.\n")

    processed = 0
    skipped = 0
    failed = 0

    for i, orderbook_path in enumerate(orderbook_files, start=1):
        stem = orderbook_path.name.replace("-orderbook.parquet", "")
        output_path = output_dir / f"{stem}-base-panel.parquet"

        if output_path.exists() and not force:
            print(f"[{i}/{len(orderbook_files)}] Skipping {stem} (already exists)")
            skipped += 1
            continue

        print(f"[{i}/{len(orderbook_files)}] Building panel for {stem}...")

        try:
            save_time_based_panel(
                orderbook_path=orderbook_path,
                output_path=output_path,
                interval=interval,
                market_id=stem,
            )
            processed += 1

        except Exception as e:
            print(f"❌ Failed {stem}: {e}")
            failed += 1

    print("\n===== SUMMARY =====")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print("===================")

if __name__ == "__main__":
    build_all_time_panels(
        reconstructed_dir="data/reconstructed_data",
        output_dir="data/time_panel",
        interval="30s",
        force=False,
    )