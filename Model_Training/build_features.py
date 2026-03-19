from pathlib import Path
import time
import argparse

import numpy as np
import pandas as pd


TIME_PANEL_DIR = Path("data/time_panel")
FEATURE_DIR = Path("data/features")


def _extract_price_levels(columns: list[str], prefix: str) -> list[int]:
    """
    Extract integer price levels from columns like yes_01, yes_02, ..., yes_99.
    """
    levels = []
    for col in columns:
        if col.startswith(prefix):
            try:
                levels.append(int(col.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return sorted(levels)


def _best_bid_from_row(row: pd.Series, side: str, price_levels: list[int]) -> float:
    """
    Find the best bid on one side of the book.

    For Kalshi orderbook:
    - best_yes_bid = highest YES price level with positive visible quantity
    - best_no_bid  = highest NO price level with positive visible quantity

    Returns price in dollars, e.g. 0.57
    If no visible depth exists, returns NaN.
    """
    for level in reversed(price_levels):  # highest price first
        qty = row[f"{side}_{level:02d}"]
        if pd.notna(qty) and qty > 0:
            return level / 100.0
    return np.nan


def _top_k_depth_from_row(row: pd.Series, side: str, price_levels: list[int], k: int) -> float:
    """
    Sum visible depth at the top-k best price levels on one side.

    Example:
    - yes_depth_top3 = sum of quantities at the 3 best YES bid levels
    - no_depth_top3  = sum of quantities at the 3 best NO bid levels
    """
    total = 0.0
    count = 0

    for level in reversed(price_levels):  # highest / most competitive levels first
        qty = row[f"{side}_{level:02d}"]
        if pd.notna(qty) and qty > 0:
            total += qty
            count += 1
            if count == k:
                break

    return total


def _safe_imbalance(a: float, b: float) -> float:
    """
    Normalized imbalance:
        (a - b) / (a + b)

    Interpretation:
    - positive -> YES side stronger
    - negative -> NO side stronger
    - near 0   -> both sides similar
    """
    denom = a + b
    if pd.isna(a) or pd.isna(b) or denom == 0:
        return np.nan
    return (a - b) / denom


def build_features_from_panel(
    panel_df: pd.DataFrame,
    market_id: str | None = None,
    include_imbalance: bool = True,
) -> pd.DataFrame:
    """
    Convert a fixed-time base panel into a feature table.

    Input:
    - market_id
    - ts
    - yes_01 ... yes_99
    - no_01 ... no_99

    Output:
    - market_id
    - ts
    - engineered market microstructure features
    """
    df = panel_df.copy()

    if "ts" not in df.columns:
        raise ValueError("panel_df must contain a 'ts' column.")

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    # If market_id is already in the panel, use it unless user explicitly provided one
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

    rows = []

    for _, row in df.iterrows():
        feat = {}

        # Keep identifiers
        feat["ts"] = row["ts"]
        if market_id is not None:
            feat["market_id"] = market_id

        # ------------------------------------------------------------------
        # Best visible bid prices
        # ------------------------------------------------------------------
        # Highest YES bid currently visible in the book
        best_yes_bid = _best_bid_from_row(row, "yes", yes_levels)

        # Highest NO bid currently visible in the book
        best_no_bid = _best_bid_from_row(row, "no", no_levels)

        # ------------------------------------------------------------------
        # Implied ask prices using YES + NO = 1 relationship
        # ------------------------------------------------------------------
        # If best NO bid is 0.42, then implied YES ask is 1 - 0.42 = 0.58
        best_yes_ask = 1.0 - best_no_bid if pd.notna(best_no_bid) else np.nan

        # If best YES bid is 0.57, then implied NO ask is 1 - 0.57 = 0.43
        best_no_ask = 1.0 - best_yes_bid if pd.notna(best_yes_bid) else np.nan

        # ------------------------------------------------------------------
        # Midprice and spread
        # ------------------------------------------------------------------
        # Midprice = center of best bid / ask on YES side
        if pd.notna(best_yes_bid) and pd.notna(best_yes_ask):
            midprice = (best_yes_bid + best_yes_ask) / 2.0
            spread = best_yes_ask - best_yes_bid
        else:
            midprice = np.nan
            spread = np.nan

        feat["best_yes_bid"] = best_yes_bid     # 当前 YES 最优买价
        feat["best_no_bid"] = best_no_bid       # 当前 NO 最优买价
        feat["best_yes_ask"] = best_yes_ask     # 由 NO 最优买价推出来的 YES 最优卖价
        feat["best_no_ask"] = best_no_ask       # 由 YES 最优买价推出来的 NO 最优卖价
        feat["midprice"] = midprice             # YES 盘口中间价
        feat["spread"] = spread                 # YES 最优卖价 - YES 最优买价

        # ------------------------------------------------------------------
        # Depth features: top 1 / top 3 / top 5 visible depth
        # ------------------------------------------------------------------
        yes_depth_top1 = _top_k_depth_from_row(row, "yes", yes_levels, k=1)
        yes_depth_top3 = _top_k_depth_from_row(row, "yes", yes_levels, k=3)
        yes_depth_top5 = _top_k_depth_from_row(row, "yes", yes_levels, k=5)

        no_depth_top1 = _top_k_depth_from_row(row, "no", no_levels, k=1)
        no_depth_top3 = _top_k_depth_from_row(row, "no", no_levels, k=3)
        no_depth_top5 = _top_k_depth_from_row(row, "no", no_levels, k=5)

        feat["yes_depth_top1"] = yes_depth_top1   # YES 最优一档挂单量
        feat["yes_depth_top3"] = yes_depth_top3   # YES 最优三档累计挂单量
        feat["yes_depth_top5"] = yes_depth_top5   # YES 最优五档累计挂单量

        feat["no_depth_top1"] = no_depth_top1     # NO 最优一档挂单量
        feat["no_depth_top3"] = no_depth_top3     # NO 最优三档累计挂单量
        feat["no_depth_top5"] = no_depth_top5     # NO 最优五档累计挂单量

        # ------------------------------------------------------------------
        # Total visible depth
        # ------------------------------------------------------------------
        yes_total_depth = row[yes_cols].fillna(0).sum()
        no_total_depth = row[no_cols].fillna(0).sum()
        total_depth = yes_total_depth + no_total_depth

        feat["yes_total_depth"] = yes_total_depth   # YES 全盘口可见总深度
        feat["no_total_depth"] = no_total_depth     # NO 全盘口可见总深度
        feat["total_depth"] = total_depth           # YES + NO 全部可见总深度

        # ------------------------------------------------------------------
        # Optional imbalance features
        # ------------------------------------------------------------------
        if include_imbalance:
            feat["imbalance_top1"] = _safe_imbalance(yes_depth_top1, no_depth_top1)
            feat["imbalance_top3"] = _safe_imbalance(yes_depth_top3, no_depth_top3)
            feat["imbalance_top5"] = _safe_imbalance(yes_depth_top5, no_depth_top5)

        rows.append(feat)

    feature_df = pd.DataFrame(rows)

    # Put market_id first if it exists
    desired_front = [c for c in ["market_id", "ts"] if c in feature_df.columns]
    other_cols = [c for c in feature_df.columns if c not in desired_front]
    feature_df = feature_df[desired_front + other_cols]

    return feature_df


def build_features_from_parquet(
    panel_path: str | Path,
    include_imbalance: bool = True,
) -> pd.DataFrame:
    """
    Read one base-panel parquet and convert it into a feature table.
    """
    panel_path = Path(panel_path)
    panel_df = pd.read_parquet(panel_path)

    market_id = None
    if "market_id" in panel_df.columns:
        vals = panel_df["market_id"].dropna().unique()
        if len(vals) == 1:
            market_id = vals[0]
    if market_id is None:
        market_id = panel_path.name.replace("-base-panel.parquet", "")

    return build_features_from_panel(
        panel_df=panel_df,
        market_id=market_id,
        include_imbalance=include_imbalance,
    )


def save_feature_table(
    panel_path: str | Path,
    output_path: str | Path,
    include_imbalance: bool = True,
) -> None:
    """
    Build features from one base panel and save to parquet.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_df = build_features_from_parquet(
        panel_path=panel_path,
        include_imbalance=include_imbalance,
    )

    feature_df.to_parquet(output_path, engine="pyarrow", compression="zstd")
    print(f"Saved features to {output_path}")
    print(feature_df.head())


def build_all_feature_tables(
    panel_dir: str | Path = TIME_PANEL_DIR,
    output_dir: str | Path = FEATURE_DIR,
    include_imbalance: bool = True,
    force: bool = False,
) -> None:
    """
    Batch process all *-base-panel.parquet files into *-features.parquet.
    """
    panel_dir = Path(panel_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_files = sorted(panel_dir.glob("*-base-panel.parquet"))

    if not panel_files:
        print(f"No base-panel parquet files found in {panel_dir}")
        return

    print(f"Found {len(panel_files)} base-panel file(s).\n")

    processed = 0
    skipped = 0
    failed = 0

    for i, panel_path in enumerate(panel_files, start=1):
        stem = panel_path.name.replace("-base-panel.parquet", "")
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
                include_imbalance=include_imbalance,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild all feature files even if they already exist",
    )
    parser.add_argument(
        "--no-imbalance",
        action="store_true",
        help="Do not include imbalance features",
    )
    args = parser.parse_args()

    build_all_feature_tables(
        panel_dir=TIME_PANEL_DIR,
        output_dir=FEATURE_DIR,
        include_imbalance=not args.no_imbalance,
        force=args.force,
    )