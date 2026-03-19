from pathlib import Path
import time
import argparse

import pandas as pd

from data_analysis.plot_kalshi_parquet import make_plot


DATA_DIR = Path("data/reconstructed_data")
OUTPUT_DIR = Path("data_analysis/visualization")


def generate_all_plots(force: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    orderbook_files = sorted(DATA_DIR.glob("*-orderbook.parquet"))

    if not orderbook_files:
        print("No orderbook parquet files found in reconstructed_data/.")
        return

    print(f"Found {len(orderbook_files)} markets to plot.\n")

    processed = 0
    skipped = 0
    failed = 0

    for i, orderbook_path in enumerate(orderbook_files, start=1):
        stem = orderbook_path.name.replace("-orderbook.parquet", "")

        trades_path = DATA_DIR / f"{stem}-trades.parquet"
        output_path = OUTPUT_DIR / f"{stem}-plot.png"

        # ✅ Skip if already exists
        if not force and output_path.exists():
            print(f"[{i}/{len(orderbook_files)}] Skipping {stem} (plot exists)")
            skipped += 1
            continue

        if not trades_path.exists():
            print(f"[{i}/{len(orderbook_files)}] Missing trades file for {stem}, skipping")
            skipped += 1
            continue

        print(f"[{i}/{len(orderbook_files)}] Plotting {stem}...")

        start = time.time()

        try:
            orderbook_df = pd.read_parquet(orderbook_path)
            trades_df = pd.read_parquet(trades_path)

            make_plot(
                orderbook_df=orderbook_df,
                trades_df=trades_df,
                out_path=output_path,
                title=stem,
                max_points=300_000,
                book_dot_size=4,
                trade_dot_size=9,
                book_min_alpha=0.01,
                book_max_alpha=0.35,
                trade_min_alpha=0.2,
                trade_max_alpha=0.95,
                dpi=160,
            )

            elapsed = time.time() - start

            print(f"✅ Done {stem} ({elapsed:.2f}s) -> {output_path.name}")
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
        help="Regenerate plots even if they already exist",
    )
    args = parser.parse_args()

    generate_all_plots(force=args.force)