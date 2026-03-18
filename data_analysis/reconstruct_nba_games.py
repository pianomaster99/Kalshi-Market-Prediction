from pathlib import Path
import time
import argparse

import pandas as pd

from data_analysis.reconstruct import parse_kalshi_file_fast_wide


DATA_DIR = Path("data/raw_data")
OUTPUT_DIR = Path("data/reconstructed_data")


def reconstruct_all_nba_games(force: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(DATA_DIR.glob("KXNBAGAME-*.ndjson.gz"))

    if not files:
        print("No NBA game files found in data/.")
        return

    print(f"Found {len(files)} NBA game file(s).\n")

    processed = 0
    skipped = 0
    failed = 0

    for i, file_path in enumerate(files, start=1):
        stem = file_path.name.removesuffix(".ndjson.gz")

        orderbook_out = OUTPUT_DIR / f"{stem}-orderbook.parquet"
        trades_out = OUTPUT_DIR / f"{stem}-trades.parquet"

        # ✅ Skip logic
        if not force and orderbook_out.exists() and trades_out.exists():
            print(f"[{i}/{len(files)}] Skipping {stem} (already exists)")
            skipped += 1
            continue

        print(f"[{i}/{len(files)}] Processing {stem}...")

        start = time.time()

        try:
            orderbook_df, trades_df = parse_kalshi_file_fast_wide(file_path)

            orderbook_df.to_parquet(orderbook_out, engine="pyarrow", compression="zstd")
            trades_df.to_parquet(trades_out, engine="pyarrow", compression="zstd")

            elapsed = time.time() - start

            print(
                f"✅ Done {stem} "
                f"({elapsed:.2f}s) -> "
                f"{orderbook_out.name}, {trades_out.name}"
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconstruct all files even if parquet already exists",
    )
    args = parser.parse_args()

    reconstruct_all_nba_games(force=args.force)