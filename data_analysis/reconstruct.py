import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _price_str_to_index(price_str: str) -> int:
    return int(round(float(price_str) * 100))


def parse_kalshi_file_fast_wide(path, max_price=99):
    path = Path(path)

    yes_book = np.zeros(max_price + 1, dtype=np.int32)
    no_book = np.zeros(max_price + 1, dtype=np.int32)

    orderbook_rows = []
    trade_rows = []

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Bad JSON on file line {line_num}: {line[:200]!r}"
                ) from e

            typ = obj.get("type")
            msg = obj.get("msg", {})

            if typ == "orderbook_snapshot":
                # Important: an empty snapshot means the book is truly empty,
                # so reset both sides fully.
                yes_book.fill(0)
                no_book.fill(0)

                yes_levels = msg.get("yes_dollars") or []
                no_levels = msg.get("no_dollars") or []

                for price_str, qty in yes_levels:
                    idx = _price_str_to_index(price_str)
                    if 1 <= idx <= max_price:
                        yes_book[idx] = int(qty)

                for price_str, qty in no_levels:
                    idx = _price_str_to_index(price_str)
                    if 1 <= idx <= max_price:
                        no_book[idx] = int(qty)

                ts_raw = msg.get("ts")
                ts = pd.to_datetime(ts_raw, utc=True) if ts_raw else pd.NaT

                row = {"ts": ts}
                row.update({f"yes_{i:02d}": int(yes_book[i]) for i in range(1, max_price + 1)})
                row.update({f"no_{i:02d}": int(no_book[i]) for i in range(1, max_price + 1)})
                orderbook_rows.append(row)

            elif typ == "orderbook_delta":
                price_str = msg.get("price_dollars")
                delta = msg.get("delta")
                side = msg.get("side")
                ts_raw = msg.get("ts")

                if price_str is None or delta is None or side not in {"yes", "no"}:
                    continue

                idx = _price_str_to_index(price_str)
                if not (1 <= idx <= max_price):
                    continue

                delta = int(delta)

                if side == "yes":
                    yes_book[idx] = max(0, yes_book[idx] + delta)
                else:
                    no_book[idx] = max(0, no_book[idx] + delta)

                ts = pd.to_datetime(ts_raw, utc=True) if ts_raw else pd.NaT

                row = {"ts": ts}
                row.update({f"yes_{i:02d}": int(yes_book[i]) for i in range(1, max_price + 1)})
                row.update({f"no_{i:02d}": int(no_book[i]) for i in range(1, max_price + 1)})
                orderbook_rows.append(row)

            elif typ == "trade":
                ts_raw = msg.get("ts")
                if ts_raw is None:
                    continue

                trade_rows.append({
                    "ts": pd.to_datetime(ts_raw, unit="s", utc=True),
                    "yes_price_dollars": msg.get("yes_price_dollars"),
                    "count": int(msg["count"]) if msg.get("count") is not None else None,
                    "taker_side": msg.get("taker_side"),
                })

    orderbook_df = pd.DataFrame(orderbook_rows)
    trades_df = pd.DataFrame(trade_rows)
    return orderbook_df, trades_df


if __name__ == "__main__":
    input_path = Path("data/KXNBAGAME-26MAR07GSWOKC-GSW.ndjson.gz")
    output_dir = Path("reconstructed_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.name.removesuffix(".ndjson.gz")

    orderbook_df, trades_df = parse_kalshi_file_fast_wide(input_path)

    orderbook_out = output_dir / f"{stem}-orderbook.parquet"
    trades_out = output_dir / f"{stem}-trades.parquet"

    orderbook_df.to_parquet(orderbook_out, engine="pyarrow", compression="zstd")
    trades_df.to_parquet(trades_out, engine="pyarrow", compression="zstd")

    print(f"Saved {orderbook_out}")
    print(f"Saved {trades_out}")
    print(orderbook_df.head())
    print(trades_df.head())