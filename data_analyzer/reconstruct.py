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
                yes_book.fill(0)
                no_book.fill(0)

                for price_str, qty in msg.get("yes_dollars", []):
                    yes_book[_price_str_to_index(price_str)] = int(qty)

                for price_str, qty in msg.get("no_dollars", []):
                    no_book[_price_str_to_index(price_str)] = int(qty)

                ts = msg.get("ts")
                ts = pd.to_datetime(ts, utc=True) if ts else pd.NaT

                row = {"ts": ts}
                row.update({f"yes_{i:02d}": int(yes_book[i]) for i in range(1, max_price + 1)})
                row.update({f"no_{i:02d}": int(no_book[i]) for i in range(1, max_price + 1)})
                orderbook_rows.append(row)

            elif typ == "orderbook_delta":
                idx = _price_str_to_index(msg["price_dollars"])
                delta = int(msg["delta"])

                if msg["side"] == "yes":
                    yes_book[idx] = max(0, yes_book[idx] + delta)
                else:
                    no_book[idx] = max(0, no_book[idx] + delta)

                ts = pd.to_datetime(msg["ts"], utc=True)

                row = {"ts": ts}
                row.update({f"yes_{i:02d}": int(yes_book[i]) for i in range(1, max_price + 1)})
                row.update({f"no_{i:02d}": int(no_book[i]) for i in range(1, max_price + 1)})
                orderbook_rows.append(row)

            elif typ == "trade":
                trade_rows.append({
                    "ts": pd.to_datetime(msg["ts"], unit="s", utc=True),
                    "yes_price_dollars": msg.get("yes_price_dollars"),
                    "count": int(msg["count"]) if msg.get("count") is not None else None,
                    "taker_side": msg.get("taker_side"),
                })

    orderbook_df = pd.DataFrame(orderbook_rows)
    trades_df = pd.DataFrame(trade_rows)
    return orderbook_df, trades_df

if __name__ == "__main__":
    orderbook_df, trades_df = parse_kalshi_file_fast_wide(
        "data/KXNBAGAME-26MAR07GSWOKC-GSW.ndjson.gz"
    )

    orderbook_df.to_parquet("reconstructed_data/orderbook.parquet", compression="zstd")
    trades_df.to_parquet("reconstructed_data/trades.parquet", compression="zstd")

    print(orderbook_df.head())
    print(trades_df.head())