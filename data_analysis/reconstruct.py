import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _price_str_to_index(price_str: str) -> int:
    return int(round(float(price_str) * 100))


def _price_any_to_index(msg) -> int | None:
    """
    Accept either:
      - price_dollars: "0.8200"
      - price: 82
    Returns integer index like 82, or None if unavailable.
    """
    if msg.get("price_dollars") is not None:
        return _price_str_to_index(str(msg["price_dollars"]))

    if msg.get("price") is not None:
        try:
            return int(msg["price"])
        except (TypeError, ValueError):
            return None

    return None


def _fill_book_from_snapshot(msg, side: str, book: np.ndarray, max_price: int) -> None:
    """
    Fill one side of the book from snapshot data.

    Supports either:
      - yes_dollars / no_dollars: [["0.8200", qty], ...]
      - yes / no: [[82, qty], ...]

    If neither exists, that simply means this side is empty.
    """
    levels_dollars = msg.get(f"{side}_dollars")
    levels_int = msg.get(side)

    if levels_dollars:
        for price_str, qty in levels_dollars:
            idx = _price_str_to_index(str(price_str))
            if 1 <= idx <= max_price:
                book[idx] = int(qty)
        return

    if levels_int:
        for price, qty in levels_int:
            try:
                idx = int(price)
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= max_price:
                book[idx] = int(qty)


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
                # Empty snapshot is valid: it means the entire book is empty.
                yes_book.fill(0)
                no_book.fill(0)

                _fill_book_from_snapshot(msg, "yes", yes_book, max_price)
                _fill_book_from_snapshot(msg, "no", no_book, max_price)

                ts_raw = msg.get("ts")
                ts = pd.to_datetime(ts_raw, utc=True) if ts_raw else pd.NaT

                row = [ts]
                row.extend(yes_book[1:max_price + 1].tolist())
                row.extend(no_book[1:max_price + 1].tolist())
                orderbook_rows.append(row)

            elif typ == "orderbook_delta":
                idx = _price_any_to_index(msg)
                delta = msg.get("delta")
                side = msg.get("side")
                ts_raw = msg.get("ts")

                if idx is None or delta is None or side not in {"yes", "no"}:
                    continue

                if not (1 <= idx <= max_price):
                    continue

                delta = int(delta)

                if side == "yes":
                    yes_book[idx] = max(0, yes_book[idx] + delta)
                else:
                    no_book[idx] = max(0, no_book[idx] + delta)

                ts = pd.to_datetime(ts_raw, utc=True) if ts_raw else pd.NaT

                row = [ts]
                row.extend(yes_book[1:max_price + 1].tolist())
                row.extend(no_book[1:max_price + 1].tolist())
                orderbook_rows.append(row)

            elif typ == "trade":
                ts_raw = msg.get("ts")
                if ts_raw is None:
                    continue

                yes_price_dollars = msg.get("yes_price_dollars")
                if yes_price_dollars is None and msg.get("yes_price") is not None:
                    yes_price_dollars = f"{int(msg['yes_price']) / 100:.4f}"

                trade_rows.append([
                    pd.to_datetime(ts_raw, unit="s", utc=True),
                    yes_price_dollars,
                    int(msg["count"]) if msg.get("count") is not None else None,
                    msg.get("taker_side"),
                ])

    orderbook_columns = (
        ["ts"]
        + [f"yes_{i:02d}" for i in range(1, max_price + 1)]
        + [f"no_{i:02d}" for i in range(1, max_price + 1)]
    )
    trade_columns = ["ts", "yes_price_dollars", "count", "taker_side"]

    orderbook_df = pd.DataFrame(orderbook_rows, columns=orderbook_columns)
    trades_df = pd.DataFrame(trade_rows, columns=trade_columns)

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