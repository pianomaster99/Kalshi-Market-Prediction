#!/usr/bin/env python3
"""
Plot Kalshi orderbook + trades from parquet files.

Expected inputs:
- orderbook parquet in wide format with:
    ts, yes_01..yes_99, no_01..no_99
  or dict format with:
    ts, yes_book, no_book

- trades parquet with:
    ts, yes_price_dollars, count, taker_side
  or:
    ts, yes_price, count, taker_side

What the plot shows:
- green dots: YES orderbook levels
- red dots: NO orderbook levels, inverted so that no price p is plotted at y = 1 - p
- black dots: trades at yes_price

Volume encoding:
- dot opacity (alpha) represents volume
- dot size is constant
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orderbook", required=True, help="Path to orderbook parquet")
    parser.add_argument("--trades", required=True, help="Path to trades parquet")
    parser.add_argument("--out", default="kalshi_plot.png", help="Output image path")
    parser.add_argument("--title", default="Kalshi Orderbook + Trades", help="Plot title")
    parser.add_argument(
        "--max-points",
        type=int,
        default=300_000,
        help="Downsample orderbook points if total exceeds this many points",
    )
    parser.add_argument(
        "--book-dot-size",
        type=float,
        default=5.0,
        help="Constant dot size for orderbook points",
    )
    parser.add_argument(
        "--trade-dot-size",
        type=float,
        default=10.0,
        help="Constant dot size for trade points",
    )
    parser.add_argument(
        "--book-min-alpha",
        type=float,
        default=0.02,
        help="Minimum alpha for orderbook points",
    )
    parser.add_argument(
        "--book-max-alpha",
        type=float,
        default=0.55,
        help="Maximum alpha for orderbook points",
    )
    parser.add_argument(
        "--trade-min-alpha",
        type=float,
        default=0.15,
        help="Minimum alpha for trade points",
    )
    parser.add_argument(
        "--trade-max-alpha",
        type=float,
        default=0.95,
        help="Maximum alpha for trade points",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output DPI",
    )
    return parser.parse_args()


def ensure_datetime_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def alpha_from_volume(
    volumes: np.ndarray,
    min_alpha: float = 0.05,
    max_alpha: float = 0.9,
) -> np.ndarray:
    """
    Convert volumes into per-point alpha values.
    Uses log scaling so large outliers do not dominate.
    """
    volumes = np.asarray(volumes, dtype=float)
    volumes = np.clip(volumes, 0, None)

    if len(volumes) == 0:
        return np.array([], dtype=float)

    nonzero = volumes[volumes > 0]
    if len(nonzero) == 0:
        return np.full(len(volumes), min_alpha, dtype=float)

    log_v = np.log1p(volumes)
    denom = np.percentile(np.log1p(nonzero), 95)
    if denom <= 0:
        denom = 1.0

    alpha = min_alpha + (max_alpha - min_alpha) * (log_v / denom)
    return np.clip(alpha, min_alpha, max_alpha)


def rgba_colors(color: str, alphas: np.ndarray) -> np.ndarray:
    """
    Build per-point RGBA colors for matplotlib scatter.
    """
    alphas = np.asarray(alphas, dtype=float)

    if color == "green":
        rgb = np.array([0.0, 0.5, 0.0])
    elif color == "red":
        rgb = np.array([1.0, 0.0, 0.0])
    elif color == "black":
        rgb = np.array([0.0, 0.0, 0.0])
    else:
        raise ValueError(f"Unsupported color: {color}")

    out = np.zeros((len(alphas), 4), dtype=float)
    out[:, :3] = rgb
    out[:, 3] = alphas
    return out


def _detect_wide_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    cleaned = []
    for c in cols:
        suffix = c[len(prefix):]
        if suffix.isdigit():
            cleaned.append(c)
    return sorted(cleaned, key=lambda c: int(c.split("_")[1]))


def _maybe_parse_book_obj(obj) -> dict[float, float]:
    """
    Fallback for parquet files that store yes_book / no_book as dict-like or JSON strings.
    """
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return {float(k): float(v) for k, v in obj.items()}

    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(s)
        if isinstance(parsed, dict):
            return {float(k): float(v) for k, v in parsed.items()}

    return {}


def orderbook_to_points(
    orderbook_df: pd.DataFrame,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      yes_x, yes_y, yes_v, no_x, no_y, no_v
    """
    df = orderbook_df.copy()
    df["ts"] = ensure_datetime_utc(df["ts"])
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    yes_cols = _detect_wide_columns(df, "yes_")
    no_cols = _detect_wide_columns(df, "no_")

    if yes_cols and no_cols:
        yes_prices = np.array([int(c.split("_")[1]) / 100.0 for c in yes_cols], dtype=float)
        no_prices = np.array([int(c.split("_")[1]) / 100.0 for c in no_cols], dtype=float)

        ts_values = df["ts"].to_numpy()
        yes_mat = df[yes_cols].fillna(0).to_numpy(dtype=float)
        no_mat = df[no_cols].fillna(0).to_numpy(dtype=float)

        yes_r, yes_c = np.nonzero(yes_mat > 0)
        no_r, no_c = np.nonzero(no_mat > 0)

        yes_x = ts_values[yes_r]
        yes_y = yes_prices[yes_c]
        yes_v = yes_mat[yes_r, yes_c]

        no_x = ts_values[no_r]
        no_y = 1.0 - no_prices[no_c]
        no_v = no_mat[no_r, no_c]
    else:
        if "yes_book" not in df.columns or "no_book" not in df.columns:
            raise ValueError(
                "Orderbook parquet must contain either wide columns "
                "(yes_01..yes_99 and no_01..no_99) or yes_book/no_book columns."
            )

        yes_x, yes_y, yes_v = [], [], []
        no_x, no_y, no_v = [], [], []

        for _, row in df.iterrows():
            ts = row["ts"]

            yes_book = _maybe_parse_book_obj(row["yes_book"])
            for p, v in yes_book.items():
                if v > 0:
                    yes_x.append(ts)
                    yes_y.append(float(p))
                    yes_v.append(float(v))

            no_book = _maybe_parse_book_obj(row["no_book"])
            for p, v in no_book.items():
                if v > 0:
                    no_x.append(ts)
                    no_y.append(1.0 - float(p))
                    no_v.append(float(v))

        yes_x = np.array(yes_x)
        yes_y = np.array(yes_y, dtype=float)
        yes_v = np.array(yes_v, dtype=float)

        no_x = np.array(no_x)
        no_y = np.array(no_y, dtype=float)
        no_v = np.array(no_v, dtype=float)

    total_points = len(yes_v) + len(no_v)
    if total_points > max_points and total_points > 0:
        frac = max_points / total_points
        rng = np.random.default_rng(0)

        yes_keep = max(1, int(len(yes_v) * frac)) if len(yes_v) else 0
        no_keep = max(1, int(len(no_v) * frac)) if len(no_v) else 0

        if len(yes_v) > yes_keep:
            idx = np.sort(rng.choice(len(yes_v), size=yes_keep, replace=False))
            yes_x, yes_y, yes_v = yes_x[idx], yes_y[idx], yes_v[idx]

        if len(no_v) > no_keep:
            idx = np.sort(rng.choice(len(no_v), size=no_keep, replace=False))
            no_x, no_y, no_v = no_x[idx], no_y[idx], no_v[idx]

    return yes_x, yes_y, yes_v, no_x, no_y, no_v


def trades_to_points(trades_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = trades_df.copy()
    df["ts"] = ensure_datetime_utc(df["ts"])
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    if "yes_price_dollars" in df.columns:
        trade_y = pd.to_numeric(df["yes_price_dollars"], errors="coerce").to_numpy(dtype=float)
    elif "yes_price" in df.columns:
        trade_y = pd.to_numeric(df["yes_price"], errors="coerce").to_numpy(dtype=float) / 100.0
    else:
        raise ValueError("Trades parquet must contain yes_price_dollars or yes_price.")

    trade_x = df["ts"].to_numpy()
    trade_v = pd.to_numeric(df["count"], errors="coerce").fillna(0).to_numpy(dtype=float)

    mask = np.isfinite(trade_y)
    return trade_x[mask], trade_y[mask], trade_v[mask]


def make_plot(
    orderbook_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    out_path: str | Path,
    title: str,
    max_points: int,
    book_dot_size: float,
    trade_dot_size: float,
    book_min_alpha: float,
    book_max_alpha: float,
    trade_min_alpha: float,
    trade_max_alpha: float,
    dpi: int,
) -> None:
    yes_x, yes_y, yes_v, no_x, no_y, no_v = orderbook_to_points(orderbook_df, max_points=max_points)
    trade_x, trade_y, trade_v = trades_to_points(trades_df)

    fig, ax = plt.subplots(figsize=(16, 8))

    if len(yes_v):
        yes_alpha = alpha_from_volume(yes_v, min_alpha=book_min_alpha, max_alpha=book_max_alpha)
        ax.scatter(
            yes_x,
            yes_y,
            s=book_dot_size,
            c=rgba_colors("green", yes_alpha),
            linewidths=0,
            label="YES book",
        )

    if len(no_v):
        no_alpha = alpha_from_volume(no_v, min_alpha=book_min_alpha, max_alpha=book_max_alpha)
        ax.scatter(
            no_x,
            no_y,
            s=book_dot_size,
            c=rgba_colors("red", no_alpha),
            linewidths=0,
            label="NO book (inverted)",
        )

    if len(trade_v):
        trade_alpha = alpha_from_volume(trade_v, min_alpha=trade_min_alpha, max_alpha=trade_max_alpha)
        ax.scatter(
            trade_x,
            trade_y,
            s=trade_dot_size,
            c=rgba_colors("black", trade_alpha),
            linewidths=0.2,
            label="Trades (YES price)",
        )

    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("YES-price scale (NO side shown as 1 - no_price)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")

    fig.autofmt_xdate()
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    orderbook_df = pd.read_parquet(args.orderbook)
    trades_df = pd.read_parquet(args.trades)

    make_plot(
        orderbook_df=orderbook_df,
        trades_df=trades_df,
        out_path=args.out,
        title=args.title,
        max_points=args.max_points,
        book_dot_size=args.book_dot_size,
        trade_dot_size=args.trade_dot_size,
        book_min_alpha=args.book_min_alpha,
        book_max_alpha=args.book_max_alpha,
        trade_min_alpha=args.trade_min_alpha,
        trade_max_alpha=args.trade_max_alpha,
        dpi=args.dpi,
    )

    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()