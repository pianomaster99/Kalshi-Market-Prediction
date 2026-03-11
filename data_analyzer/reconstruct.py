import pandas as pd
import gzip
import json
from datetime import datetime
from dataclases import dataclass
from typing import Tuple

@dataclass
class OrderBookSnapshot:


def reconstruct(gz_filename):
    orderbook = []
    trades = []
    orderbook_columns = ["datetime", "snapshot"]
    trades_columns = ["datetime", "yes_price_dollars", "count", "taker_side"]
    with gzip.open(gz_filename, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            msg = data.get['msg']
            data_type = data["type"]
            if data_type == "trade":
                row = [
                    datetime.datetime.fromtimestamp(msg["ts"]), 
                    msg["yes_price_dollars"], 
                    msg["count"],
                    msg["taker_side"]
                    ]
                trades.append(row)
            elif data_type == "orderbook_delta":

        return df(orderbook, columns=orderbook_columns), df(trades, columns=trades_columns)


reconstruct("data/KXNBAGAME-26MAR07GSWOKC-GSW.ndjson.gz")