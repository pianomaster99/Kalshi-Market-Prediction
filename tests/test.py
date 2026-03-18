import pandas as pd

orderbook = pd.read_parquet("reconstructed_data/KXNBAGAME-26MAR07GSWOKC-GSW-orderbook.parquet")
print(orderbook)