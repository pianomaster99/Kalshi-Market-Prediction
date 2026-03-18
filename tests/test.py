import pandas as pd

files = [
    "reconstructed_data/KXNBAGAME-26MAR11CLEORL-CLE-orderbook.parquet",
    "reconstructed_data/KXNBAGAME-26MAR12CHILAL-CHI-orderbook.parquet",
]

for path in files:
    df = pd.read_parquet(path)
    print(path)
    print(df.shape)
    print(df.head())
    print(df["ts"].notna().sum() if "ts" in df.columns else "no ts")
    print()