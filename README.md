# Kalshi Market Prediction

A full data pipeline for collecting, reconstructing, and analyzing Kalshi market data, with a focus on NBA game prediction markets.

Built by Henry Chen and Tom Liu.

---

## Overview

This project builds an end-to-end system for:

- Streaming live Kalshi market data via websockets
- Storing raw order book and trade events
- Reconstructing full order books over time
- Converting data into efficient parquet format
- Visualizing market behavior and trade activity

The goal is to better understand market microstructure and pricing dynamics in prediction markets.

---

## Features

- Real-time websocket data ingestion
- Compressed raw data storage (.ndjson.gz)
- Order book reconstruction from event streams
- Trade extraction and analysis
- Fast parquet-based data processing
- Visualization of order books and trades

---

## Repository Structure

```
Kalshi-Market-Prediction/
├── data/
│   ├── raw_data/                # Raw websocket logs (.ndjson.gz)
│   └── reconstructed_data/      # Processed parquet outputs
│
├── data_collection/
│   ├── kalshi_api/              # Websocket + API utilities
│   ├── src/                     # Main ingestion logic
│   └── file_handler.py          # File writing utilities
│
├── data_analysis/
│   ├── visualization/           # Plot helpers
│   ├── reconstruct.py           # Single-file reconstruction
│   ├── reconstruct_nba_games.py # Batch reconstruction
│   └── plot_kalshi_parquet.py   # Visualization script
│
├── server_scripts/              # Scripts for running on servers
├── tests/                       # Testing utilities
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/pianomaster99/Kalshi-Market-Prediction.git
cd Kalshi-Market-Prediction

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Data Collection

Start the websocket data collector:

```bash
python3 -m data_collection.src.main
```

This launches an interactive CLI for subscribing to market tickers and controlling data capture.

### Commands

| Command | Description |
|--------|-------------|
| write TICKER | Start writing data for ticker |
| delete TICKER | Stop writing data |
| subwrite T1 T2 ... | Subscribe + write group |
| unsubdelete T1 T2 ... | Unsubscribe + stop group |
| list | Show active subscriptions |
| help | Show commands |
| quit | Exit |

### Output

- Format: newline-delimited JSON
- Compression: gzip (.ndjson.gz)
- Location: data/raw_data/

---

## Reconstruction

Convert raw event logs into structured parquet files.

### Single file

```bash
python3 -m data_analysis.reconstruct
```

### Batch (NBA markets)

```bash
python3 -m data_analysis.reconstruct_nba_games
```

Force overwrite:

```bash
python3 -m data_analysis.reconstruct_nba_games --force
```

### Output

Each raw file produces:

- *-orderbook.parquet  
- *-trades.parquet  

Saved in:

```
data/reconstructed_data/
```

---

## Visualization

Generate plots from reconstructed data:

```bash
python3 -m data_analysis.plot_kalshi_parquet \
  --orderbook data/reconstructed_data/FILE-orderbook.parquet \
  --trades data/reconstructed_data/FILE-trades.parquet \
  --out output.png \
  --title "Market Visualization"
```

### Plot Features

- YES order book levels
- NO order book levels (inverted)
- Trade points over time
- Volume encoded as opacity

---

## Server Usage

Run long data collection jobs using:

```bash
bash server_scripts/run_kalshi.sh
```

This script:

- Syncs repo to /tmp
- Creates a virtual environment
- Installs dependencies
- Runs the collector
- Syncs data back to data/raw_data

---

## Testing

Run test scripts:

```bash
python3 -m tests.test_read_file
python3 -m tests.test_get_tickers
python3 -m tests.test_KalshiWSClient
```

---

## Workflow

1. Collect raw data → data/raw_data  
2. Reconstruct → data/reconstructed_data  
3. Analyze / visualize  

---

## Tech Stack

- Python
- WebSockets
- Pandas
- PyArrow (Parquet)
- Matplotlib

---

## Notes

- Raw data is stored as compressed event streams
- Reconstruction builds full order books from deltas
- Current focus is NBA prediction markets (KXNBAGAME-*)

---

## Authors

- Henry Chen  
- Tom Liu