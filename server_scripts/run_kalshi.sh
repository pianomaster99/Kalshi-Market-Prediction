#!/bin/bash
set -euo pipefail

REPO_AFS="$HOME/Kalshi-Market-Prediction"
REPO_TMP="/tmp/$USER/Kalshi-Market-Prediction-local"
VENV_TMP="/tmp/$USER/kalshi-venv"

DATA_TMP="/tmp/$USER/kalshi-data/raw_data"
DATA_AFS="$REPO_AFS/data/raw_data"

mkdir -p "/tmp/$USER"
mkdir -p "$DATA_TMP"
mkdir -p "$DATA_AFS"

echo "[1/6] Syncing repo to /tmp..."
rsync -a --delete \
  --exclude '.git' \
  --exclude 'data' \
  --exclude '__pycache__' \
  --exclude '.venv' \
  "$REPO_AFS/" "$REPO_TMP/"

echo "[2/6] Creating venv if needed..."
if [ ! -x "$VENV_TMP/bin/python3" ]; then
  python3 -m venv "$VENV_TMP"
fi

echo "[3/6] Activating venv..."
source "$VENV_TMP/bin/activate"

echo "[4/6] Installing requirements..."
pip install -r "$REPO_TMP/requirements.txt"

export KALSHI_OUTPUT_DIR="$DATA_TMP"

echo "[5/6] Launching program..."
cd "$REPO_TMP"
python3 -m data_collection.src.main

echo "[6/6] Syncing data back to AFS..."
rsync -a "$DATA_TMP/" "$DATA_AFS/"