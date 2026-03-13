#!/bin/bash
set -euo pipefail

DATA_TMP="/tmp/$USER/kalshi-data"
DATA_AFS="$HOME/Kalshi-Market-Prediction/data"

mkdir -p "$DATA_AFS"

echo "Copying data from $DATA_TMP to $DATA_AFS ..."
rsync -av --remove-source-files "$DATA_TMP/" "$DATA_AFS/"

find "$DATA_TMP" -mindepth 1 -type d -empty -delete

echo "Done."
