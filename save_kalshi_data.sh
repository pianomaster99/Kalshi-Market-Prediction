#!/bin/bash
set -euo pipefail

DATA_TMP="/tmp/$USER/kalshi-data"
DATA_AFS="$HOME/Kalshi-Market-Prediction/data"

mkdir -p "$DATA_AFS"

echo "Syncing data from $DATA_TMP to $DATA_AFS ..."
rsync -ah --info=progress2 --partial --ignore-existing "$DATA_TMP/" "$DATA_AFS/"

echo "Deleting files from tmp only if they now exist in AFS..."
for f in "$DATA_TMP"/*; do
  [ -e "$f" ] || continue
  base=$(basename "$f")
  if [ -f "$DATA_AFS/$base" ]; then
    rm -f "$f"
  fi
done

find "$DATA_TMP" -mindepth 1 -type d -empty -delete

echo "Done."
