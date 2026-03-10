#!/bin/bash
# Convert Lhotse Shar cuts to a single Parquet metadata table.
# Audio stays in the original tar files.
#
# Usage:
#   bash examples/lhotse/run_lhotse_to_parquet.sh /path/to/shar
#   bash examples/lhotse/run_lhotse_to_parquet.sh /path/to/shar1 /path/to/shar2 /path/to/shar3
#   OUTPUT_DIR=/path/to/output bash examples/lhotse/run_lhotse_to_parquet.sh /path/to/shar1 /path/to/shar2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/data/output/parquet_from_shar}"

if [ $# -eq 0 ]; then
    echo "Usage: bash $0 <shar_dir1> [shar_dir2] ..."
    echo ""
    echo "  OUTPUT_DIR  Output directory (default: examples/data/output/parquet_from_shar)"
    exit 1
fi

for dir in "$@"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: $dir is not a directory"
        exit 1
    fi
done

python "${SCRIPT_DIR}/lhotse_shar_to_parquet.py" \
    --shar_dir "$@" \
    --output_dir "${OUTPUT_DIR}"
