#!/bin/bash
# Run LibriSpeech Grain test for each data format.
# Usage: bash examples/run_test.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/test_librispeech_grain.py"

# Run from /tmp to avoid local source tree shadowing the installed grain package.
cd /tmp

for source in jsonl parquet tfrecord arrayrecord; do
  echo "=========================================="
  echo "  Testing: $source"
  echo "=========================================="
  python "$TEST_SCRIPT" --source "$source"
  echo
done
