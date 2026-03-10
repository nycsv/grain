"""Convert Lhotse Shar data (cuts + recording tars) to a Parquet metadata table.

Reads all cuts.*.jsonl.gz files from one or more Shar directories, flattens
the nested Lhotse cut structure into a flat table, and writes a single Parquet
file. Audio stays in the original tar files — only metadata is consolidated.

Usage:
  # Single directory
  python examples/lhotse_shar_to_parquet.py \
      --shar_dir /path/to/shar \
      --output_dir /path/to/output

  # Multiple directories
  python examples/lhotse_shar_to_parquet.py \
      --shar_dir /path/to/shar1 /path/to/shar2 /path/to/shar3 \
      --output_dir /path/to/output

Example with test data:
  python examples/lhotse_shar_to_parquet.py \
      --shar_dir examples/data/output/shar_test \
      --output_dir examples/data/output/parquet_from_shar
"""

import argparse
import gzip
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Parse Lhotse cuts
# ---------------------------------------------------------------------------

def read_cuts_jsonl_gz(path: str) -> list[dict]:
    """Read a gzipped JSONL cuts file and return list of dicts."""
    records = []
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def flatten_cut(cut: dict) -> dict:
    """Flatten a nested Lhotse cut dict into a flat row for Parquet.

    Lhotse cut structure:
      {
        "id": "...",
        "start": 0.0,
        "duration": 5.855,
        "channel": 0,
        "supervisions": [{"text": "...", "speaker": "...", ...}],
        "recording": {"sampling_rate": 16000, "num_samples": 93680, ...},
        "type": "MonoCut"
      }

    Flattened to:
      {
        "id": "...",
        "start": 0.0,
        "duration": 5.855,
        "channel": 0,
        "text": "...",
        "speaker": "...",
        "sampling_rate": 16000,
        "num_samples": 93680,
        "type": "MonoCut",
        "source_audio_path": "...",
      }
    """
    row = {
        "id": cut["id"],
        "start": cut.get("start", 0.0),
        "duration": cut["duration"],
        "channel": cut.get("channel", 0),
        "type": cut.get("type", ""),
    }

    # Flatten first supervision (most common case: one supervision per cut)
    sups = cut.get("supervisions", [])
    if sups:
        sup = sups[0]
        row["text"] = sup.get("text", "")
        row["speaker"] = sup.get("speaker", "")
        row["language"] = sup.get("language", "")
        row["gender"] = sup.get("gender", "")
        # Store full supervisions JSON if multiple
        if len(sups) > 1:
            row["supervisions_json"] = json.dumps(sups)
        else:
            row["supervisions_json"] = ""
    else:
        row["text"] = ""
        row["speaker"] = ""
        row["language"] = ""
        row["gender"] = ""
        row["supervisions_json"] = ""

    # Flatten recording metadata
    rec = cut.get("recording", {})
    row["sampling_rate"] = rec.get("sampling_rate", 0)
    row["num_samples"] = rec.get("num_samples", 0)

    # Extract original audio source path (if available)
    sources = rec.get("sources", [])
    if sources:
        source_type = sources[0].get("type", "")
        source_path = sources[0].get("source", "")
        row["source_type"] = source_type
        row["source_audio_path"] = source_path
    else:
        row["source_type"] = ""
        row["source_audio_path"] = ""

    return row


# ---------------------------------------------------------------------------
# Discover Shar files
# ---------------------------------------------------------------------------

def find_cuts_files(shar_dirs: list[str]) -> list[str]:
    """Find all cuts JSONL files across multiple Shar directories.

    Returns:
      cuts_paths — sorted by (directory, shard number).
    """
    cuts_paths = []

    for shar_dir in shar_dirs:
        shar_path = Path(shar_dir)
        if not shar_path.is_dir():
            print(f"  WARNING: {shar_dir} is not a directory, skipping")
            continue
        cuts_paths.extend(
            sorted(str(p) for p in shar_path.glob("cuts.*.jsonl*"))
        )

    return cuts_paths


# ---------------------------------------------------------------------------
# Build Parquet table
# ---------------------------------------------------------------------------

def build_parquet(cuts_paths: list[str], output_path: str) -> int:
    """Read all cuts shards and write a single Parquet file.

    Returns the total number of rows written.
    """
    all_rows: list[dict] = []

    for i, cuts_path in enumerate(cuts_paths):
        cuts = read_cuts_jsonl_gz(cuts_path)
        shard_name = os.path.basename(cuts_path)
        source_dir = os.path.dirname(os.path.abspath(cuts_path))
        for cut in cuts:
            row = flatten_cut(cut)
            row["shard"] = shard_name
            row["source_dir"] = source_dir
            all_rows.append(row)
        print(f"  Read {source_dir}/{shard_name}: {len(cuts)} cuts")

    if not all_rows:
        print("  WARNING: No cuts found!")
        return 0

    # Build PyArrow table from flat dicts
    columns: dict[str, list] = {key: [] for key in all_rows[0].keys()}
    for row in all_rows:
        for key in columns:
            columns[key].append(row.get(key, ""))

    table = pa.table(columns)
    pq.write_table(table, output_path)
    print(f"\n  Wrote Parquet: {output_path}")
    print(f"  Total rows: {len(all_rows)}")
    print(f"  Columns: {table.column_names}")
    return len(all_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert Lhotse Shar data to Parquet + optional ArrayRecord."
    )
    parser.add_argument(
        "--shar_dir",
        nargs="+",
        required=True,
        help="One or more paths to Lhotse Shar directories containing cuts.*.jsonl.gz",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for output Parquet file.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover files
    print(f"Scanning {len(args.shar_dir)} Shar director{'y' if len(args.shar_dir) == 1 else 'ies'}:")
    for d in args.shar_dir:
        print(f"  - {d}")
    cuts_paths = find_cuts_files(args.shar_dir)
    print(f"  Found {len(cuts_paths)} cuts shards total")

    if not cuts_paths:
        print("ERROR: No cuts.*.jsonl* files found in the Shar directories.")
        return

    # Build Parquet
    parquet_path = os.path.join(args.output_dir, "metadata.parquet")
    print(f"\nBuilding Parquet metadata table...")
    num_rows = build_parquet(cuts_paths, parquet_path)

    # Summary
    size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"  Done! {parquet_path} ({size_mb:.2f} MB)")
    print(f"{'='*60}")

    # Quick verification
    table = pq.read_table(parquet_path)
    print(f"  Rows: {table.num_rows}")
    print(f"  Columns: {table.column_names}")
    print(f"  Sample row:")
    for col in table.column_names:
        val = table.column(col)[0].as_py()
        if isinstance(val, str) and len(val) > 60:
            val = val[:60] + "..."
        print(f"    {col}: {val}")


if __name__ == "__main__":
    main()
