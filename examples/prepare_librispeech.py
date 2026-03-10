"""Prepare LibriSpeech dev-clean data in multiple formats for Grain.

Steps:
  1. Parse LibriSpeech transcripts and audio metadata -> NeMo-style JSONL
  2. Convert JSONL -> Parquet
  3. Convert JSONL -> TFRecord
  4. Convert JSONL -> ArrayRecord

Usage:
  python examples/prepare_librispeech.py \
      --librispeech_dir examples/data/LibriSpeech/dev-clean \
      --output_dir examples/data/output
"""

import argparse
import binascii
import json
import os
import pickle
import struct
import wave

from pathlib import Path


# ---------------------------------------------------------------------------
# Step 1: LibriSpeech -> NeMo JSONL manifest
# ---------------------------------------------------------------------------

def _get_flac_duration(filepath: str) -> float:
    """Get duration in seconds from a FLAC file using mutagen or fallback."""
    try:
        from mutagen.flac import FLAC
        audio = FLAC(filepath)
        return audio.info.length
    except ImportError:
        # Fallback: use subprocess with soxi or ffprobe
        import subprocess
        try:
            result = subprocess.run(
                ["soxi", "-D", filepath],
                capture_output=True, text=True, check=True,
            )
            return float(result.stdout.strip())
        except FileNotFoundError:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    filepath,
                ],
                capture_output=True, text=True, check=True,
            )
            return float(result.stdout.strip())


def build_nemo_manifest(librispeech_dir: str, output_path: str) -> int:
    """Build a NeMo-style JSONL manifest from LibriSpeech directory.

    NeMo manifest format (one JSON per line):
      {"audio_filepath": "...", "text": "...", "duration": 5.23}
    """
    librispeech_dir = Path(librispeech_dir).resolve()
    records = []

    # Find all transcript files
    trans_files = sorted(librispeech_dir.rglob("*.trans.txt"))
    print(f"  Found {len(trans_files)} transcript files")

    for trans_file in trans_files:
        chapter_dir = trans_file.parent
        with open(trans_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt_id, text = line.split(" ", 1)
                audio_path = chapter_dir / f"{utt_id}.flac"
                if not audio_path.exists():
                    print(f"  WARNING: missing audio {audio_path}")
                    continue

                duration = _get_flac_duration(str(audio_path))
                records.append({
                    "audio_filepath": str(audio_path),
                    "text": text,
                    "duration": round(duration, 3),
                })

    # Write JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"  Wrote {len(records)} records to {output_path}")
    return len(records)


# ---------------------------------------------------------------------------
# Step 2: JSONL -> Parquet
# ---------------------------------------------------------------------------

def jsonl_to_parquet(jsonl_path: str, parquet_path: str):
    """Convert NeMo JSONL manifest to Parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            records.append(json.loads(line))

    table = pa.table({
        "audio_filepath": [r["audio_filepath"] for r in records],
        "text": [r["text"] for r in records],
        "duration": [r["duration"] for r in records],
    })
    pq.write_table(table, parquet_path)
    print(f"  Wrote Parquet: {parquet_path} ({len(records)} rows)")


# ---------------------------------------------------------------------------
# Step 3: JSONL -> TFRecord
# ---------------------------------------------------------------------------

def _masked_crc32c(data: bytes) -> int:
    """Compute masked CRC-32C for TFRecord format (using CRC-32 as stand-in)."""
    crc = binascii.crc32(data) & 0xFFFFFFFF
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def jsonl_to_tfrecord(jsonl_path: str, tfrecord_path: str):
    """Convert NeMo JSONL manifest to TFRecord (raw JSON bytes per record)."""
    count = 0
    with open(jsonl_path, "r") as fin, open(tfrecord_path, "wb") as fout:
        for line in fin:
            data = line.strip().encode("utf-8")
            length = len(data)
            length_bytes = struct.pack("<Q", length)
            fout.write(length_bytes)
            fout.write(struct.pack("<I", _masked_crc32c(length_bytes)))
            fout.write(data)
            fout.write(struct.pack("<I", _masked_crc32c(data)))
            count += 1
    print(f"  Wrote TFRecord: {tfrecord_path} ({count} records)")


# ---------------------------------------------------------------------------
# Step 4: JSONL -> ArrayRecord
# ---------------------------------------------------------------------------

def jsonl_to_arrayrecord(jsonl_path: str, arrayrecord_path: str):
    """Convert NeMo JSONL manifest to ArrayRecord (pickled dicts)."""
    from array_record.python import array_record_module

    writer = array_record_module.ArrayRecordWriter(
        arrayrecord_path, "group_size:1"
    )
    count = 0
    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            writer.write(pickle.dumps(record))
            count += 1
    writer.close()
    print(f"  Wrote ArrayRecord: {arrayrecord_path} ({count} records)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare LibriSpeech dev-clean in multiple formats."
    )
    parser.add_argument(
        "--librispeech_dir",
        default="examples/data/LibriSpeech/dev-clean",
        help="Path to extracted LibriSpeech dev-clean directory.",
    )
    parser.add_argument(
        "--output_dir",
        default="examples/data/output",
        help="Directory for output files.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, "librispeech_dev_clean.jsonl")
    parquet_path = os.path.join(output_dir, "librispeech_dev_clean.parquet")
    tfrecord_path = os.path.join(output_dir, "librispeech_dev_clean.tfrecord")
    arrayrecord_path = os.path.join(output_dir, "librispeech_dev_clean.array_record")

    print("Step 1: Building NeMo JSONL manifest...")
    build_nemo_manifest(args.librispeech_dir, jsonl_path)

    print("\nStep 2: Converting JSONL -> Parquet...")
    jsonl_to_parquet(jsonl_path, parquet_path)

    print("\nStep 3: Converting JSONL -> TFRecord...")
    jsonl_to_tfrecord(jsonl_path, tfrecord_path)

    print("\nStep 4: Converting JSONL -> ArrayRecord...")
    try:
        jsonl_to_arrayrecord(jsonl_path, arrayrecord_path)
    except ImportError as e:
        print(f"  Skipped ArrayRecord (missing dependency): {e}")

    print("\nDone! Output files:")
    for f in [jsonl_path, parquet_path, tfrecord_path, arrayrecord_path]:
        if os.path.exists(f):
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"  {f} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
