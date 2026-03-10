"""Example script demonstrating Grain with multiple data sources.

Covers: Parquet, TFRecord, ArrayRecord, and JSONL (NeMo-style).

Usage:
  pip install grain pyarrow array-record
  python examples/multi_source_example.py --source parquet
  python examples/multi_source_example.py --source tfrecord
  python examples/multi_source_example.py --source arrayrecord
  python examples/multi_source_example.py --source jsonl
  python examples/multi_source_example.py --source all
"""

import argparse
import json
import os
import pickle
import struct
import tempfile
from typing import Any

import grain
import numpy as np


# ---------------------------------------------------------------------------
# Helpers to generate sample data files
# ---------------------------------------------------------------------------

SAMPLE_RECORDS = [
    {"text": "Grain is a data loading library for JAX.", "label": 0},
    {"text": "It supports deterministic data pipelines.", "label": 1},
    {"text": "MapDataset provides random access.", "label": 0},
    {"text": "IterDataset supports streaming reads.", "label": 1},
    {"text": "Multiple data sources are supported.", "label": 0},
]


def _write_parquet(path: str) -> str:
    """Write sample data to a Parquet file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        "text": [r["text"] for r in SAMPLE_RECORDS],
        "label": [r["label"] for r in SAMPLE_RECORDS],
    })
    filepath = os.path.join(path, "sample.parquet")
    pq.write_table(table, filepath)
    print(f"  Wrote Parquet file: {filepath}")
    return filepath


def _masked_crc32c(data: bytes) -> int:
    """Compute masked CRC-32C for TFRecord format."""
    import struct

    # Use a simple CRC-32 as a stand-in (TFRecord technically uses CRC-32C).
    # For demo purposes this is sufficient; real TFRecord writers use CRC-32C.
    import binascii

    crc = binascii.crc32(data) & 0xFFFFFFFF
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _write_tfrecord(path: str) -> str:
    """Write sample data to a TFRecord file (raw bytes, no protobuf)."""
    filepath = os.path.join(path, "sample.tfrecord")
    with open(filepath, "wb") as f:
        for record in SAMPLE_RECORDS:
            data = json.dumps(record).encode("utf-8")
            length = len(data)
            length_bytes = struct.pack("<Q", length)
            f.write(length_bytes)
            f.write(struct.pack("<I", _masked_crc32c(length_bytes)))
            f.write(data)
            f.write(struct.pack("<I", _masked_crc32c(data)))
    print(f"  Wrote TFRecord file: {filepath}")
    return filepath


def _write_arrayrecord(path: str) -> str:
    """Write sample data to an ArrayRecord file."""
    from array_record.python import array_record_module

    filepath = os.path.join(path, "sample.array_record")
    writer = array_record_module.ArrayRecordWriter(filepath, "group_size:1")
    for record in SAMPLE_RECORDS:
        writer.write(pickle.dumps(record))
    writer.close()
    print(f"  Wrote ArrayRecord file: {filepath}")
    return filepath


def _write_jsonl(path: str) -> str:
    """Write sample data to a JSONL file (NeMo-style)."""
    filepath = os.path.join(path, "sample.jsonl")
    with open(filepath, "w") as f:
        for record in SAMPLE_RECORDS:
            f.write(json.dumps(record) + "\n")
    print(f"  Wrote JSONL file: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Custom transforms
# ---------------------------------------------------------------------------

class AddLength(grain.transforms.Map):
    """Adds a 'text_length' field to each dict element."""

    def map(self, element: dict[str, Any]) -> dict[str, Any]:
        element["text_length"] = len(element["text"])
        return element


class ParseBytes(grain.transforms.Map):
    """Unpickle bytes into a Python object."""

    def map(self, element: bytes) -> dict[str, Any]:
        return pickle.loads(element)


class ParseJSON(grain.transforms.Map):
    """Parse a JSON bytes/string into a Python dict."""

    def map(self, element) -> dict[str, Any]:
        if isinstance(element, bytes):
            element = element.decode("utf-8")
        return json.loads(element)


# ---------------------------------------------------------------------------
# JSONL data source (random-access via line offsets)
# ---------------------------------------------------------------------------

class JsonlDataSource:
    """A random-access data source for JSONL files.

    Builds a line-offset index on init so that individual records can be
    retrieved by integer index, satisfying Grain's RandomAccessDataSource
    protocol (``__len__`` + ``__getitem__``).

    This is useful for NeMo-style JSONL datasets.
    """

    def __init__(self, path: str):
        self._path = path
        self._offsets: list[int] = []
        with open(path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self._offsets.append(offset)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> str:
        with open(self._path, "r") as f:
            f.seek(self._offsets[index])
            return f.readline().strip()


# ---------------------------------------------------------------------------
# Example runners
# ---------------------------------------------------------------------------

def _print_dataset(ds, name: str, n: int = 3):
    """Print up to n elements from a dataset."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    for i, element in enumerate(ds):
        if i >= n:
            print(f"  ... ({len(list(ds)) if hasattr(ds, '__len__') else '?'} total elements)")
            break
        print(f"  [{i}] {element}")
    print()


def run_parquet(tmpdir: str):
    """Demonstrate reading from a Parquet file."""
    filepath = _write_parquet(tmpdir)

    # ParquetIterDataset is an IterDataset (streaming, no random access).
    ds = (
        grain.experimental.ParquetIterDataset(filepath)
        .map(AddLength())
    )
    _print_dataset(ds, "Parquet (IterDataset)")


def run_tfrecord(tmpdir: str):
    """Demonstrate reading from a TFRecord file."""
    filepath = _write_tfrecord(tmpdir)

    # TFRecordIterDataset is an IterDataset (streaming).
    # Records are raw bytes; we parse them with ParseJSON.
    ds = (
        grain.experimental.TFRecordIterDataset(filepath)
        .map(ParseJSON())
        .map(AddLength())
    )
    _print_dataset(ds, "TFRecord (IterDataset)")


def run_arrayrecord(tmpdir: str):
    """Demonstrate reading from an ArrayRecord file."""
    filepath = _write_arrayrecord(tmpdir)

    source = grain.sources.ArrayRecordDataSource(filepath)
    print(f"  ArrayRecord source length: {len(source)}")

    # ArrayRecordDataSource is random-access -> use MapDataset.
    ds = (
        grain.MapDataset.source(source)
        .shuffle(seed=42)
        .map(ParseBytes())
        .map(AddLength())
        .batch(batch_size=2)
    )
    _print_dataset(ds, "ArrayRecord (MapDataset)")


def run_jsonl(tmpdir: str):
    """Demonstrate reading from a JSONL file (NeMo-style)."""
    filepath = _write_jsonl(tmpdir)

    source = JsonlDataSource(filepath)
    print(f"  JSONL source length: {len(source)}")

    # JsonlDataSource is random-access -> use MapDataset.
    ds = (
        grain.MapDataset.source(source)
        .shuffle(seed=42)
        .map(ParseJSON())
        .map(AddLength())
        .batch(batch_size=2)
    )
    _print_dataset(ds, "JSONL / NeMo (MapDataset)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

RUNNERS = {
    "parquet": run_parquet,
    "tfrecord": run_tfrecord,
    "arrayrecord": run_arrayrecord,
    "jsonl": run_jsonl,
}


def main():
    parser = argparse.ArgumentParser(
        description="Grain multi-source example."
    )
    parser.add_argument(
        "--source",
        choices=list(RUNNERS.keys()) + ["all"],
        default="all",
        help="Which data source to demo (default: all).",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        if args.source == "all":
            for name, runner in RUNNERS.items():
                try:
                    runner(tmpdir)
                except ImportError as e:
                    print(f"\n  Skipping {name}: {e}\n")
        else:
            RUNNERS[args.source](tmpdir)


if __name__ == "__main__":
    main()
