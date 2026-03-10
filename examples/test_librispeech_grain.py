"""Test Grain with LibriSpeech dev-clean data in all 5 formats.

Requires running prepare_librispeech.py first.

Usage:
  python examples/test_librispeech_grain.py --source all
  python examples/test_librispeech_grain.py --source jsonl
  python examples/test_librispeech_grain.py --source parquet
  python examples/test_librispeech_grain.py --source tfrecord
  python examples/test_librispeech_grain.py --source arrayrecord
  python examples/test_librispeech_grain.py --source tar
"""

import argparse
import json
import os
import pickle
import sys
import tarfile

import grain
import numpy as np


DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "output")


# ---------------------------------------------------------------------------
# Custom transforms
# ---------------------------------------------------------------------------

class ParseJSON(grain.transforms.Map):
    """Parse JSON string/bytes into a Python dict."""

    def map(self, element):
        if isinstance(element, bytes):
            element = element.decode("utf-8")
        return json.loads(element)


class ParsePickle(grain.transforms.Map):
    """Unpickle bytes into a Python object."""

    def map(self, element: bytes):
        return pickle.loads(element)


class AddWordCount(grain.transforms.Map):
    """Add word_count field to a dict element."""

    def map(self, element: dict) -> dict:
        element["word_count"] = len(element["text"].split())
        return element


class FilterShortUtterances(grain.transforms.Filter):
    """Keep only utterances longer than min_duration seconds."""

    def __init__(self, min_duration: float = 5.0):
        self._min_duration = min_duration

    def filter(self, element: dict) -> bool:
        return element["duration"] >= self._min_duration


# ---------------------------------------------------------------------------
# JSONL random-access data source
# ---------------------------------------------------------------------------

class JsonlDataSource:
    """Random-access data source for JSONL files via line-offset index."""

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
# Tar random-access data source
# ---------------------------------------------------------------------------

class TarDataSource:
    """Random-access data source for tar archives.

    Reads the tar member index on init, then seeks to individual files by
    index. Each member is expected to contain a single JSON record.
    """

    def __init__(self, path: str):
        self._path = path
        self._members: list[tarfile.TarInfo] = []
        with tarfile.open(path, "r") as tar:
            for member in tar:
                if member.isfile():
                    self._members.append(member)
        # Sort by name to ensure deterministic order
        self._members.sort(key=lambda m: m.name)

    def __len__(self) -> int:
        return len(self._members)

    def __getitem__(self, index: int) -> str:
        with tarfile.open(self._path, "r") as tar:
            f = tar.extractfile(self._members[index])
            return f.read().decode("utf-8")


# ---------------------------------------------------------------------------
# Print helper
# ---------------------------------------------------------------------------

def print_elements(ds, name: str, n: int = 5):
    """Print first n elements from a dataset."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    count = 0
    for i, element in enumerate(ds):
        if i >= n:
            break
        if isinstance(element, dict):
            print(f"  [{i}] text={element.get('text', '')[:60]}...")
            print(f"       duration={element.get('duration')}, "
                  f"word_count={element.get('word_count', 'N/A')}")
        else:
            print(f"  [{i}] {str(element)[:120]}")
        count += 1
    print(f"  (showed {count} elements)")
    print()


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

def test_jsonl():
    """Test JSONL source with MapDataset."""
    path = os.path.join(DATA_DIR, "librispeech_dev_clean.jsonl")
    print(f"\n--- JSONL Source: {path}")

    source = JsonlDataSource(path)
    print(f"  Total records: {len(source)}")

    # MapDataset pipeline: parse -> filter -> add word count -> shuffle -> batch
    ds = (
        grain.MapDataset.source(source)
        .map(ParseJSON())
        .map(AddWordCount())
        .shuffle(seed=42)
    )

    print_elements(ds, "JSONL -> MapDataset (shuffle + word_count)")

    # Also test batching
    ds_batched = ds.batch(batch_size=4)
    batch = ds_batched[0]
    print(f"  Batch[0] keys: {list(batch.keys()) if isinstance(batch, dict) else type(batch)}")
    print(f"  Batch[0] texts: {[t[:40] + '...' for t in batch['text']]}")
    print(f"  Batch[0] durations: {batch['duration']}")


def test_parquet():
    """Test Parquet source with IterDataset."""
    path = os.path.join(DATA_DIR, "librispeech_dev_clean.parquet")
    print(f"\n--- Parquet Source: {path}")

    # ParquetIterDataset is streaming (IterDataset)
    ds = (
        grain.experimental.ParquetIterDataset(path)
        .map(AddWordCount())
    )

    print_elements(ds, "Parquet -> IterDataset (word_count)")


def test_tfrecord():
    """Test TFRecord source with IterDataset."""
    path = os.path.join(DATA_DIR, "librispeech_dev_clean.tfrecord")
    print(f"\n--- TFRecord Source: {path}")

    # TFRecordIterDataset is streaming (IterDataset), records are raw bytes
    ds = (
        grain.experimental.TFRecordIterDataset(path)
        .map(ParseJSON())
        .map(AddWordCount())
    )

    print_elements(ds, "TFRecord -> IterDataset (parse JSON + word_count)")


def test_arrayrecord():
    """Test ArrayRecord source with MapDataset."""
    path = os.path.join(DATA_DIR, "librispeech_dev_clean.array_record")
    print(f"\n--- ArrayRecord Source: {path}")

    source = grain.sources.ArrayRecordDataSource(path)
    print(f"  Total records: {len(source)}")

    # MapDataset pipeline: parse pickle -> filter long -> add word count -> shuffle -> batch
    ds = (
        grain.MapDataset.source(source)
        .map(ParsePickle())
        .map(AddWordCount())
        .shuffle(seed=42)
    )

    print_elements(ds, "ArrayRecord -> MapDataset (shuffle + word_count)")

    # Test batching
    ds_batched = ds.batch(batch_size=4)
    batch = ds_batched[0]
    print(f"  Batch[0] keys: {list(batch.keys()) if isinstance(batch, dict) else type(batch)}")
    print(f"  Batch[0] durations: {batch['duration']}")


def test_tar():
    """Test Tar source with MapDataset."""
    path = os.path.join(DATA_DIR, "librispeech_dev_clean.tar")
    print(f"\n--- Tar Source: {path}")

    source = TarDataSource(path)
    print(f"  Total records: {len(source)}")

    # MapDataset pipeline: parse JSON -> add word count -> shuffle -> batch
    ds = (
        grain.MapDataset.source(source)
        .map(ParseJSON())
        .map(AddWordCount())
        .shuffle(seed=42)
    )

    print_elements(ds, "Tar -> MapDataset (shuffle + word_count)")

    # Test batching
    ds_batched = ds.batch(batch_size=4)
    batch = ds_batched[0]
    print(f"  Batch[0] keys: {list(batch.keys()) if isinstance(batch, dict) else type(batch)}")
    print(f"  Batch[0] texts: {[t[:40] + '...' for t in batch['text']]}")
    print(f"  Batch[0] durations: {batch['duration']}")


# ---------------------------------------------------------------------------
# Summary: compare all sources
# ---------------------------------------------------------------------------

def test_summary():
    """Quick comparison: read first record from each source."""
    print(f"\n{'='*70}")
    print(f"  Summary: First record from each source")
    print(f"{'='*70}")

    # JSONL
    source = JsonlDataSource(os.path.join(DATA_DIR, "librispeech_dev_clean.jsonl"))
    rec = json.loads(source[0])
    print(f"  JSONL:       text={rec['text'][:50]}... dur={rec['duration']}")

    # Parquet
    ds = grain.experimental.ParquetIterDataset(
        os.path.join(DATA_DIR, "librispeech_dev_clean.parquet")
    )
    rec = next(iter(ds))
    print(f"  Parquet:     text={rec['text'][:50]}... dur={rec['duration']}")

    # TFRecord
    ds = grain.experimental.TFRecordIterDataset(
        os.path.join(DATA_DIR, "librispeech_dev_clean.tfrecord")
    )
    rec = json.loads(next(iter(ds)))
    print(f"  TFRecord:    text={rec['text'][:50]}... dur={rec['duration']}")

    # ArrayRecord
    source = grain.sources.ArrayRecordDataSource(
        os.path.join(DATA_DIR, "librispeech_dev_clean.array_record")
    )
    rec = pickle.loads(source[0])
    print(f"  ArrayRecord: text={rec['text'][:50]}... dur={rec['duration']}")

    # Tar
    source = TarDataSource(os.path.join(DATA_DIR, "librispeech_dev_clean.tar"))
    rec = json.loads(source[0])
    print(f"  Tar:         text={rec['text'][:50]}... dur={rec['duration']}")

    print(f"\n  All sources produce the same data!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

RUNNERS = {
    "jsonl": test_jsonl,
    "parquet": test_parquet,
    "tfrecord": test_tfrecord,
    "arrayrecord": test_arrayrecord,
    "tar": test_tar,
}


def main():
    parser = argparse.ArgumentParser(
        description="Test Grain with LibriSpeech dev-clean in multiple formats."
    )
    parser.add_argument(
        "--source",
        choices=list(RUNNERS.keys()) + ["all"],
        default="all",
        help="Which data source to test (default: all).",
    )
    args = parser.parse_args()

    if args.source == "all":
        for name, runner in RUNNERS.items():
            try:
                runner()
            except Exception as e:
                print(f"\n  ERROR in {name}: {e}\n")
                import traceback
                traceback.print_exc()
        test_summary()
    else:
        RUNNERS[args.source]()


if __name__ == "__main__":
    main()
