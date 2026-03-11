# Lhotse Data Management System

A metadata-driven system for managing large-scale speech datasets. Maintains a
Parquet catalog for analysis and refinement, and versions training datasets by
symlinking selected Lhotse Shar shards — no audio duplication.

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Raw Lhotse Shar Data                        │
│                                                                     │
│  shar_pool/                                                         │
│  ├── librispeech/                                                   │
│  │   ├── cuts.000000.jsonl.gz   recording.000000.tar               │
│  │   ├── cuts.000001.jsonl.gz   recording.000001.tar               │
│  │   └── ...                                                        │
│  ├── commonvoice/                                                   │
│  │   ├── cuts.000000.jsonl.gz   recording.000000.tar               │
│  │   └── ...                                                        │
│  └── internal/                                                      │
│      └── ...                                                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                    ingest (one-time)
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Parquet Metadata Catalog                        │
│                                                                     │
│  catalog/metadata.parquet                                           │
│                                                                     │
│  ┌─────────┬──────┬─────────┬──────┬─────────┬──────┬───────────┐  │
│  │ id      │ dur  │ speaker │ text │ sr      │shard │ cut_json  │  │
│  ├─────────┼──────┼─────────┼──────┼─────────┼──────┼───────────┤  │
│  │ utt-001 │ 5.8  │ spk_A   │ ...  │ 16000   │ 000  │ {full...} │  │
│  │ utt-002 │ 3.1  │ spk_B   │ ...  │ 16000   │ 000  │ {full...} │  │
│  │ ...     │      │         │      │         │      │           │  │
│  └─────────┴──────┴─────────┴──────┴─────────┴──────┴───────────┘  │
│                                                                     │
│  Queryable with DuckDB:                                             │
│    - Filter by duration, speaker, language, source                  │
│    - Data quality checks, outlier detection                         │
│    - Export filtered subsets                                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                  query + select shards
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Versioned Training Datasets                      │
│                                                                     │
│  versions/                                                          │
│  ├── v1_baseline/                                                   │
│  │   ├── cuts.000000.jsonl.gz  -> ../../shar_pool/librispeech/...  │
│  │   ├── recording.000000.tar  -> ../../shar_pool/librispeech/...  │
│  │   ├── cuts.000001.jsonl.gz  -> ../../shar_pool/commonvoice/...  │
│  │   ├── recording.000001.tar  -> ../../shar_pool/commonvoice/...  │
│  │   └── VERSION.json          (metadata: date, query, stats)      │
│  │                                                                  │
│  ├── v2_long_only/             (duration >= 5s subset)              │
│  │   ├── cuts.000003.jsonl.gz  -> ../../shar_pool/...              │
│  │   ├── recording.000003.tar  -> ../../shar_pool/...              │
│  │   └── VERSION.json                                               │
│  │                                                                  │
│  └── v3_en_finetune/           (English speakers only)              │
│      ├── ...                   -> symlinks                          │
│      └── VERSION.json                                               │
│                                                                     │
│  Symlinks only — no audio copied. Each version is a different       │
│  view of the same underlying shard pool.                            │
└─────────────────────────────────────────────────────────────────────┘
```

## How It Works

### 1. Ingest: Shar → Parquet Catalog

Convert Lhotse Shar directories into a single queryable Parquet table. Each
row stores flat columns for fast queries plus the full original cut JSON so
nothing is lost.

```bash
python lhotse_shar_to_parquet.py \
    --shar_dir /data/shar_pool/librispeech /data/shar_pool/commonvoice \
    --output_dir /data/catalog
```

Output: one `metadata.parquet` covering all sources.

### 2. Analyze: DuckDB Queries

Query the catalog without loading data into memory. Works at 100M+ rows.

```bash
# Full analysis
python query_metadata.py --parquet /data/catalog/metadata.parquet

# Specific sections
python query_metadata.py --parquet /data/catalog/metadata.parquet --query duration quality outliers
```

Available queries: `overview`, `duration`, `speaker`, `text`, `quality`,
`source`, `gender`, `language`, `crosstab`, `outliers`, `export`.

### 3. Refine: Filter with SQL

Use DuckDB to filter and export subsets. The `export` query section shows
ready-to-use examples.

```sql
-- Remove outliers
COPY (SELECT * FROM meta WHERE duration BETWEEN 2 AND 30) TO 'clean.parquet';

-- Balanced speaker sampling
COPY (
    SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY speaker ORDER BY RANDOM()) AS rn
        FROM meta
    ) WHERE rn <= 100
) TO 'balanced.parquet';
```

### 4. Version: Symlink Shard Pairs

Create a training dataset version by symlinking selected shard pairs (cuts +
recording tar) into a version folder. No audio is copied.

```bash
# Identify which shards contain the data you want
duckdb -c "
    SELECT DISTINCT source_dir, shard
    FROM 'catalog/metadata.parquet'
    WHERE duration BETWEEN 5 AND 15
      AND speaker IN (SELECT speaker FROM 'catalog/metadata.parquet'
                      GROUP BY speaker HAVING SUM(duration)/3600 > 1)
"

# Create version folder with symlinks
mkdir -p versions/v2_long_only
ln -s /data/shar_pool/librispeech/cuts.000003.jsonl.gz   versions/v2_long_only/
ln -s /data/shar_pool/librispeech/recording.000003.tar   versions/v2_long_only/
# ... repeat for each selected shard pair
```

Each version folder is a valid Lhotse Shar directory — Lhotse reads it
directly for training.

## Directory Layout

```
project/
├── shar_pool/                   # Raw data (read-only after ingest)
│   ├── librispeech/
│   ├── commonvoice/
│   └── internal/
│
├── catalog/                     # Parquet metadata (rebuilt on ingest)
│   └── metadata.parquet
│
├── versions/                    # Versioned training sets (symlinks only)
│   ├── v1_baseline/
│   │   ├── cuts.*.jsonl.gz     -> symlinks to shar_pool
│   │   ├── recording.*.tar     -> symlinks to shar_pool
│   │   └── VERSION.json
│   ├── v2_long_only/
│   └── v3_en_finetune/
│
└── scripts/                     # This tooling
    ├── lhotse_shar_to_parquet.py
    ├── query_metadata.py
    ├── run_lhotse_to_parquet.sh
    └── requirements.txt
```

## Key Design Decisions

**Why Parquet for metadata?**
- Columnar format — queries only read needed columns
- DuckDB scans 100M+ rows in seconds without loading into memory
- Single file replaces thousands of scattered `cuts.*.jsonl.gz` files
- Full cut JSON stored alongside flat columns — nothing lost

**Why symlinks for versioning?**
- Audio files are large (TB scale) — copying is wasteful and slow
- Symlinks are instant and use zero disk space
- Each version is a valid Lhotse Shar directory — no format conversion
- Easy to create, delete, or modify versions without touching source data

**Why shard-level granularity?**
- Lhotse Shar reads cuts + recording tar as a pair — can't split within a shard
- Shard-level selection is the natural atomic unit
- For finer filtering, re-shard the selected data into new shards

## Workflow Example

```
1. Receive new data batch
   └── Ingest into shar_pool/batch_20260310/

2. Update catalog
   └── python lhotse_shar_to_parquet.py --shar_dir shar_pool/* --output_dir catalog/

3. Analyze
   └── python query_metadata.py --parquet catalog/metadata.parquet

4. Find quality issues
   └── --query quality outliers
       "Found 1,200 utterances with duration > 60s, 500 with empty text"

5. Create clean training version
   └── DuckDB query → select shards → symlink into versions/v4_clean/

6. Train
   └── CutSet.from_shar(fields={"recording": "versions/v4_clean/"})
```

## Requirements

```
pip install -r requirements.txt
```

- `lhotse` — reading Shar cuts
- `pyarrow` — writing Parquet
- `duckdb` — querying metadata
