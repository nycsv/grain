"""Query and analyze the Parquet metadata table using DuckDB.

Scales to 100M+ rows without loading everything into memory.

Usage:
  # Run all queries
  python examples/lhotse/query_metadata.py --parquet metadata.parquet

  # Run specific query sections
  python examples/lhotse/query_metadata.py --parquet metadata.parquet --query overview duration
  python examples/lhotse/query_metadata.py --parquet metadata.parquet --query quality

  # Multiple parquet files
  python examples/lhotse/query_metadata.py --parquet shard1.parquet shard2.parquet

  # List available queries
  python examples/lhotse/query_metadata.py --parquet metadata.parquet --query list
"""

import argparse
import duckdb


# ---------------------------------------------------------------------------
# Query sections
# ---------------------------------------------------------------------------

def query_overview(conn):
    """Basic overview of the dataset."""
    print("=== Overview ===")
    row_count = conn.sql("SELECT COUNT(*) FROM meta").fetchone()[0]
    columns = [col[0] for col in conn.sql("DESCRIBE meta").fetchall()]
    print(f"  Total rows:    {row_count:,}")
    print(f"  Columns ({len(columns)}): {columns}")
    conn.sql("""
        SELECT
            COUNT(DISTINCT speaker) AS unique_speakers,
            COUNT(DISTINCT source_dir) AS unique_sources,
            COUNT(DISTINCT shard) AS unique_shards,
            COUNT(DISTINCT sampling_rate) AS unique_sample_rates
        FROM meta
    """).show()


def query_duration(conn):
    """Duration statistics."""
    print("=== Duration Stats ===")
    stats = conn.sql("""
        SELECT
            SUM(duration) / 3600 AS total_hours,
            AVG(duration) AS mean_s,
            MEDIAN(duration) AS median_s,
            MIN(duration) AS min_s,
            MAX(duration) AS max_s,
            STDDEV(duration) AS std_s,
            PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY duration) AS p5,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY duration) AS p25,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY duration) AS p75,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration) AS p95,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration) AS p99
        FROM meta
    """).fetchone()
    print(f"  Total hours:  {stats[0]:,.2f}")
    print(f"  Mean:         {stats[1]:.2f}s")
    print(f"  Median:       {stats[2]:.2f}s")
    print(f"  Min:          {stats[3]:.2f}s")
    print(f"  Max:          {stats[4]:.2f}s")
    print(f"  Std:          {stats[5]:.2f}s")
    print(f"  Percentiles:  P5={stats[6]:.2f}s  P25={stats[7]:.2f}s  P75={stats[8]:.2f}s  P95={stats[9]:.2f}s  P99={stats[10]:.2f}s")

    # Duration distribution histogram
    print("\n=== Duration Distribution ===")
    dist = conn.sql("""
        SELECT
            CASE
                WHEN duration < 1 THEN '0-1s'
                WHEN duration < 2 THEN '1-2s'
                WHEN duration < 5 THEN '2-5s'
                WHEN duration < 10 THEN '5-10s'
                WHEN duration < 15 THEN '10-15s'
                WHEN duration < 20 THEN '15-20s'
                WHEN duration < 30 THEN '20-30s'
                WHEN duration < 60 THEN '30-60s'
                ELSE '60s+'
            END AS bin,
            COUNT(*) AS count,
            ROUND(SUM(duration) / 3600, 2) AS hours,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
        FROM meta
        GROUP BY bin
        ORDER BY
            CASE bin
                WHEN '0-1s' THEN 1 WHEN '1-2s' THEN 2
                WHEN '2-5s' THEN 3 WHEN '5-10s' THEN 4
                WHEN '10-15s' THEN 5 WHEN '15-20s' THEN 6
                WHEN '20-30s' THEN 7 WHEN '30-60s' THEN 8
                WHEN '60s+' THEN 9
            END
    """).fetchall()
    for bin_label, count, hours, pct in dist:
        bar = "#" * int(pct / 2)
        print(f"  {bin_label:>7s}: {count:>10,} ({pct:5.1f}%) {hours:>8.1f}h {bar}")


def query_speaker(conn):
    """Speaker-level analysis."""
    print("=== Speaker Summary ===")
    conn.sql("""
        SELECT
            COUNT(DISTINCT speaker) AS total_speakers,
            ROUND(AVG(spk_hours), 2) AS mean_hours_per_speaker,
            ROUND(MEDIAN(spk_hours), 2) AS median_hours_per_speaker,
            ROUND(MIN(spk_hours), 2) AS min_hours,
            ROUND(MAX(spk_hours), 2) AS max_hours
        FROM (
            SELECT speaker, SUM(duration) / 3600 AS spk_hours
            FROM meta GROUP BY speaker
        )
    """).show()

    print("=== Top 20 Speakers by Hours ===")
    conn.sql("""
        SELECT
            speaker,
            COUNT(*) AS utterances,
            ROUND(SUM(duration) / 3600, 2) AS hours,
            ROUND(AVG(duration), 2) AS mean_dur,
            ROUND(MIN(duration), 2) AS min_dur,
            ROUND(MAX(duration), 2) AS max_dur
        FROM meta
        GROUP BY speaker
        ORDER BY hours DESC
        LIMIT 20
    """).show()

    print("=== Bottom 10 Speakers by Hours ===")
    conn.sql("""
        SELECT
            speaker,
            COUNT(*) AS utterances,
            ROUND(SUM(duration) / 3600, 2) AS hours
        FROM meta
        GROUP BY speaker
        ORDER BY hours ASC
        LIMIT 10
    """).show()

    print("=== Speaker Hours Distribution ===")
    dist = conn.sql("""
        SELECT
            CASE
                WHEN spk_hours < 0.1 THEN '<6min'
                WHEN spk_hours < 0.5 THEN '6-30min'
                WHEN spk_hours < 1 THEN '30min-1h'
                WHEN spk_hours < 5 THEN '1-5h'
                WHEN spk_hours < 10 THEN '5-10h'
                WHEN spk_hours < 50 THEN '10-50h'
                ELSE '50h+'
            END AS bin,
            COUNT(*) AS num_speakers,
            ROUND(SUM(spk_hours), 1) AS total_hours
        FROM (
            SELECT speaker, SUM(duration) / 3600 AS spk_hours
            FROM meta GROUP BY speaker
        )
        GROUP BY bin
        ORDER BY
            CASE bin
                WHEN '<6min' THEN 1 WHEN '6-30min' THEN 2
                WHEN '30min-1h' THEN 3 WHEN '1-5h' THEN 4
                WHEN '5-10h' THEN 5 WHEN '10-50h' THEN 6
                WHEN '50h+' THEN 7
            END
    """).fetchall()
    for bin_label, num_speakers, total_hours in dist:
        print(f"  {bin_label:>10s}: {num_speakers:>6,} speakers  ({total_hours:>8.1f}h total)")


def query_text(conn):
    """Text and transcript analysis."""
    print("=== Text Stats ===")
    conn.sql("""
        SELECT
            ROUND(AVG(LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1), 1) AS mean_words,
            MAX(LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1) AS max_words,
            MIN(LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1) AS min_words,
            ROUND(AVG(LENGTH(text)), 1) AS mean_chars,
            MAX(LENGTH(text)) AS max_chars,
            SUM(CASE WHEN text = '' OR text IS NULL THEN 1 ELSE 0 END) AS empty_count
        FROM meta
    """).show()

    print("=== Word Count Distribution ===")
    dist = conn.sql("""
        WITH word_counts AS (
            SELECT LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1 AS wc
            FROM meta
            WHERE text IS NOT NULL AND text != ''
        )
        SELECT
            CASE
                WHEN wc <= 1 THEN '1 word'
                WHEN wc <= 5 THEN '2-5'
                WHEN wc <= 10 THEN '6-10'
                WHEN wc <= 20 THEN '11-20'
                WHEN wc <= 50 THEN '21-50'
                WHEN wc <= 100 THEN '51-100'
                ELSE '100+'
            END AS bin,
            COUNT(*) AS count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
        FROM word_counts
        GROUP BY bin
        ORDER BY
            CASE bin
                WHEN '1 word' THEN 1 WHEN '2-5' THEN 2
                WHEN '6-10' THEN 3 WHEN '11-20' THEN 4
                WHEN '21-50' THEN 5 WHEN '51-100' THEN 6
                WHEN '100+' THEN 7
            END
    """).fetchall()
    for bin_label, count, pct in dist:
        bar = "#" * int(pct / 2)
        print(f"  {bin_label:>8s}: {count:>10,} ({pct:5.1f}%) {bar}")

    print("\n=== Longest Transcripts ===")
    conn.sql("""
        SELECT
            id,
            speaker,
            ROUND(duration, 2) AS dur,
            LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1 AS words,
            LEFT(text, 80) || '...' AS text_preview
        FROM meta
        WHERE text IS NOT NULL AND text != ''
        ORDER BY LENGTH(text) DESC
        LIMIT 5
    """).show()

    print("=== Shortest Transcripts (non-empty) ===")
    conn.sql("""
        SELECT
            id,
            speaker,
            ROUND(duration, 2) AS dur,
            LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1 AS words,
            text
        FROM meta
        WHERE text IS NOT NULL AND text != ''
        ORDER BY LENGTH(text) ASC
        LIMIT 5
    """).show()


def query_quality(conn):
    """Data quality checks."""
    print("=== Data Quality Checks ===\n")

    # Duplicates
    dup_count = conn.sql("""
        SELECT COUNT(*) FROM (
            SELECT id FROM meta GROUP BY id HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    print(f"  Duplicate IDs: {dup_count:,}")
    if dup_count > 0:
        print("  Top duplicates:")
        conn.sql("""
            SELECT id, COUNT(*) AS occurrences
            FROM meta GROUP BY id HAVING COUNT(*) > 1
            ORDER BY occurrences DESC LIMIT 5
        """).show()

    # Empty/null fields
    print("  Missing/empty values:")
    for col in ["text", "speaker", "language", "gender"]:
        try:
            empty = conn.sql(f"""
                SELECT COUNT(*) FROM meta
                WHERE {col} IS NULL OR {col} = ''
            """).fetchone()[0]
            total = conn.sql("SELECT COUNT(*) FROM meta").fetchone()[0]
            pct = 100.0 * empty / total if total > 0 else 0
            print(f"    {col:>15s}: {empty:>10,} empty ({pct:.1f}%)")
        except Exception:
            pass

    # Suspicious durations
    print("\n  Suspicious durations:")
    zero_dur = conn.sql("SELECT COUNT(*) FROM meta WHERE duration <= 0").fetchone()[0]
    tiny_dur = conn.sql("SELECT COUNT(*) FROM meta WHERE duration > 0 AND duration < 0.1").fetchone()[0]
    huge_dur = conn.sql("SELECT COUNT(*) FROM meta WHERE duration > 300").fetchone()[0]
    print(f"    duration <= 0:    {zero_dur:,}")
    print(f"    duration < 0.1s:  {tiny_dur:,}")
    print(f"    duration > 5min:  {huge_dur:,}")

    if huge_dur > 0:
        print("    Longest utterances:")
        conn.sql("""
            SELECT id, speaker, ROUND(duration, 2) AS dur, LEFT(text, 60) AS text_preview
            FROM meta ORDER BY duration DESC LIMIT 5
        """).show()

    # Text-duration mismatch (very short duration but long text, or vice versa)
    print("  Text-duration mismatches:")
    conn.sql("""
        SELECT
            id, speaker,
            ROUND(duration, 2) AS dur,
            LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1 AS words,
            ROUND((LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1) / NULLIF(duration, 0), 1) AS words_per_sec,
            LEFT(text, 50) AS text_preview
        FROM meta
        WHERE text IS NOT NULL AND text != '' AND duration > 0
        ORDER BY (LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1) / duration DESC
        LIMIT 5
    """).show()

    # Sampling rate consistency
    print("  Sampling rate breakdown:")
    conn.sql("""
        SELECT sampling_rate, COUNT(*) AS count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
        FROM meta GROUP BY sampling_rate ORDER BY count DESC
    """).show()


def query_source(conn):
    """Per source directory / shard analysis."""
    columns = [col[0] for col in conn.sql("DESCRIBE meta").fetchall()]

    if "source_dir" in columns:
        print("=== Per Source Directory ===")
        conn.sql("""
            SELECT
                source_dir,
                COUNT(*) AS utterances,
                COUNT(DISTINCT speaker) AS speakers,
                ROUND(SUM(duration) / 3600, 2) AS hours,
                ROUND(AVG(duration), 2) AS mean_dur,
                ROUND(MIN(duration), 2) AS min_dur,
                ROUND(MAX(duration), 2) AS max_dur
            FROM meta
            GROUP BY source_dir
            ORDER BY hours DESC
        """).show()

    if "shard" in columns:
        print("=== Per Shard ===")
        conn.sql("""
            SELECT
                shard,
                COUNT(*) AS utterances,
                ROUND(SUM(duration) / 3600, 4) AS hours,
                ROUND(AVG(duration), 2) AS mean_dur
            FROM meta
            GROUP BY shard
            ORDER BY shard
        """).show()

        print("=== Shard Size Distribution ===")
        conn.sql("""
            SELECT
                MIN(shard_count) AS min_utt_per_shard,
                MAX(shard_count) AS max_utt_per_shard,
                ROUND(AVG(shard_count), 0) AS mean_utt_per_shard,
                COUNT(*) AS total_shards
            FROM (
                SELECT shard, COUNT(*) AS shard_count FROM meta GROUP BY shard
            )
        """).show()


def query_gender(conn):
    """Gender breakdown (if available)."""
    columns = [col[0] for col in conn.sql("DESCRIBE meta").fetchall()]
    if "gender" not in columns:
        print("=== Gender: column not found ===\n")
        return

    non_empty = conn.sql("SELECT COUNT(*) FROM meta WHERE gender != '' AND gender IS NOT NULL").fetchone()[0]
    if non_empty == 0:
        print("=== Gender: all values empty ===\n")
        return

    print("=== Gender Breakdown ===")
    conn.sql("""
        SELECT
            gender,
            COUNT(*) AS utterances,
            COUNT(DISTINCT speaker) AS speakers,
            ROUND(SUM(duration) / 3600, 2) AS hours,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
        FROM meta
        WHERE gender IS NOT NULL AND gender != ''
        GROUP BY gender
        ORDER BY hours DESC
    """).show()

    print("=== Gender x Duration ===")
    conn.sql("""
        SELECT
            gender,
            ROUND(AVG(duration), 2) AS mean_dur,
            ROUND(MEDIAN(duration), 2) AS median_dur,
            ROUND(MIN(duration), 2) AS min_dur,
            ROUND(MAX(duration), 2) AS max_dur
        FROM meta
        WHERE gender IS NOT NULL AND gender != ''
        GROUP BY gender
    """).show()


def query_language(conn):
    """Language breakdown (if available)."""
    columns = [col[0] for col in conn.sql("DESCRIBE meta").fetchall()]
    if "language" not in columns:
        print("=== Language: column not found ===\n")
        return

    non_empty = conn.sql("SELECT COUNT(*) FROM meta WHERE language != '' AND language IS NOT NULL").fetchone()[0]
    if non_empty == 0:
        print("=== Language: all values empty ===\n")
        return

    print("=== Language Breakdown ===")
    conn.sql("""
        SELECT
            language,
            COUNT(*) AS utterances,
            COUNT(DISTINCT speaker) AS speakers,
            ROUND(SUM(duration) / 3600, 2) AS hours,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
        FROM meta
        GROUP BY language
        ORDER BY hours DESC
    """).show()


def query_crosstab(conn):
    """Cross-tabulation queries."""
    columns = [col[0] for col in conn.sql("DESCRIBE meta").fetchall()]

    print("=== Speaker x Duration Buckets ===")
    conn.sql("""
        SELECT
            speaker,
            SUM(CASE WHEN duration < 5 THEN 1 ELSE 0 END) AS short_lt5s,
            SUM(CASE WHEN duration >= 5 AND duration < 10 THEN 1 ELSE 0 END) AS mid_5_10s,
            SUM(CASE WHEN duration >= 10 AND duration < 20 THEN 1 ELSE 0 END) AS long_10_20s,
            SUM(CASE WHEN duration >= 20 THEN 1 ELSE 0 END) AS vlong_gt20s,
            COUNT(*) AS total
        FROM meta
        GROUP BY speaker
        ORDER BY total DESC
        LIMIT 10
    """).show()

    if "source_dir" in columns:
        print("=== Source Directory x Speaker Count ===")
        conn.sql("""
            SELECT
                source_dir,
                COUNT(DISTINCT speaker) AS speakers,
                COUNT(*) AS utterances,
                ROUND(SUM(duration) / 3600, 2) AS hours,
                ROUND(AVG(LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1), 1) AS mean_words
            FROM meta
            GROUP BY source_dir
            ORDER BY hours DESC
        """).show()


def query_outliers(conn):
    """Find outliers and edge cases."""
    print("=== Duration Outliers (beyond P1/P99) ===")
    bounds = conn.sql("""
        SELECT
            PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY duration) AS p1,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration) AS p99
        FROM meta
    """).fetchone()
    p1, p99 = bounds[0], bounds[1]
    below = conn.sql(f"SELECT COUNT(*) FROM meta WHERE duration < {p1}").fetchone()[0]
    above = conn.sql(f"SELECT COUNT(*) FROM meta WHERE duration > {p99}").fetchone()[0]
    print(f"  P1 = {p1:.2f}s, P99 = {p99:.2f}s")
    print(f"  Below P1:  {below:,} utterances")
    print(f"  Above P99: {above:,} utterances")

    print("=== Speakers with Only 1 Utterance ===")
    single = conn.sql("""
        SELECT COUNT(*) FROM (
            SELECT speaker FROM meta GROUP BY speaker HAVING COUNT(*) = 1
        )
    """).fetchone()[0]
    total_speakers = conn.sql("SELECT COUNT(DISTINCT speaker) FROM meta").fetchone()[0]
    print(f"  {single:,} of {total_speakers:,} speakers ({100.0 * single / max(total_speakers, 1):.1f}%)\n")

    print("=== Words-per-Second Outliers ===")
    print("  Fastest speech (most words/sec):")
    conn.sql("""
        SELECT
            id, speaker, ROUND(duration, 2) AS dur,
            LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1 AS words,
            ROUND((LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1) / NULLIF(duration, 0), 1) AS wps,
            LEFT(text, 50) AS text_preview
        FROM meta
        WHERE text IS NOT NULL AND text != '' AND duration > 0.5
        ORDER BY wps DESC
        LIMIT 5
    """).show()

    print("  Slowest speech (fewest words/sec):")
    conn.sql("""
        SELECT
            id, speaker, ROUND(duration, 2) AS dur,
            LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1 AS words,
            ROUND((LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1) / NULLIF(duration, 0), 1) AS wps,
            LEFT(text, 50) AS text_preview
        FROM meta
        WHERE text IS NOT NULL AND text != '' AND duration > 0.5
        ORDER BY wps ASC
        LIMIT 5
    """).show()


def query_export(conn):
    """Show export query examples (not executed, just printed)."""
    print("=== Export Query Examples ===")
    print("  Copy these to filter and export subsets:\n")

    examples = [
        (
            "Filter by duration (5-15s) to new Parquet",
            "COPY (SELECT * FROM meta WHERE duration BETWEEN 5 AND 15) TO 'filtered_5_15s.parquet' (FORMAT PARQUET)",
        ),
        (
            "Filter by speaker",
            "COPY (SELECT * FROM meta WHERE speaker = '1272') TO 'speaker_1272.parquet' (FORMAT PARQUET)",
        ),
        (
            "Filter by source directory",
            "COPY (SELECT * FROM meta WHERE source_dir LIKE '%librispeech%') TO 'librispeech_only.parquet' (FORMAT PARQUET)",
        ),
        (
            "Remove outliers (P1-P99 range)",
            """COPY (
    WITH bounds AS (
        SELECT
            PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY duration) AS p1,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration) AS p99
        FROM meta
    )
    SELECT m.* FROM meta m, bounds
    WHERE m.duration BETWEEN bounds.p1 AND bounds.p99
) TO 'no_outliers.parquet' (FORMAT PARQUET)""",
        ),
        (
            "Export to CSV",
            "COPY (SELECT id, speaker, duration, text FROM meta) TO 'metadata.csv' (HEADER, DELIMITER ',')",
        ),
        (
            "Export IDs only (for filtering audio later)",
            "COPY (SELECT id FROM meta WHERE duration >= 5) TO 'long_ids.txt' (FORMAT CSV, HEADER FALSE)",
        ),
        (
            "Balanced speaker sampling (max 100 per speaker)",
            """COPY (
    SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY speaker ORDER BY RANDOM()) AS rn
        FROM meta
    ) WHERE rn <= 100
) TO 'balanced_100_per_speaker.parquet' (FORMAT PARQUET)""",
        ),
        (
            "Deduplicate by ID (keep first)",
            """COPY (
    SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY source_dir) AS rn
        FROM meta
    ) WHERE rn = 1
) TO 'deduped.parquet' (FORMAT PARQUET)""",
        ),
    ]

    for title, sql in examples:
        print(f"  -- {title}")
        print(f"  {sql};\n")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

QUERIES = {
    "overview": ("Dataset overview", query_overview),
    "duration": ("Duration statistics and distribution", query_duration),
    "speaker": ("Speaker-level analysis", query_speaker),
    "text": ("Text and transcript analysis", query_text),
    "quality": ("Data quality checks", query_quality),
    "source": ("Per source directory / shard analysis", query_source),
    "gender": ("Gender breakdown", query_gender),
    "language": ("Language breakdown", query_language),
    "crosstab": ("Cross-tabulation queries", query_crosstab),
    "outliers": ("Outlier detection", query_outliers),
    "export": ("Export query examples", query_export),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Query and analyze Parquet metadata using DuckDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--parquet",
        nargs="+",
        required=True,
        help="Path(s) to metadata parquet file(s).",
    )
    parser.add_argument(
        "--query",
        nargs="*",
        default=None,
        help="Query sections to run (default: all). Use 'list' to see available.",
    )
    args = parser.parse_args()

    # List available queries
    if args.query and "list" in args.query:
        print("Available query sections:")
        for name, (desc, _) in QUERIES.items():
            print(f"  {name:>12s} - {desc}")
        return

    conn = duckdb.connect()

    # Register parquet file(s) as a view
    paths = args.parquet
    if len(paths) == 1:
        src = f"'{paths[0]}'"
    else:
        src = f"read_parquet({paths})"
    conn.execute(f"CREATE VIEW meta AS SELECT * FROM {src}")

    # Determine which queries to run
    if args.query:
        selected = args.query
    else:
        selected = list(QUERIES.keys())

    for name in selected:
        if name not in QUERIES:
            print(f"Unknown query: {name}. Use --query list to see available.\n")
            continue
        desc, fn = QUERIES[name]
        print(f"\n{'='*70}")
        print(f"  {name.upper()}: {desc}")
        print(f"{'='*70}\n")
        fn(conn)

    conn.close()


if __name__ == "__main__":
    main()
