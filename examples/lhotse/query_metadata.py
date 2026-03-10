"""Query and analyze the Parquet metadata table using DuckDB.

Scales to 100M+ rows without loading everything into memory.

Usage:
  python examples/lhotse/query_metadata.py --parquet metadata.parquet
  python examples/lhotse/query_metadata.py --parquet /path/to/*.parquet
"""

import argparse
import duckdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        nargs="+",
        required=True,
        help="Path(s) to metadata parquet file(s). Supports glob patterns.",
    )
    args = parser.parse_args()

    conn = duckdb.connect()

    # Register parquet file(s) as a view
    paths = args.parquet
    if len(paths) == 1:
        src = f"'{paths[0]}'"
    else:
        src = f"read_parquet({paths})"

    conn.execute(f"CREATE VIEW meta AS SELECT * FROM {src}")

    # --- Overview ---
    row_count = conn.sql("SELECT COUNT(*) FROM meta").fetchone()[0]
    columns = [col[0] for col in conn.sql("DESCRIBE meta").fetchall()]
    print(f"Total rows: {row_count:,}")
    print(f"Columns: {columns}\n")

    # --- Duration stats ---
    print("=== Duration Stats ===")
    stats = conn.sql("""
        SELECT
            SUM(duration) / 3600 AS total_hours,
            AVG(duration) AS mean_s,
            MEDIAN(duration) AS median_s,
            MIN(duration) AS min_s,
            MAX(duration) AS max_s,
            STDDEV(duration) AS std_s
        FROM meta
    """).fetchone()
    print(f"  Total hours:  {stats[0]:,.2f}")
    print(f"  Mean:         {stats[1]:.2f}s")
    print(f"  Median:       {stats[2]:.2f}s")
    print(f"  Min:          {stats[3]:.2f}s")
    print(f"  Max:          {stats[4]:.2f}s")
    print(f"  Std:          {stats[5]:.2f}s")

    # --- Per speaker (top 10) ---
    print("\n=== Per Speaker (top 10 by hours) ===")
    spk = conn.sql("""
        SELECT
            speaker,
            COUNT(*) AS count,
            ROUND(SUM(duration) / 3600, 2) AS hours,
            ROUND(AVG(duration), 2) AS mean_dur
        FROM meta
        GROUP BY speaker
        ORDER BY hours DESC
        LIMIT 10
    """)
    spk.show()

    # --- Per source directory ---
    has_source_dir = "source_dir" in columns
    if has_source_dir:
        print("=== Per Source Directory ===")
        conn.sql("""
            SELECT
                source_dir,
                COUNT(*) AS count,
                ROUND(SUM(duration) / 3600, 2) AS hours
            FROM meta
            GROUP BY source_dir
            ORDER BY hours DESC
        """).show()

    # --- Duration distribution ---
    print("=== Duration Distribution ===")
    dist = conn.sql("""
        SELECT
            CASE
                WHEN duration < 2 THEN '0-2s'
                WHEN duration < 5 THEN '2-5s'
                WHEN duration < 10 THEN '5-10s'
                WHEN duration < 15 THEN '10-15s'
                WHEN duration < 20 THEN '15-20s'
                WHEN duration < 30 THEN '20-30s'
                WHEN duration < 60 THEN '30-60s'
                ELSE '60s+'
            END AS bin,
            COUNT(*) AS count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
        FROM meta
        GROUP BY bin
        ORDER BY
            CASE bin
                WHEN '0-2s' THEN 1
                WHEN '2-5s' THEN 2
                WHEN '5-10s' THEN 3
                WHEN '10-15s' THEN 4
                WHEN '15-20s' THEN 5
                WHEN '20-30s' THEN 6
                WHEN '30-60s' THEN 7
                WHEN '60s+' THEN 8
            END
    """).fetchall()
    for bin_label, count, pct in dist:
        bar = "#" * int(pct / 2)
        print(f"  {bin_label:>7s}: {count:>10,} ({pct:5.1f}%) {bar}")

    # --- Text stats ---
    print("\n=== Text Stats ===")
    text_stats = conn.sql("""
        SELECT
            ROUND(AVG(LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1), 1) AS mean_words,
            MAX(LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1) AS max_words,
            SUM(CASE WHEN text = '' OR text IS NULL THEN 1 ELSE 0 END) AS empty_count
        FROM meta
    """).fetchone()
    print(f"  Mean words/utt:   {text_stats[0]}")
    print(f"  Max words/utt:    {text_stats[1]}")
    print(f"  Empty text count: {text_stats[2]:,}")

    # --- Sampling rates ---
    print("\n=== Sampling Rates ===")
    conn.sql("""
        SELECT sampling_rate, COUNT(*) AS count
        FROM meta
        GROUP BY sampling_rate
        ORDER BY count DESC
    """).show()

    # --- Filter examples ---
    print("=== Filter Examples ===")
    long_stats = conn.sql("""
        SELECT COUNT(*) AS count, ROUND(SUM(duration) / 3600, 2) AS hours
        FROM meta WHERE duration >= 10
    """).fetchone()
    print(f"  Duration >= 10s: {long_stats[0]:,} utterances ({long_stats[1]}h)")

    short_count = conn.sql("SELECT COUNT(*) FROM meta WHERE duration < 2").fetchone()[0]
    print(f"  Duration < 2s:   {short_count:,} utterances")

    top_spk = conn.sql("""
        SELECT speaker, COUNT(*) AS count, ROUND(SUM(duration) / 3600, 2) AS hours
        FROM meta
        GROUP BY speaker
        ORDER BY hours DESC
        LIMIT 1
    """).fetchone()
    if top_spk:
        print(f"  Speaker '{top_spk[0]}': {top_spk[1]:,} utterances ({top_spk[2]}h)")

    conn.close()


if __name__ == "__main__":
    main()
