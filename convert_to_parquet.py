"""
Convert merged_wisdm.csv → merged_wisdm.parquet (one-time operation).

Run once:
    python convert_to_parquet.py --csv merged_wisdm.csv

This will:
  1. Read the CSV (slow, but only once)
  2. Clean column names
  3. Cast numeric columns to proper dtypes (shrinks memory)
  4. Write a compressed Parquet file

Subsequent loads will be 5-15× faster and ~70% smaller on disk.

Requirements:
    pip install pandas pyarrow
"""

import argparse
import os
import sys
import time

import pandas as pd


def convert(csv_path: str, output_path: str = None) -> str:
    if output_path is None:
        output_path = csv_path.rsplit(".", 1)[0] + ".parquet"

    print(f"[1/4] Reading CSV: {csv_path}")
    print(f"       File size: {os.path.getsize(csv_path) / 1e9:.2f} GB")
    t0 = time.time()

    # Read with explicit low-memory settings
    df = pd.read_csv(
        csv_path,
        engine="c",              # C parser is fastest
        low_memory=False,        # avoids mixed-type warnings
        na_values=["", " ", "NA", "NaN", "null"],
    )
    t_read = time.time() - t0
    print(f"       Read in {t_read:.1f}s — {len(df):,} rows × {len(df.columns)} cols")

    # ── Clean column names ────────────────────────────────────────────────
    print("[2/4] Cleaning columns ...")
    df.columns = (df.columns.str.strip().str.lower()
                   .str.replace(";", "", regex=False))

    # ── Optimise dtypes ───────────────────────────────────────────────────
    print("[3/4] Optimising dtypes ...")

    # Numeric axes → float32 (halves memory vs float64)
    for col in ("x", "y", "z"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # Timestamp → int64 if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    # Categorical columns → category dtype (huge savings for repeated strings)
    for col in ("device", "sensor", "activity_code", "activity_label",
                "subject_id", "user"):
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Subject/user IDs — keep as-is but ensure consistent type
    for col in ("subject_id", "user"):
        if col in df.columns and df[col].dtype.name != "category":
            df[col] = df[col].astype("category")

    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"       In-memory: {mem_mb:.0f} MB")

    # ── Write Parquet ─────────────────────────────────────────────────────
    print(f"[4/4] Writing Parquet: {output_path}")
    t1 = time.time()
    df.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",    # fast decompression, good ratio
        index=False,
    )
    t_write = time.time() - t1

    out_size = os.path.getsize(output_path) / 1e6
    csv_size = os.path.getsize(csv_path) / 1e6
    ratio = (1 - out_size / csv_size) * 100

    print(f"\n  ✓ Done!")
    print(f"    CSV:     {csv_size:,.0f} MB")
    print(f"    Parquet: {out_size:,.0f} MB  ({ratio:.0f}% smaller)")
    print(f"    Read:    {t_read:.1f}s")
    print(f"    Write:   {t_write:.1f}s")
    print(f"\n  Now use:  --csv {output_path}")

    return output_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert WISDM CSV → Parquet")
    p.add_argument("--csv", required=True, help="Path to merged_wisdm.csv")
    p.add_argument("--output", default=None, help="Output .parquet path")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found")
        sys.exit(1)

    convert(args.csv, args.output)
