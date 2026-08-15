"""
NeuroTraffic — Data Replay Simulator (HDFS-Aware)
===================================================
Owner: Sasmitha S (CB.AI.U4AID24051)
Course: Big Data Analytics (23AID302)

Replays historical data as a real-time stream:
  - Reads historical_predictions.parquet (local OR from HDFS)
  - Pushes one day's data (all 263 zones) every N seconds to:
      * streaming/input/        (for the Spark pipeline to consume)
      * streaming/output/latest.csv  (for the Streamlit dashboard)
      * HDFS /data/streaming/output/latest.csv  (when HDFS mode is enabled)

Run this in a separate terminal while the Streamlit app is running:
    python streaming/replay_simulator.py

Options:
    --speed N           : push one batch every N seconds (default: 5)
    --loop              : replay infinitely when data runs out
    --from YYYY-MM-DD   : start from a specific date (default: 2023-01-01)
    --hdfs              : also write batches to HDFS (overrides config.USE_HDFS)
    --no-hdfs           : disable HDFS writes even if config.USE_HDFS=True
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Paths ──
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, DASHBOARD_DIR)
import config
import hdfs_helper


# ══════════════════════════════════════════════════════════
# Enrichment
# ══════════════════════════════════════════════════════════

def setup_dirs():
    os.makedirs(config.STREAMING_INPUT_DIR,  exist_ok=True)
    os.makedirs(config.STREAMING_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.STREAMING_OUTPUT_DIR, "history"), exist_ok=True)


def enrich_batch(batch_df, nodes_df, date_str):
    """Add anomaly scores and cascade predictions to a batch."""
    batch_df = batch_df.copy()

    # ── Anomaly score from z-scores ──
    batch_df["anomaly_score"] = np.clip(
        (batch_df["z_subway"].abs() / 5 + batch_df["z_pickups"].abs() / 6 +
         batch_df["z_bikes"].abs() / 7) / 3 * 1.5,
        0, 1
    ).round(4)

    # Boost confirmed anomalies
    mask = batch_df["is_anomaly"] == 1
    batch_df.loc[mask, "anomaly_score"] = np.clip(
        batch_df.loc[mask, "anomaly_score"] * 1.5 + 0.3, 0.8, 1.0
    )

    # ── Disruption flag ──
    batch_df["is_disruption"] = (batch_df["anomaly_score"] >= 0.80).astype(int)

    # ── Predicted surge % ──
    batch_df["predicted_surge_pct"] = np.where(
        batch_df["is_disruption"] == 1,
        (batch_df["anomaly_score"] * 350).astype(int),
        0
    )

    # ── Time to peak ──
    batch_df["time_to_peak_min"] = np.where(
        batch_df["is_disruption"] == 1,
        np.random.randint(3, 20, len(batch_df)),
        0
    )

    # ── Cascade zone IDs ──
    def get_cascade(row):
        if row["is_disruption"] == 1:
            nearby = list(range(row["zone_id"], min(row["zone_id"] + 5, 264)))
            return ",".join(str(z) for z in nearby if z != row["zone_id"])
        return ""
    batch_df["cascade_zone_ids"] = batch_df.apply(get_cascade, axis=1)

    # ── Timestamp ──
    batch_df["timestamp"] = f"{date_str} 08:00:00"

    # ── Merge node info for lat/lon ──
    if nodes_df is not None:
        batch_df = batch_df.merge(
            nodes_df[["zone_id", "lat", "lon", "name", "primary_line"]],
            on="zone_id", how="left"
        )

    # ── Affected line ──
    batch_df["affected_line"] = np.where(
        batch_df["is_disruption"] == 1,
        batch_df.get("primary_line", pd.Series(["-"] * len(batch_df))).fillna("-"),
        "-"
    )

    return batch_df


# ══════════════════════════════════════════════════════════
# HDFS write helpers
# ══════════════════════════════════════════════════════════

def write_csv_to_hdfs(df, hdfs_path):
    """Write a DataFrame as CSV to HDFS; silently log errors."""
    try:
        ok = hdfs_helper.write_hdfs_csv(df, hdfs_path)
        if not ok:
            print(f"    [HDFS WARN] Failed to write {hdfs_path}")
    except Exception as e:
        print(f"    [HDFS ERR] {e}")


def ensure_hdfs_dirs():
    """Create required HDFS directories if they don't exist."""
    for d in [
        "/data/streaming",
        "/data/streaming/input",
        "/data/streaming/output",
        "/data/streaming/output/history",
    ]:
        hdfs_helper.make_hdfs_directory(d)


# ══════════════════════════════════════════════════════════
# Data loading (supports HDFS for historical data)
# ══════════════════════════════════════════════════════════

def load_historical(use_hdfs):
    """Load historical_predictions.parquet from HDFS or locally."""
    if use_hdfs and hdfs_helper.is_hdfs_available():
        print("  Loading historical data from HDFS ...")
        df = hdfs_helper.load_dataframe(
            config.HISTORICAL_PARQUET,
            config.HDFS_HISTORICAL_PARQUET,
            use_hdfs=True
        )
        if df is not None:
            return df
        print("  [WARN] HDFS load failed — falling back to local file.")

    if not os.path.exists(config.HISTORICAL_PARQUET):
        print(f"  ERROR: {config.HISTORICAL_PARQUET} not found.")
        print("  Run: python data/generate_sample_data.py")
        sys.exit(1)

    return pd.read_parquet(config.HISTORICAL_PARQUET)


def load_nodes(use_hdfs):
    """Load graph_nodes.csv from HDFS or locally."""
    if use_hdfs and hdfs_helper.is_hdfs_available():
        df = hdfs_helper.load_dataframe(
            config.GRAPH_NODES_CSV,
            config.HDFS_GRAPH_NODES_CSV,
            use_hdfs=True
        )
        if df is not None:
            return df

    if os.path.exists(config.GRAPH_NODES_CSV):
        return pd.read_csv(config.GRAPH_NODES_CSV)
    return None


# ══════════════════════════════════════════════════════════
# Main simulator
# ══════════════════════════════════════════════════════════

def run_simulator(speed_seconds=5, loop=False, start_from="2023-01-01", use_hdfs=None):
    """
    Parameters
    ----------
    use_hdfs : bool or None
        If None, uses config.USE_HDFS.
        If True / False, overrides the config value.
    """
    if use_hdfs is None:
        use_hdfs = config.USE_HDFS

    setup_dirs()

    hdfs_ok = False
    if use_hdfs:
        hdfs_ok = hdfs_helper.is_hdfs_available()
        if hdfs_ok:
            ensure_hdfs_dirs()
        else:
            print("  [WARN] HDFS requested but not reachable — HDFS writes disabled.")
            use_hdfs = False

    print("\n" + "=" * 60)
    print("  NeuroTraffic — Data Replay Simulator")
    print(f"  Speed:      1 batch every {speed_seconds} seconds")
    print(f"  Loop mode:  {loop}")
    print(f"  Start date: {start_from}")
    print(f"  HDFS mode:  {'ON (' + hdfs_helper.WEBHDFS_BASE_URL + ')' if use_hdfs else 'OFF (local only)'}")
    print("=" * 60)

    # ── Load data ──
    print("\n[1/2] Loading data ...")
    hist_df = load_historical(use_hdfs)
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    hist_df = hist_df[hist_df["date"] >= pd.Timestamp(start_from)].sort_values(["date", "zone_id"])
    print(f"  Loaded {len(hist_df):,} rows · {hist_df['date'].nunique()} dates · {hist_df['zone_id'].nunique()} zones")

    nodes_df = load_nodes(use_hdfs)
    if nodes_df is not None:
        print(f"  Loaded {len(nodes_df)} graph nodes")

    dates = sorted(hist_df["date"].unique())
    print(f"\n[2/2] Starting replay — {len(dates)} batches to send ...")
    print("      (each batch = all zones for one day)\n")

    latest_cols = [c for c in [
        "zone_id", "timestamp", "anomaly_score", "is_disruption", "predicted_surge_pct",
        "affected_line", "cascade_zone_ids", "time_to_peak_min",
        "z_subway", "z_pickups", "z_bikes", "taxi_pickups", "subway_ridership", "bike_starts",
    ]]
    raw_cols_base = [
        "zone_id", "date", "hourly_timestamp", "taxi_pickups", "taxi_dropoffs",
        "subway_ridership", "bike_starts", "bike_ends", "temperature_c",
        "humidity_pct", "precipitation_mm", "wind_speed_kmh", "is_rain", "is_snow",
        "is_extreme_cold", "is_extreme_heat", "complaint_count", "event_count",
        "page_rank", "community_id", "in_degree", "out_degree", "hour_of_day",
        "is_rush_hour", "is_weekend", "z_subway", "z_pickups", "z_bikes",
        "is_anomaly", "prediction",
    ]

    batch_num = 0
    while True:
        for dt in dates:
            batch_num += 1
            date_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
            batch = hist_df[hist_df["date"] == dt].copy()

            # Enrich
            enriched = enrich_batch(batch, nodes_df, date_str)

            # ── Write raw batch → streaming/input/ (for Spark pipeline) ──
            raw_cols = [c for c in raw_cols_base if c in batch.columns]
            input_path = os.path.join(config.STREAMING_INPUT_DIR, f"batch_{date_str}.csv")
            batch[raw_cols].to_csv(input_path, index=False)

            # ── Build latest DataFrame ──
            out_cols = [c for c in latest_cols if c in enriched.columns]
            latest_df = enriched[out_cols]

            # ── Write latest.csv → local streaming/output/ ──
            latest_df.to_csv(config.LATEST_CSV, index=False)

            # ── Archive locally ──
            archive_path = os.path.join(
                config.STREAMING_OUTPUT_DIR, "history", f"predictions_{date_str}.csv"
            )
            latest_df.to_csv(archive_path, index=False)

            # ── Write to HDFS ──
            if use_hdfs:
                # latest.csv → HDFS (what the Streamlit dashboard reads)
                write_csv_to_hdfs(latest_df, config.HDFS_LATEST_CSV)

                # raw batch → HDFS streaming input (for Spark pipeline on cluster)
                hdfs_input_path = f"{config.HDFS_STREAMING_INPUT_DIR}/batch_{date_str}.csv"
                write_csv_to_hdfs(batch[raw_cols], hdfs_input_path)

                # archive → HDFS
                hdfs_archive = f"{config.HDFS_STREAMING_OUTPUT_DIR}/history/predictions_{date_str}.csv"
                write_csv_to_hdfs(latest_df, hdfs_archive)

            n_disrupt  = int((enriched["is_disruption"] == 1).sum()) if "is_disruption" in enriched.columns else 0
            peak_score = enriched["anomaly_score"].max() if "anomaly_score" in enriched.columns else 0
            flag = "[CRIT]" if n_disrupt > 0 else "[OK]  "
            hdfs_tag = " [HDFS]" if use_hdfs else ""

            print(f"  {flag} Batch {batch_num:03d} | {date_str} | {len(batch)} zones | "
                  f"{n_disrupt} disruptions | peak={peak_score:.3f}{hdfs_tag}")

            time.sleep(speed_seconds)

        if not loop:
            print("\n  [SUCCESS] Replay complete — all dates sent.")
            print("  Use --loop to replay infinitely.")
            break
        else:
            print("\n  [LOOP] Looping — replaying from the beginning ...")
            batch_num = 0


# ══════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroTraffic Data Replay Simulator")
    parser.add_argument("--speed",   type=int, default=5,           help="Seconds between batches (default: 5)")
    parser.add_argument("--loop",    action="store_true",           help="Loop replay infinitely")
    parser.add_argument("--from",    dest="start_from", type=str,   default="2023-01-01",
                        help="Start date (YYYY-MM-DD)")
    # HDFS override flags
    hdfs_group = parser.add_mutually_exclusive_group()
    hdfs_group.add_argument("--hdfs",    action="store_true",  default=None,
                            help="Enable HDFS writes (overrides config.USE_HDFS)")
    hdfs_group.add_argument("--no-hdfs", action="store_true",  dest="no_hdfs", default=False,
                            help="Disable HDFS writes (overrides config.USE_HDFS)")
    args = parser.parse_args()

    # Determine effective use_hdfs
    if args.no_hdfs:
        use_hdfs = False
    elif args.hdfs:
        use_hdfs = True
    else:
        use_hdfs = None   # use config.USE_HDFS

    run_simulator(args.speed, args.loop, args.start_from, use_hdfs=use_hdfs)
