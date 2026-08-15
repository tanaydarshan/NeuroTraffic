"""
NeuroTraffic — Spark Structured Streaming Pipeline (HDFS-Integrated)
======================================================================
Owner: Sasmitha S (CB.AI.U4AID24051)
Course: Big Data Analytics (23AID302)

Architecture (LOCAL mode):
    streaming/input/*.csv  →  Spark  →  streaming/output/latest.csv
                                              ↑ (read by Streamlit dashboard)

Architecture (HDFS mode):
    hdfs:///data/streaming/input/*.csv  →  Spark  →  hdfs:///data/streaming/output/
                                                            ↑ (read via WebHDFS by dashboard)

How it works:
    1. Watches the configured input directory for new CSV batch files
    2. Reads each new batch through Spark Structured Streaming
    3. Applies z-score anomaly detection (matches disruption_detector.py logic)
    4. Writes anomaly_score, is_disruption, predicted_surge_pct to output
    5. Posts latest batch summary to HDFS /data/streaming/output/latest.csv

Running modes:

    LOCAL (no Spark cluster):
        spark-submit --master local[*] streaming_pipeline.py

    HDFS mode (on the Docker cluster):
        docker exec spark-master /spark/bin/spark-submit \\
            --master spark://spark-master:7077 \\
            --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \\
            --driver-memory 2g \\
            /scripts/streaming_pipeline.py --hdfs

    HDFS mode (from host with local Spark, connecting to cluster):
        spark-submit --master local[*] \\
            --conf spark.hadoop.fs.defaultFS=hdfs://localhost:9000 \\
            streaming_pipeline.py --hdfs

Prerequisites:
    pip install pyspark pyarrow pandas numpy requests
"""

import os
import sys
import argparse

# ── Path setup ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import config
import hdfs_helper

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, IntegerType, DoubleType, StringType, TimestampType
    )
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    print("[WARNING] PySpark not available. Run with spark-submit, not python directly.")


# ══════════════════════════════════════════════════════════
# Schema for incoming batch CSV files
# ══════════════════════════════════════════════════════════

INPUT_SCHEMA = StructType([
    StructField("zone_id",           IntegerType(), True),
    StructField("date",              StringType(),  True),
    StructField("hourly_timestamp",  StringType(),  True),
    StructField("taxi_pickups",      IntegerType(), True),
    StructField("taxi_dropoffs",     IntegerType(), True),
    StructField("subway_ridership",  IntegerType(), True),
    StructField("bike_starts",       IntegerType(), True),
    StructField("bike_ends",         IntegerType(), True),
    StructField("temperature_c",     DoubleType(),  True),
    StructField("humidity_pct",      DoubleType(),  True),
    StructField("precipitation_mm",  DoubleType(),  True),
    StructField("wind_speed_kmh",    DoubleType(),  True),
    StructField("is_rain",           IntegerType(), True),
    StructField("is_snow",           IntegerType(), True),
    StructField("is_extreme_cold",   IntegerType(), True),
    StructField("is_extreme_heat",   IntegerType(), True),
    StructField("complaint_count",   IntegerType(), True),
    StructField("event_count",       IntegerType(), True),
    StructField("page_rank",         DoubleType(),  True),
    StructField("community_id",      IntegerType(), True),
    StructField("in_degree",         IntegerType(), True),
    StructField("out_degree",        IntegerType(), True),
    StructField("hour_of_day",       IntegerType(), True),
    StructField("is_rush_hour",      IntegerType(), True),
    StructField("is_weekend",        IntegerType(), True),
    StructField("z_subway",          DoubleType(),  True),
    StructField("z_pickups",         DoubleType(),  True),
    StructField("z_bikes",           DoubleType(),  True),
    StructField("is_anomaly",        IntegerType(), True),
    StructField("prediction",        IntegerType(), True),
]) if PYSPARK_AVAILABLE else None


# ══════════════════════════════════════════════════════════
# UDF: compute anomaly_score from z-scores
# ══════════════════════════════════════════════════════════

def compute_anomaly_score(z_sub, z_pick, z_bikes, is_anomaly):
    """
    Convert z-scores to a normalized anomaly score [0, 1].
    High z_subway magnitude (drop) + high z_pickups/z_bikes (surge) → high score.
    Confirmed anomalies from ML pipeline are boosted.
    """
    if z_sub is None or z_pick is None or z_bikes is None:
        return 0.0
    base = (abs(z_sub) / 5.0 + abs(z_pick) / 6.0 + abs(z_bikes) / 7.0) / 3.0 * 1.5
    score = min(max(base, 0.0), 1.0)
    if is_anomaly == 1:
        score = min(score * 1.5 + 0.3, 1.0)
    return round(score, 4)


# ══════════════════════════════════════════════════════════
# Spark session builder
# ══════════════════════════════════════════════════════════

def create_spark(use_hdfs=False, hdfs_rpc_url=None):
    """Build SparkSession, optionally configured for HDFS."""
    builder = (
        SparkSession.builder
        .appName("NeuroTraffic_StreamingPipeline")
        .config("spark.sql.streaming.schemaInference", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
    )

    if use_hdfs:
        rpc = hdfs_rpc_url or hdfs_helper.HDFS_RPC_URL
        builder = (
            builder
            .config("spark.hadoop.fs.defaultFS", rpc)
            # Allow WebHDFS redirects to work even from host machine
            .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        )
        print(f"  Spark ← HDFS: {rpc}")

    return builder.getOrCreate()


# ══════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════

def build_pipeline(spark, use_hdfs=False):
    """Build and start the Spark Structured Streaming pipeline."""

    # ── Register UDF ──
    anomaly_score_udf = F.udf(compute_anomaly_score, DoubleType())

    # ── Choose input/output paths ──
    if use_hdfs:
        input_dir  = f"{hdfs_helper.HDFS_RPC_URL}{config.HDFS_STREAMING_INPUT_DIR}"
        output_dir = f"{hdfs_helper.HDFS_RPC_URL}{config.HDFS_STREAMING_OUTPUT_DIR}/history"
        checkpoint = f"{hdfs_helper.HDFS_RPC_URL}{config.HDFS_STREAMING_OUTPUT_DIR}/_checkpoints"
    else:
        input_dir  = config.STREAMING_INPUT_DIR
        output_dir = os.path.join(config.STREAMING_OUTPUT_DIR, "history")
        checkpoint = os.path.join(config.STREAMING_OUTPUT_DIR, "_checkpoints")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint, exist_ok=True)

    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_dir}")
    print(f"  Checkpoint: {checkpoint}")

    # ── 1. Read stream ──
    stream_df = (
        spark.readStream
        .option("header", "true")
        .option("maxFilesPerTrigger", "1")
        .schema(INPUT_SCHEMA)
        .csv(input_dir)
    )

    # ── 2. Compute anomaly scores ──
    scored_df = stream_df.withColumn(
        "anomaly_score",
        anomaly_score_udf(
            F.col("z_subway"), F.col("z_pickups"), F.col("z_bikes"), F.col("is_anomaly")
        )
    )

    # ── 3. Flag disruptions ──
    scored_df = scored_df.withColumn(
        "is_disruption",
        F.when(F.col("anomaly_score") >= 0.80, 1).otherwise(0)
    )

    # ── 4. Predict surge ──
    scored_df = scored_df.withColumn(
        "predicted_surge_pct",
        F.when(F.col("is_disruption") == 1,
               (F.col("anomaly_score") * 350).cast(IntegerType()))
        .otherwise(0)
    )

    # ── 5. Time to peak (proxy: high-PageRank zones peak sooner) ──
    scored_df = scored_df.withColumn(
        "time_to_peak_min",
        F.when(F.col("is_disruption") == 1,
               (10 - F.col("page_rank") * 1000).cast(IntegerType()))
        .otherwise(0)
    )

    # ── 6. Cross-modal ratios ──
    scored_df = scored_df.withColumn(
        "taxi_to_subway_ratio",
        F.when(F.col("subway_ridership") > 0,
               F.col("taxi_pickups").cast(DoubleType()) / F.col("subway_ridership"))
        .otherwise(0.0)
    ).withColumn(
        "bike_to_subway_ratio",
        F.when(F.col("subway_ridership") > 0,
               F.col("bike_starts").cast(DoubleType()) / F.col("subway_ridership"))
        .otherwise(0.0)
    )

    # ── 7. Processing timestamp ──
    scored_df = scored_df.withColumn("processed_at", F.current_timestamp())

    # ── 8. Select output columns ──
    output_cols = [
        "zone_id", "hourly_timestamp", "anomaly_score", "is_disruption",
        "predicted_surge_pct", "time_to_peak_min",
        "z_subway", "z_pickups", "z_bikes",
        "taxi_to_subway_ratio", "bike_to_subway_ratio",
        "taxi_pickups", "taxi_dropoffs", "subway_ridership", "bike_starts",
        "temperature_c", "is_rain", "is_snow",
        "complaint_count", "event_count",
        "page_rank", "community_id", "in_degree", "out_degree",
        "is_rush_hour", "is_weekend", "hour_of_day",
        "prediction", "processed_at",
    ]
    output_df = scored_df.select(output_cols)

    # ── 9. Foreach-batch writer: write CSV + update latest.csv ──
    def foreach_batch(batch_df, batch_id):
        if batch_df.rdd.isEmpty():
            return

        # Write to the history partition
        (
            batch_df.write
            .mode("append")
            .option("header", "true")
            .csv(output_dir)
        )

        # Convert to pandas and update latest.csv
        import pandas as pd
        pdf = batch_df.toPandas()

        if use_hdfs:
            # Write latest to HDFS via WebHDFS
            ok = hdfs_helper.write_hdfs_csv(pdf, config.HDFS_LATEST_CSV)
            if ok:
                print(f"  [batch {batch_id}] Wrote {len(pdf)} rows → HDFS latest.csv")
            else:
                print(f"  [batch {batch_id}] HDFS write failed — falling back to local")
                pdf.to_csv(config.LATEST_CSV, index=False)
        else:
            pdf.to_csv(config.LATEST_CSV, index=False)
            print(f"  [batch {batch_id}] Wrote {len(pdf)} rows → local latest.csv")

    # ── 10. Start query ──
    query = (
        output_df.writeStream
        .outputMode("append")
        .foreachBatch(foreach_batch)
        .option("checkpointLocation", checkpoint)
        .trigger(processingTime="5 seconds")
        .start()
    )

    return query


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NeuroTraffic Spark Streaming Pipeline")
    hdfs_group = parser.add_mutually_exclusive_group()
    hdfs_group.add_argument("--hdfs",    action="store_true",  default=None,
                            help="Read/write via HDFS (overrides config.USE_HDFS)")
    hdfs_group.add_argument("--no-hdfs", action="store_true",  dest="no_hdfs", default=False,
                            help="Force local mode (overrides config.USE_HDFS)")
    parser.add_argument("--hdfs-rpc",    type=str, default=None,
                        help="Override HDFS RPC URL (e.g. hdfs://namenode:9000)")
    args = parser.parse_args()

    if not PYSPARK_AVAILABLE:
        print("[ERROR] PySpark is not installed. Install with: pip install pyspark")
        sys.exit(1)

    if args.no_hdfs:
        use_hdfs = False
    elif args.hdfs:
        use_hdfs = True
    else:
        use_hdfs = config.USE_HDFS

    print("\n" + "=" * 60)
    print("  NeuroTraffic — Spark Structured Streaming Pipeline")
    print("=" * 60)
    print(f"\n  HDFS mode: {'ON' if use_hdfs else 'OFF'}")

    spark = create_spark(use_hdfs=use_hdfs, hdfs_rpc_url=args.hdfs_rpc)
    spark.sparkContext.setLogLevel("ERROR")

    print("\n[1/2] Building streaming pipeline ...")
    query = build_pipeline(spark, use_hdfs=use_hdfs)

    print("\n[2/2] Pipeline running — waiting for new batches ...")
    if use_hdfs:
        print(f"      Push CSV files to HDFS {config.HDFS_STREAMING_INPUT_DIR}/")
        print(f"      Output → HDFS {config.HDFS_STREAMING_OUTPUT_DIR}/")
    else:
        print(f"      Push CSV files to {config.STREAMING_INPUT_DIR}/")
        print(f"      Output → {config.STREAMING_OUTPUT_DIR}/")
    print("      Use replay_simulator.py to generate those files.")
    print("      Press Ctrl+C to stop.\n")

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\n\n  Stopping streaming pipeline ...")
        query.stop()
        spark.stop()
        print("  Stopped. Goodbye!")


if __name__ == "__main__":
    main()
