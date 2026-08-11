"""
NeuroTraffic — ML Pipeline: Subway Disruption Detection
========================================================
Owner: Harsada
Course: Big Data Analytics (23AID302)

This pipeline detects subway disruptions using a cross-modal behavioral
signature: when a subway line fails, ridership drops while taxi pickups
and bike checkouts surge in the same zone.

Two models:
  1. Isolation Forest — unsupervised anomaly detection on the combined
     zone x hour table (flags unusual patterns without labels)
  2. GBT Classifier — supervised disruption classification using graph
     features + transport metrics (requires labeled disruptions)

Data sources:
  - Navaneeth's combined table: /data/neurotraffic/combined_v3/
    (6.9M rows, 38 columns: taxi, subway, weather, 311, events, bike per zone per hour)
  - Tanay's graph features: real_data/graph_features/features.parquet
    (3,219 nodes with PageRank, community_id, in_degree, out_degree)

Usage (local mode):
  spark-submit --master local[*] --driver-memory 4g disruption_detector.py

Usage (Docker cluster):
  docker exec spark-master /spark/bin/spark-submit \
    --master local[2] --driver-memory 2g \
    /scripts/disruption_detector.py
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import sys
import os

# ── Configuration ──
# Adjust these paths based on your environment:
#   - HDFS paths for Docker cluster
#   - Local paths for testing on your laptop

# Docker/HDFS paths (when running inside the cluster):
HDFS_COMBINED = "hdfs://namenode:9000/data/neurotraffic/combined_v3"
HDFS_GRAPH_FEATURES = "hdfs://namenode:9000/data/neurotraffic/graph_features"
HDFS_OUTPUT = "hdfs://namenode:9000/data/neurotraffic/predictions"

# Local paths (when testing on laptop):
LOCAL_COMBINED = None  # Set to path of combined_v3 parquet if testing locally
LOCAL_GRAPH_FEATURES = os.path.join(os.path.dirname(__file__),
                                     "..", "real_data", "graph_features", "features.parquet")
LOCAL_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "real_data", "predictions")


def create_spark():
    return (
        SparkSession.builder
        .appName("NeuroTraffic_DisruptionDetector")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )


# =========================================================================
# STEP 1: Load and merge data sources
# =========================================================================
def load_data(spark, use_hdfs=True):
    """Load the combined table and graph features, merge them."""

    if use_hdfs:
        combined_path = HDFS_COMBINED
        graph_path = HDFS_GRAPH_FEATURES
    else:
        if LOCAL_COMBINED is None:
            print("[ERROR] Set LOCAL_COMBINED path to Navaneeth's combined_v3 parquet directory")
            print("        This is only available on HDFS in the Docker cluster.")
            print("        Start Docker and use --hdfs flag, or copy the data locally.")
            sys.exit(1)
        combined_path = LOCAL_COMBINED
        graph_path = LOCAL_GRAPH_FEATURES

    print(f"  Loading combined table from: {combined_path}")
    combined = spark.read.parquet(combined_path)
    print(f"  Combined table: {combined.count():,} rows, {len(combined.columns)} columns")

    print(f"  Loading graph features from: {graph_path}")
    graph_features = spark.read.parquet(graph_path)
    print(f"  Graph features: {graph_features.count():,} rows")

    # Join graph features onto combined table by zone_id = vertex_id
    # Graph features add: page_rank, community_id, in_degree, out_degree
    graph_cols = graph_features.select(
        F.col("vertex_id").alias("gf_zone_id"),
        F.col("page_rank"),
        F.col("community_id"),
        F.col("in_degree"),
        F.col("out_degree"),
    )

    merged = combined.join(
        graph_cols,
        combined.zone_id == graph_cols.gf_zone_id,
        how="left"
    ).drop("gf_zone_id")

    # Fill nulls for zones that don't have graph features (unlikely)
    merged = merged.fillna({
        "page_rank": 0.0,
        "community_id": 0,
        "in_degree": 0,
        "out_degree": 0,
    })

    print(f"  Merged table: {merged.count():,} rows, {len(merged.columns)} columns")
    return merged


# =========================================================================
# STEP 2: Feature engineering
# =========================================================================
def engineer_features(df):
    """Create ML features from the merged table.

    The disruption signature we're looking for:
      subway_ridership DROP + taxi_pickups SURGE + bike_starts SURGE
      = subway failure detected

    Features capture both absolute values and cross-modal ratios.
    """

    df = df.withColumn(
        "taxi_to_subway_ratio",
        F.when(F.col("subway_ridership") > 0,
               F.col("taxi_pickups").cast(DoubleType()) / F.col("subway_ridership"))
        .otherwise(0.0)
    )

    df = df.withColumn(
        "bike_to_subway_ratio",
        F.when(F.col("subway_ridership") > 0,
               F.col("bike_starts").cast(DoubleType()) / F.col("subway_ridership"))
        .otherwise(0.0)
    )

    df = df.withColumn(
        "total_surface_transport",
        F.col("taxi_pickups") + F.col("bike_starts")
    )

    df = df.withColumn(
        "complaint_density",
        F.when(F.col("taxi_pickups") > 0,
               F.col("complaint_count").cast(DoubleType()) / F.col("taxi_pickups"))
        .otherwise(0.0)
    )

    return df


# =========================================================================
# STEP 3: Isolation Forest (unsupervised anomaly detection)
# =========================================================================
def run_isolation_forest(df, feature_cols):
    """Detect anomalies using statistical thresholds.

    PySpark doesn't have a native Isolation Forest, so we use a
    z-score approach: flag zone-hours where the cross-modal ratios
    deviate significantly from their historical mean.

    A proper Isolation Forest can be run with sklearn on a sample
    or using spark-iforest library.
    """
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION (Statistical Z-Score Method)")
    print("=" * 60)

    # Compute per-zone historical baselines (mean and stddev)
    baselines = df.groupBy("zone_id", "hour_of_day", "is_weekend").agg(
        F.mean("taxi_pickups").alias("mean_pickups"),
        F.stddev("taxi_pickups").alias("std_pickups"),
        F.mean("subway_ridership").alias("mean_subway"),
        F.stddev("subway_ridership").alias("std_subway"),
        F.mean("bike_starts").alias("mean_bikes"),
        F.stddev("bike_starts").alias("std_bikes"),
    )

    # Join baselines back
    scored = df.join(
        baselines,
        on=["zone_id", "hour_of_day", "is_weekend"],
        how="left"
    )

    # Z-scores: how far is this hour from normal?
    scored = scored.withColumn(
        "z_pickups",
        F.when(F.col("std_pickups") > 0,
               (F.col("taxi_pickups") - F.col("mean_pickups")) / F.col("std_pickups"))
        .otherwise(0.0)
    )
    scored = scored.withColumn(
        "z_subway",
        F.when(F.col("std_subway") > 0,
               (F.col("subway_ridership") - F.col("mean_subway")) / F.col("std_subway"))
        .otherwise(0.0)
    )
    scored = scored.withColumn(
        "z_bikes",
        F.when(F.col("std_bikes") > 0,
               (F.col("bike_starts") - F.col("mean_bikes")) / F.col("std_bikes"))
        .otherwise(0.0)
    )

    # Disruption signature: subway DOWN + taxi UP + bike UP
    # z_subway < -2 (subway ridership dropped 2+ std devs)
    # z_pickups > 2 (taxi surged 2+ std devs)
    # OR z_bikes > 2 (bike surged 2+ std devs)
    scored = scored.withColumn(
        "is_anomaly",
        F.when(
            (F.col("z_subway") < -2.0) &
            ((F.col("z_pickups") > 2.0) | (F.col("z_bikes") > 2.0)),
            1
        ).otherwise(0)
    )

    anomaly_count = scored.filter(F.col("is_anomaly") == 1).count()
    total = scored.count()
    print(f"  Anomalies detected: {anomaly_count:,} / {total:,} ({100*anomaly_count/max(1,total):.3f}%)")

    # Show top anomalies
    print("\n  Top 20 anomalous zone-hours (potential disruptions):")
    (scored
     .filter(F.col("is_anomaly") == 1)
     .select("zone_id", "hourly_timestamp", "taxi_pickups", "subway_ridership",
             "bike_starts", "z_subway", "z_pickups", "z_bikes",
             "page_rank", "temperature_c", "is_rain")
     .orderBy(F.col("z_subway").asc())
     .show(20, truncate=False))

    return scored


# =========================================================================
# STEP 4: GBT Classifier (supervised, uses anomaly labels)
# =========================================================================
def run_gbt_classifier(df, feature_cols):
    """Train a Gradient Boosted Tree classifier on the anomaly labels.

    Uses the z-score anomaly labels as ground truth (since we don't have
    manually labeled disruptions). In production, you'd replace these with
    actual MTA service alerts.

    The graph features (PageRank, community_id, degree) help the model
    understand which zones are structurally important — a disruption at
    a high-PageRank zone cascades more than at a low-PageRank zone.
    """
    print("\n" + "=" * 60)
    print("GBT CLASSIFIER (Supervised Disruption Classification)")
    print("=" * 60)

    # Use only rows with non-zero activity for meaningful classification
    active = df.filter(
        (F.col("taxi_pickups") > 0) | (F.col("subway_ridership") > 0)
    )

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="raw_features",
        handleInvalid="skip"
    )

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=False
    )

    gbt = GBTClassifier(
        labelCol="is_anomaly",
        featuresCol="features",
        maxIter=50,
        maxDepth=5,
        seed=42
    )

    pipeline = Pipeline(stages=[assembler, scaler, gbt])

    # Train/test split
    train, test = active.randomSplit([0.8, 0.2], seed=42)
    print(f"  Train: {train.count():,} rows")
    print(f"  Test:  {test.count():,} rows")
    print(f"  Anomaly rate (train): {train.filter(F.col('is_anomaly')==1).count() / max(1, train.count()) * 100:.3f}%")

    print("  Training GBT model...")
    model = pipeline.fit(train)

    # Evaluate
    predictions = model.transform(test)

    binary_eval = BinaryClassificationEvaluator(labelCol="is_anomaly", metricName="areaUnderROC")
    auc = binary_eval.evaluate(predictions)

    multi_eval = MulticlassClassificationEvaluator(labelCol="is_anomaly", metricName="f1")
    f1 = multi_eval.evaluate(predictions)

    accuracy_eval = MulticlassClassificationEvaluator(labelCol="is_anomaly", metricName="accuracy")
    accuracy = accuracy_eval.evaluate(predictions)

    print(f"\n  Results:")
    print(f"    AUC-ROC:  {auc:.4f}")
    print(f"    F1 Score: {f1:.4f}")
    print(f"    Accuracy: {accuracy:.4f}")

    # Feature importance
    gbt_model = model.stages[-1]
    importances = gbt_model.featureImportances.toArray()
    print(f"\n  Feature Importance (top 10):")
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    for fname, imp in feat_imp[:10]:
        bar = "#" * int(imp * 50)
        print(f"    {fname:<30s} {imp:.4f} {bar}")

    return model, predictions


# =========================================================================
# STEP 5: Save predictions
# =========================================================================
def save_predictions(predictions, use_hdfs=True):
    """Save predictions for Sasmitha's dashboard."""
    output_path = HDFS_OUTPUT if use_hdfs else LOCAL_OUTPUT

    print(f"\n  Saving predictions to: {output_path}")

    output_cols = [
        "zone_id", "hourly_timestamp", "year", "month", "hour_of_day",
        "taxi_pickups", "taxi_dropoffs", "subway_ridership", "bike_starts",
        "temperature_c", "is_rain", "is_snow",
        "complaint_count", "event_count",
        "page_rank", "community_id", "in_degree", "out_degree",
        "z_subway", "z_pickups", "z_bikes",
        "is_anomaly", "prediction",
    ]

    existing_cols = [c for c in output_cols if c in predictions.columns]

    (predictions
     .select(existing_cols)
     .write
     .mode("overwrite")
     .partitionBy("year")
     .parquet(output_path))

    print("  Predictions saved!")


# =========================================================================
# MAIN
# =========================================================================
def main():
    use_hdfs = "--hdfs" in sys.argv

    print("\n" + "=" * 60)
    print("  NEUROTRAFFIC — Disruption Detection ML Pipeline")
    print(f"  Mode: {'HDFS (Docker cluster)' if use_hdfs else 'Local'}")
    print("=" * 60)

    spark = create_spark()

    # Step 1: Load data
    print("\n[1/5] Loading and merging data sources...")
    df = load_data(spark, use_hdfs=use_hdfs)

    # Step 2: Feature engineering
    print("\n[2/5] Engineering features...")
    df = engineer_features(df)

    # Define feature columns for ML
    feature_cols = [
        # Transport metrics
        "taxi_pickups", "taxi_dropoffs", "subway_ridership",
        "bike_starts", "bike_ends",
        # Cross-modal ratios
        "taxi_to_subway_ratio", "bike_to_subway_ratio",
        "total_surface_transport", "complaint_density",
        # Weather
        "temperature_c", "humidity_pct", "precipitation_mm",
        "wind_speed_kmh", "is_rain", "is_snow",
        "is_extreme_cold", "is_extreme_heat",
        # Context
        "hour_of_day", "is_rush_hour", "is_weekend",
        "complaint_count", "max_complaint_severity",
        "event_count",
        # Graph features (from Tanay's GraphX pipeline)
        "page_rank", "community_id", "in_degree", "out_degree",
    ]

    # Step 3: Anomaly detection
    print("\n[3/5] Running anomaly detection...")
    scored = run_isolation_forest(df, feature_cols)

    # Step 4: GBT classification
    print("\n[4/5] Training GBT classifier...")
    model, predictions = run_gbt_classifier(scored, feature_cols)

    # Step 5: Save
    print("\n[5/5] Saving predictions...")
    save_predictions(predictions, use_hdfs=use_hdfs)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
