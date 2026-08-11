"""
NeuroTraffic — ML Pipeline: Subway Disruption Detection
========================================================
Owner: Harsada K (CB.AI.U4AID24019)
Course: Big Data Analytics (23AID302)

Complete ML pipeline that:
  1. Loads Navaneeth's combined_v3 table + Tanay's graph features
  2. Engineers cross-modal features
  3. Runs Isolation Forest (sklearn) for anomaly detection
  4. Trains GBT Classifier for cascade prediction
  5. Saves predictions + models for Sasmitha's streaming pipeline

Data sources:
  - Combined table: hdfs://namenode:9000/data/neurotraffic/combined_v3
    (6.9M rows, 38 columns: taxi, subway, weather, 311, events, bike per zone per hour)
  - Graph features: hdfs://namenode:9000/data/neurotraffic/graph_features
    (3,219 nodes with page_rank, community_id, in_degree, out_degree)

Usage:
  spark-submit --master local[*] --driver-memory 4g disruption_detector.py --hdfs
  spark-submit --master local[*] --driver-memory 4g disruption_detector.py --local /path/to/data
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml import Pipeline
import numpy as np
import pickle
import sys
import os

# ── Paths ──
HDFS_BASE = "hdfs://namenode:9000/data/neurotraffic"
HDFS_COMBINED = f"{HDFS_BASE}/combined_v3"
HDFS_GRAPH = f"{HDFS_BASE}/graph_features"
HDFS_PREDICTIONS = f"{HDFS_BASE}/predictions"
HDFS_MODELS = f"{HDFS_BASE}/models"

LOCAL_GRAPH = os.path.join(
    os.path.dirname(__file__), "..", "real_data", "graph_features", "features.parquet"
)

# ── Feature columns ──
# These match Navaneeth's combined_v3 schema exactly
FEATURE_COLS = [
    "taxi_pickups", "taxi_dropoffs", "subway_ridership",
    "bike_starts", "bike_ends",
    "taxi_to_subway_ratio", "bike_to_subway_ratio",
    "total_surface_transport", "complaint_density",
    "temperature_c", "humidity_pct", "precipitation_mm",
    "wind_speed_kmh", "is_rain", "is_snow",
    "is_extreme_cold", "is_extreme_heat",
    "hour_of_day", "is_rush_hour", "is_weekend",
    "complaint_count", "max_complaint_severity",
    "event_count",
    "page_rank", "community_id", "in_degree", "out_degree",
]

# Subset for Isolation Forest (core cross-modal features)
IF_FEATURE_COLS = [
    "subway_ridership", "taxi_pickups", "bike_starts",
    "temperature_c", "is_rush_hour", "day_of_week",
    "is_weekend", "page_rank",
]


def create_spark():
    return (
        SparkSession.builder
        .appName("NeuroTraffic_DisruptionDetector")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


# =========================================================================
# STEP 1: Load and merge data
# =========================================================================
def load_data(spark, use_hdfs=True, local_combined=None):
    combined_path = HDFS_COMBINED if use_hdfs else local_combined
    graph_path = HDFS_GRAPH if use_hdfs else LOCAL_GRAPH

    if not use_hdfs and not local_combined:
        print("[ERROR] Pass --local /path/to/combined_v3 for local mode")
        sys.exit(1)

    print(f"  Loading combined table from: {combined_path}")
    combined = spark.read.parquet(combined_path)
    row_count = combined.count()
    print(f"  Combined: {row_count:,} rows, {len(combined.columns)} columns")

    print(f"  Loading graph features from: {graph_path}")
    try:
        graph = spark.read.parquet(graph_path)
        print(f"  Graph features: {graph.count():,} rows")

        graph_cols = graph.select(
            F.col("vertex_id").alias("gf_zone_id"),
            F.col("page_rank"),
            F.col("community_id"),
            F.col("in_degree"),
            F.col("out_degree"),
        )

        merged = combined.join(
            graph_cols,
            combined.zone_id == graph_cols.gf_zone_id,
            how="left",
        ).drop("gf_zone_id")
    except Exception as e:
        print(f"  Graph features not available ({e}), using placeholders")
        merged = combined.withColumn("page_rank", F.lit(0.0))
        merged = merged.withColumn("community_id", F.lit(0))
        merged = merged.withColumn("in_degree", F.lit(0))
        merged = merged.withColumn("out_degree", F.lit(0))

    merged = merged.fillna({
        "page_rank": 0.0, "community_id": 0,
        "in_degree": 0, "out_degree": 0,
        "taxi_pickups": 0, "taxi_dropoffs": 0,
        "subway_ridership": 0, "bike_starts": 0, "bike_ends": 0,
        "complaint_count": 0, "event_count": 0,
    })

    print(f"  Merged: {merged.count():,} rows, {len(merged.columns)} columns")
    return merged


# =========================================================================
# STEP 2: Feature engineering
# =========================================================================
def engineer_features(df):
    df = df.withColumn(
        "taxi_to_subway_ratio",
        F.when(F.col("subway_ridership") > 0,
               F.col("taxi_pickups").cast(DoubleType()) / F.col("subway_ridership"))
        .otherwise(0.0),
    )
    df = df.withColumn(
        "bike_to_subway_ratio",
        F.when(F.col("subway_ridership") > 0,
               F.col("bike_starts").cast(DoubleType()) / F.col("subway_ridership"))
        .otherwise(0.0),
    )
    df = df.withColumn(
        "total_surface_transport",
        F.col("taxi_pickups") + F.col("bike_starts"),
    )
    df = df.withColumn(
        "complaint_density",
        F.when(F.col("taxi_pickups") > 0,
               F.col("complaint_count").cast(DoubleType()) / F.col("taxi_pickups"))
        .otherwise(0.0),
    )
    return df


# =========================================================================
# STEP 3: Isolation Forest (sklearn on sampled data)
# =========================================================================
def run_isolation_forest(df, contamination=0.015):
    from sklearn.ensemble import IsolationForest

    print(f"\n{'=' * 60}")
    print("ISOLATION FOREST — Anomaly Detection")
    print(f"{'=' * 60}")

    # Also compute z-score baselines for comparison
    baselines = df.groupBy("zone_id", "hour_of_day", "is_weekend").agg(
        F.mean("taxi_pickups").alias("mean_pickups"),
        F.stddev("taxi_pickups").alias("std_pickups"),
        F.mean("subway_ridership").alias("mean_subway"),
        F.stddev("subway_ridership").alias("std_subway"),
        F.mean("bike_starts").alias("mean_bikes"),
        F.stddev("bike_starts").alias("std_bikes"),
    )

    scored = df.join(baselines, on=["zone_id", "hour_of_day", "is_weekend"], how="left")

    scored = scored.withColumn(
        "z_subway",
        F.when(F.col("std_subway") > 0,
               (F.col("subway_ridership") - F.col("mean_subway")) / F.col("std_subway"))
        .otherwise(0.0),
    )
    scored = scored.withColumn(
        "z_pickups",
        F.when(F.col("std_pickups") > 0,
               (F.col("taxi_pickups") - F.col("mean_pickups")) / F.col("std_pickups"))
        .otherwise(0.0),
    )
    scored = scored.withColumn(
        "z_bikes",
        F.when(F.col("std_bikes") > 0,
               (F.col("bike_starts") - F.col("mean_bikes")) / F.col("std_bikes"))
        .otherwise(0.0),
    )

    # Collect IF features to driver for sklearn
    # Sample if data is too large (>2M rows)
    total = scored.count()
    sample_frac = min(1.0, 2_000_000 / max(1, total))

    if sample_frac < 1.0:
        print(f"  Sampling {sample_frac*100:.1f}% ({int(total*sample_frac):,} rows) for Isolation Forest")
        sample_df = scored.sample(fraction=sample_frac, seed=42)
    else:
        sample_df = scored

    pandas_df = sample_df.select(
        "zone_id", "hourly_timestamp", *IF_FEATURE_COLS
    ).toPandas()

    X = pandas_df[IF_FEATURE_COLS].fillna(0).values
    print(f"  Feature matrix: {X.shape}")

    # Train Isolation Forest
    model_if = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    print(f"  Training (contamination={contamination})...")
    model_if.fit(X)

    # Score
    raw_scores = model_if.decision_function(X)
    predictions = model_if.predict(X)

    s_min, s_max = raw_scores.min(), raw_scores.max()
    anomaly_scores = 1.0 - (raw_scores - s_min) / (s_max - s_min)
    anomaly_scores = np.clip(anomaly_scores, 0.0, 1.0)

    n_anomalies = (predictions == -1).sum()
    print(f"  Anomalies: {n_anomalies:,} / {len(predictions):,} "
          f"({n_anomalies/len(predictions)*100:.3f}%)")

    # Feature comparison: normal vs anomalous
    anomaly_mask = predictions == -1
    normal_mask = predictions == 1

    print(f"\n  Cross-modal signature (anomaly / normal ratio):")
    for i, feat in enumerate(IF_FEATURE_COLS[:3]):
        normal_mean = X[normal_mask, i].mean() if normal_mask.sum() > 0 else 0
        anomaly_mean = X[anomaly_mask, i].mean() if anomaly_mask.sum() > 0 else 0
        ratio = anomaly_mean / max(normal_mean, 1e-9)
        direction = "DOWN" if ratio < 0.8 else ("UP" if ratio > 1.2 else "~same")
        print(f"    {feat:>22}: {direction} ({ratio:.2f}x)")

    # Save model
    model_path = "isolation_forest_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_if, f)
    scaler_path = "if_scaler_params.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({
            "feature_cols": IF_FEATURE_COLS,
            "score_min": float(s_min),
            "score_max": float(s_max),
            "contamination": contamination,
        }, f)
    print(f"  Model saved to {model_path}")

    # Upload to HDFS
    os.system(f"hadoop fs -mkdir -p {HDFS_MODELS}/isolation_forest/")
    os.system(f"hadoop fs -put -f {model_path} {HDFS_MODELS}/isolation_forest/")
    os.system(f"hadoop fs -put -f {scaler_path} {HDFS_MODELS}/isolation_forest/")

    # Add z-score based anomaly label for GBT training
    # Disruption signature: subway DOWN + (taxi UP or bike UP)
    scored = scored.withColumn(
        "is_anomaly",
        F.when(
            (F.col("z_subway") < -2.0) &
            ((F.col("z_pickups") > 2.0) | (F.col("z_bikes") > 2.0)),
            1,
        ).otherwise(0),
    )

    anomaly_count = scored.filter(F.col("is_anomaly") == 1).count()
    print(f"\n  Z-score anomalies (for GBT labels): {anomaly_count:,} / {total:,} "
          f"({100*anomaly_count/max(1,total):.3f}%)")

    print("\n  Top 20 anomalous zone-hours:")
    (scored
     .filter(F.col("is_anomaly") == 1)
     .select("zone_id", "hourly_timestamp", "taxi_pickups", "subway_ridership",
             "bike_starts", "z_subway", "z_pickups", "z_bikes",
             "page_rank", "temperature_c", "is_rain")
     .orderBy(F.col("z_subway").asc())
     .show(20, truncate=False))

    return scored, model_if


# =========================================================================
# STEP 4: GBT Classifier
# =========================================================================
def run_gbt_classifier(df, feature_cols):
    print(f"\n{'=' * 60}")
    print("GBT CLASSIFIER — Disruption Classification")
    print(f"{'=' * 60}")

    active = df.filter(
        (F.col("taxi_pickups") > 0) | (F.col("subway_ridership") > 0)
    )

    # Cast all feature cols to double
    for col in feature_cols:
        if col in active.columns:
            active = active.withColumn(col, F.col(col).cast(DoubleType()))

    available_cols = [c for c in feature_cols if c in active.columns]
    missing_cols = [c for c in feature_cols if c not in active.columns]
    if missing_cols:
        print(f"  Skipping missing columns: {missing_cols}")

    assembler = VectorAssembler(
        inputCols=available_cols,
        outputCol="raw_features",
        handleInvalid="skip",
    )

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=False,
    )

    gbt = GBTClassifier(
        labelCol="is_anomaly",
        featuresCol="features",
        maxIter=100,
        maxDepth=5,
        stepSize=0.1,
        subsamplingRate=0.8,
        seed=42,
    )

    pipeline = Pipeline(stages=[assembler, scaler, gbt])

    train, test = active.randomSplit([0.8, 0.2], seed=42)
    train_count = train.count()
    test_count = test.count()
    anomaly_rate = train.filter(F.col("is_anomaly") == 1).count() / max(1, train_count)

    print(f"  Train: {train_count:,} | Test: {test_count:,}")
    print(f"  Anomaly rate: {anomaly_rate*100:.3f}%")
    print(f"  Features: {len(available_cols)}")
    print("  Training GBT (maxIter=100, maxDepth=5)...")

    model = pipeline.fit(train)
    predictions = model.transform(test)

    # Evaluate
    auc = BinaryClassificationEvaluator(
        labelCol="is_anomaly", metricName="areaUnderROC"
    ).evaluate(predictions)

    f1 = MulticlassClassificationEvaluator(
        labelCol="is_anomaly", metricName="f1"
    ).evaluate(predictions)

    accuracy = MulticlassClassificationEvaluator(
        labelCol="is_anomaly", metricName="accuracy"
    ).evaluate(predictions)

    precision = MulticlassClassificationEvaluator(
        labelCol="is_anomaly", metricName="weightedPrecision"
    ).evaluate(predictions)

    recall = MulticlassClassificationEvaluator(
        labelCol="is_anomaly", metricName="weightedRecall"
    ).evaluate(predictions)

    print(f"\n  Results:")
    print(f"    AUC-ROC:   {auc:.4f}")
    print(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1 Score:  {f1:.4f}")

    # Feature importance
    gbt_model = model.stages[-1]
    importances = gbt_model.featureImportances.toArray()
    print(f"\n  Feature Importance (top 10):")
    feat_imp = sorted(zip(available_cols, importances), key=lambda x: -x[1])
    for fname, imp in feat_imp[:10]:
        bar = "#" * int(imp * 50)
        print(f"    {fname:<30s} {imp:.4f} {bar}")

    # Confusion matrix
    pred_labels = predictions.select("is_anomaly", "prediction").toPandas()
    y_true = pred_labels["is_anomaly"].astype(int).values
    y_pred = pred_labels["prediction"].astype(int).values

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                Normal  Anomaly")
    print(f"    Actual Normal  {tn:>7,}  {fp:>7,}")
    print(f"    Actual Anomaly {fn:>7,}  {tp:>7,}")

    # Save GBT model
    gbt_path = f"{HDFS_MODELS}/gbt_cascade"
    try:
        model.save(gbt_path)
        print(f"\n  GBT pipeline saved to {gbt_path}")
    except Exception as e:
        local_path = "gbt_cascade_model"
        model.save(local_path)
        print(f"\n  GBT pipeline saved locally to {local_path}/ (HDFS save failed: {e})")

    return model, predictions, {
        "auc": auc, "f1": f1, "accuracy": accuracy,
        "precision": precision, "recall": recall,
        "feature_importance": feat_imp,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


# =========================================================================
# STEP 5: Save predictions
# =========================================================================
def save_predictions(predictions, use_hdfs=True):
    output_path = HDFS_PREDICTIONS if use_hdfs else "predictions_output"

    output_cols = [
        "zone_id", "hourly_timestamp", "year", "month", "hour_of_day",
        "zone_borough", "zone_name",
        "taxi_pickups", "taxi_dropoffs", "subway_ridership",
        "bike_starts", "bike_ends",
        "temperature_c", "is_rain", "is_snow",
        "complaint_count", "event_count",
        "page_rank", "community_id", "in_degree", "out_degree",
        "z_subway", "z_pickups", "z_bikes",
        "is_anomaly", "prediction",
    ]

    existing_cols = [c for c in output_cols if c in predictions.columns]

    print(f"\n  Saving {len(existing_cols)} columns to: {output_path}")
    (predictions
     .select(existing_cols)
     .write
     .mode("overwrite")
     .partitionBy("year")
     .parquet(output_path))
    print("  Predictions saved!")


# =========================================================================
# STEP 6: Generate evaluation report data
# =========================================================================
def save_evaluation_report(metrics, output_dir="evaluation_output"):
    import json
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "model": "GBT Classifier",
        "dataset": "combined_v3 (6.9M rows)",
        "auc_roc": metrics["auc"],
        "f1_score": metrics["f1"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "confusion_matrix": metrics["confusion_matrix"],
        "top_features": [
            {"name": name, "importance": float(imp)}
            for name, imp in metrics["feature_importance"][:15]
        ],
    }

    report_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Evaluation report saved to {report_path}")

    # Also upload to HDFS for the dashboard
    os.system(f"hadoop fs -mkdir -p {HDFS_BASE}/evaluation/")
    os.system(f"hadoop fs -put -f {report_path} {HDFS_BASE}/evaluation/")


# =========================================================================
# MAIN
# =========================================================================
def main():
    use_hdfs = "--hdfs" in sys.argv
    local_combined = None
    if "--local" in sys.argv:
        idx = sys.argv.index("--local")
        local_combined = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        use_hdfs = False

    print(f"\n{'=' * 60}")
    print("  NEUROTRAFFIC — Disruption Detection ML Pipeline")
    print(f"  Mode: {'HDFS (Docker cluster)' if use_hdfs else 'Local'}")
    print(f"{'=' * 60}")

    spark = create_spark()

    print("\n[1/6] Loading and merging data sources...")
    df = load_data(spark, use_hdfs=use_hdfs, local_combined=local_combined)

    print("\n[2/6] Engineering features...")
    df = engineer_features(df)

    print("\n[3/6] Running Isolation Forest...")
    scored, if_model = run_isolation_forest(df, contamination=0.015)

    print("\n[4/6] Training GBT classifier...")
    model, predictions, metrics = run_gbt_classifier(scored, FEATURE_COLS)

    print("\n[5/6] Saving predictions...")
    save_predictions(predictions, use_hdfs=use_hdfs)

    print("\n[6/6] Saving evaluation report...")
    save_evaluation_report(metrics)

    print(f"\n{'=' * 60}")
    print("  PIPELINE COMPLETE")
    print(f"  AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f} | "
          f"Accuracy: {metrics['accuracy']:.1%}")
    print(f"{'=' * 60}")

    spark.stop()


if __name__ == "__main__":
    main()
