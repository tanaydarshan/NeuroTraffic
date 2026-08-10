# NeuroTraffic

**Multi-modal NYC transport graph for subway disruption detection and demand cascade prediction**

Big Data Analytics Course (23AID302) | Amrita Vishwa Vidyapeetham

## Team

| Member | Roll No | Module | Directory |
|--------|---------|--------|-----------|
| **Tanay** | CB.AI.U4AID24059 | Graph Engine (Spark GraphX) | `graph-engine/` (this root) |
| **Navaneeth** | — | Data Ingestion & Cleaning (Spark) | `data-pipeline/` |
| **Harsada** | — | ML Pipeline (demand prediction) | `ml-pipeline/` |
| **Sasmitha** | — | Dashboard & Visualization | `dashboard/` |

## Architecture

```
NYC Open Data  ──►  Data Pipeline  ──►  Graph Engine  ──►  ML Pipeline  ──►  Dashboard
(taxi, subway,      (Navaneeth)         (Tanay)            (Harsada)         (Sandhya)
 bike raw data)     Spark ETL           Spark GraphX       Spark MLlib       Streamlit/
                    ↓                   ↓                  ↓                 Plotly
                    Cleaned Parquet     Graph Features     Predictions
                    on HDFS             Parquet            Parquet
```

## Data Flow & Integration Contracts

| Stage | HDFS Path | Format | Producer | Consumer |
|-------|-----------|--------|----------|----------|
| Cleaned nodes | `/data/nodes/*.parquet` | Parquet | Navaneeth | Tanay |
| Combined trips | `/data/combined/trips.parquet` | Parquet | Navaneeth | Tanay |
| Graph features | `/data/graph_features/features.parquet` | Parquet | Tanay | Harsada |
| ML predictions | `/data/predictions/*.parquet` | Parquet | Harsada | Sasmitha |

### Node Parquet schemas

**taxi_zones.parquet**: `zone_id`, `zone_name`, `lat`, `lon`, `capacity`, `borough`

**subway_stations.parquet**: `station_id`, `station_name`, `lat`, `lon`, `capacity`, `borough`

**bike_docks.parquet**: `dock_id`, `dock_name`, `lat`, `lon`, `capacity`, `borough`

### Edge Parquet schema

**trips.parquet**: `src_node_id`, `dst_node_id`, `src_type`, `dst_type`, `trip_count`, `avg_travel_time`, `hour_of_day`, `day_type`

### Graph features schema (Tanay → Harsada)

**features.parquet**: `vertex_id`, `node_type`, `name`, `lat`, `lon`, `borough`, `capacity`, `page_rank`, `community_id`, `in_degree`, `out_degree`

## Graph Engine (Tanay's module)

### Tech Stack
- Apache Spark 3.5.1 + GraphX
- Scala 2.12.18
- SBT 1.9.8 with sbt-assembly

### Graph Schema
- **Nodes**: 263 taxi zones (IDs 1-263) + 493 subway stations (IDs 1000+) + 2,463 bike docks (IDs 5000+) = **3,219 nodes**
- **Edges**: Trip-based + geographic proximity cross-modal edges = **643,652 edges**
- **Algorithms**: PageRank, Label Propagation, Degree computation

### Quick Start

```bash
# 1. Build fat JAR
run.bat build

# 2. Generate sample data (137 nodes — for quick testing)
run.bat gendata

# 3. Generate full-scale data from real NYC sources
pip install pandas pyarrow
python prepare_fullscale_data.py

# 4. Run full pipeline with sample data
run.bat full

# 5. Run with full-scale data
run.bat full --data fullscale_data

# 6. Other modes
run.bat pagerank --data fullscale_data
run.bat timewindow --data fullscale_data
run.bat stats --data fullscale_data
```

### Prerequisites
- Java 8 (JRE 1.8) — Spark 3.5 does not support Java 11+
- Apache Spark 3.5.1
- SBT 1.9.8
- winutils.exe + hadoop.dll (for Windows — set HADOOP_HOME)

## For Teammates

### Navaneeth (Data Pipeline)
Push your Spark ETL code to `data-pipeline/`. Your output Parquet files should match the schemas above. The graph engine reads from `$basePath/nodes/` and `$basePath/combined/`.

### Harsada (ML Pipeline)
Push your ML code to `ml-pipeline/`. Your input is `features.parquet` from the graph engine — it has PageRank, community ID, in/out degree for every node. You can test with the full-scale features already generated.

### Sasmitha (Dashboard)
Push your dashboard code to `dashboard/`. You'll consume both graph features and ML predictions.

### How to contribute
1. Clone: `git clone https://github.com/tanaydarshan/NeuroTraffic.git`
2. Create your module directory and push your code
3. Don't push large data files — they're in `.gitignore`
