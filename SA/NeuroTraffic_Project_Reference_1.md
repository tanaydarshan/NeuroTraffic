# NeuroTraffic — Complete Project Reference

> **Use this file as context in any new Claude chat to continue working on the project.**

---

## 1. Project identity

- **Project name:** NeuroTraffic — City-Scale Multi-Modal Transport Intelligence Using Big Data Analytics
- **Course:** 23AID302 — Big Data Analytics | Section AID-A | AY 2026-27
- **University:** Amrita Vishwa Vidyapeetham, Coimbatore — Dept. of AI & Data Science
- **Team:**
  - V. Tanay Darshan — CB.AI.U4AID24059 — System architect + Graph engine (Spark GraphX, Scala)
  - Ravula Navaneeth — CB.AI.U4AID24044 — Data pipeline (HDFS + Spark SQL cleaning + joining)
  - Harsada K — CB.AI.U4AID24019 — ML models (Spark MLlib — Isolation Forest + GBT)
  - Sasmitha S — CB.AI.U4AID24051 — Streaming pipeline + Dashboard (Spark Streaming + Plotly Dash)

---

## 2. One-line summary

Fuse 3 billion+ NYC taxi, bike, and subway records into a cross-modal transport graph that detects subway disruptions from behavioral signatures and predicts how demand cascades across all transport modes — something no single-mode system can do.

---

## 3. Problem statement

Current urban transport analytics systems process each mode (taxi, subway, bike) in isolation, making them unable to detect cross-modal disruption patterns or predict how failures in one mode cascade demand surges into others. There is no scalable big data system that unifies multi-modal transport data into a single graph structure to enable real-time disruption detection from cross-modal behavioral signatures and cascade demand prediction across the connected transport network.

---

## 4. The 5 V's of Big Data — how each is reflected

| V | How it appears in NeuroTraffic |
|---|---|
| **Volume** | 3 billion+ records across 6 data sources, ~350 GB total |
| **Velocity** | Spark Structured Streaming on real-time turnstile and taxi event feeds |
| **Variety** | Structured (trip CSVs, Parquet), semi-structured (weather JSON, 311 JSON), unstructured (event permit text) |
| **Veracity** | GPS drift in taxi data, turnstile counter resets, duplicate bike trips, weather station outages, missing values |
| **Value** | Real-time disruption detection, cascade demand prediction, cross-modal demand forecasting saves lives and money |

---

## 5. Data sources — complete details

### Source 1: NYC Taxi & Limousine Commission (TLC)
- **URL:** https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **Format:** Parquet files (one per month, per taxi type: yellow, green, FHV)
- **Size:** ~227 GB (2009–2024)
- **Records:** ~1.3 billion taxi trips
- **Key columns:** pickup_datetime, dropoff_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count, trip_distance, fare_amount, PULocationID, DOLocationID
- **Data type:** Structured
- **Known issues:** GPS coordinates outside NYC bounds (lat 0.0, lon 0.0), trips with 0 passengers, trips with 0 distance, some months have different column names

### Source 2: Citi Bike System Data
- **URL:** https://citibikenyc.com/system-data (or https://s3.amazonaws.com/tripdata/)
- **Format:** CSV files (one per month)
- **Size:** ~10 GB
- **Records:** 100 million+ bike trips
- **Key columns:** ride_id, started_at, ended_at, start_station_name, start_station_id, end_station_name, end_station_id, start_lat, start_lng, end_lat, end_lng, member_casual
- **Data type:** Structured
- **Known issues:** Duplicate ride_ids, some trips with identical start and end station (likely docking errors), station names change over time, some coordinates missing

### Source 3: MTA Turnstile Usage Data
- **URL:** https://data.ny.gov/Transportation/Turnstile-Usage-Data/ (or http://web.mta.info/developers/turnstile.html for legacy)
- **Format:** CSV (weekly files)
- **Size:** ~50 GB
- **Records:** Billions of entry/exit counts
- **Key columns:** C/A (control area), UNIT, SCP (subunit channel position), STATION, LINENAME, DATE, TIME, ENTRIES (cumulative), EXITS (cumulative)
- **Data type:** Structured
- **Known issues:** Counters are CUMULATIVE (not per-period — you must compute differences). Counters reset at arbitrary values. Negative differences indicate resets. Some turnstiles report every 4 hours, some irregularly. Station names don't always match other datasets.

### Source 4: NOAA Weather Data
- **URL:** https://www.ncdc.noaa.gov/cdo-web/ (or NOAA ISD via FTP)
- **Format:** JSON (via API) or CSV (via bulk download)
- **Size:** ~5 GB
- **Records:** Hourly weather observations for NYC weather stations
- **Key columns:** datetime, temperature, precipitation, wind_speed, visibility, weather_condition
- **Data type:** Semi-structured (JSON from API)
- **Known issues:** Missing readings during station outages, multiple stations with slightly different readings, units vary between metric and imperial

### Source 5: NYC 311 Complaints
- **URL:** https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/
- **Format:** JSON (via Socrata API) or CSV
- **Size:** ~15 GB
- **Records:** 30 million+ complaints
- **Key columns:** created_date, complaint_type, descriptor, incident_zip, latitude, longitude, status, borough
- **Data type:** Semi-structured (JSON with nested fields)
- **Relevant complaint types:** Street Condition, Traffic Signal Condition, Water Main Break, Blocked Driveway, Construction, Noise — these indicate road disruptions that affect transport
- **Known issues:** Not all complaints have coordinates, complaint_type taxonomy changed over the years

### Source 6: NYC Event Permits
- **URL:** https://data.cityofnewyork.us/City-Government/Film-Permits/ and https://data.cityofnewyork.us/City-Government/Street-Event-Permits/
- **Format:** CSV/JSON
- **Size:** ~1 GB
- **Records:** Thousands of event records
- **Key columns:** event_name, start_date, end_date, event_type, street_closures (free text), borough, community_board
- **Data type:** Unstructured (the street_closures field is free-text descriptions like "Madison Ave between 23rd and 26th St closed for marathon")
- **Known issues:** Free text is inconsistent, no standardized location format

---

## 6. Technical architecture — full pipeline

```
Step 1: DATA ACQUISITION
├── Download all 6 datasets from their sources
├── Organize locally by source
└── Upload to HDFS cluster

Step 2: HDFS STORAGE
├── /data/taxi/          (Parquet, partitioned by year/month)
├── /data/bike/          (CSV, partitioned by year/month)
├── /data/turnstile/     (CSV, partitioned by week)
├── /data/weather/       (JSON, partitioned by year)
├── /data/311/           (JSON, partitioned by year)
└── /data/events/        (CSV/text)

Step 3: SPARK SQL — CLEANING + JOINING
├── Clean taxi: remove GPS outliers, zero-passenger trips, zero-distance trips
├── Clean bike: remove duplicates, standardize station names, drop missing coords
├── Clean turnstile: compute per-period counts from cumulative, fix counter resets
├── Clean weather: interpolate missing readings, standardize units
├── Clean 311: extract relevant complaint types, parse locations
├── Clean events: NLP on free-text street closures to extract affected areas
├── JOIN all sources by zone_id + hourly_timestamp
└── Write combined table to HDFS as Parquet: /data/combined/

Step 4: SPARK GRAPHX — GRAPH CONSTRUCTION (Scala)
├── Define nodes: 263 taxi zones + 472 subway stations + 2000+ bike docks
├── Define edges: trip volumes between nodes, weighted by count and travel time
├── Build time-windowed graphs (different edge weights for different hours)
├── Run PageRank → identify most critical nodes
├── Run Label Propagation → identify transport communities
└── Extract graph features for ML: node centrality, degree, community membership

Step 5: SPARK MLlib — ML MODELS
├── Feature engineering: [turnstile_exits, taxi_pickups, bike_checkouts,
│                         temperature, is_rush_hour, day_of_week, node_centrality]
├── Isolation Forest: train on normal data → detect anomalies (disruptions)
├── GBT Classifier: train on historical disruption events → predict cascade surges
├── Evaluation: precision, recall, F1, RMSE, confusion matrix
└── Save trained models to HDFS

Step 6: SPARK STRUCTURED STREAMING — REAL-TIME
├── Data replay simulator: reads historical data, pushes to streaming directory
├── Streaming pipeline: reads incoming batches every few seconds
├── Apply Isolation Forest → flag anomalies
├── If anomaly detected → apply GBT → predict cascade
└── Write predictions to dashboard sink

Step 7: STREAMLIT WEB APP — MULTI-PAGE APPLICATION
├── Page 1: Live Monitor
│   ├── NYC map (Plotly/Folium) with color-coded zones (green/yellow/red)
│   ├── Real-time anomaly scores updating as streaming pipeline runs
│   └── Cascade alerts with predicted surge percentages and timelines
├── Page 2: Disruption Simulator
│   ├── User selects a subway line + time of day
│   ├── Clicks "Simulate disruption"
│   └── App shows predicted cascade — which zones turn red, in what order, with what surge %
├── Page 3: Historical Explorer
│   ├── User selects a date range
│   ├── Shows actual disruptions that happened, how demand shifted
│   └── Model performance: predicted vs actual surges
├── Page 4: Analytics Dashboard
│   ├── Model evaluation metrics (accuracy, confusion matrix, feature importance)
│   ├── Graph analytics (PageRank top nodes, community map)
│   └── Cross-modal correlation heatmaps
└── Note: App reads pre-computed results from the pipeline (Parquet/CSV/SQLite).
    The 350 GB Spark processing runs separately on the Hadoop cluster.
    The app does NOT run Spark directly — it visualizes the outputs.
```

---

## 7. Tech stack — every tool and its purpose

| Tool | Purpose | Language |
|---|---|---|
| Hadoop HDFS | Distributed storage for all 350 GB across cluster nodes | CLI |
| MapReduce | Batch aggregation of turnstile counts and daily trip summaries | Java/Python |
| Apache Spark SQL | Data cleaning, transformation, joining all 6 sources | PySpark (Python) |
| Spark GraphX | Multi-modal transport graph construction, PageRank, Label Propagation, community detection | Scala |
| Spark MLlib | Isolation Forest (anomaly detection), GBT Classifier (cascade prediction), feature engineering | PySpark (Python) |
| Spark Structured Streaming | Real-time ingestion, scoring incoming events, triggering cascade predictions | PySpark (Python) |
| Streamlit | Multi-page web app framework — Live Monitor, Disruption Simulator, Historical Explorer, Analytics Dashboard | Python |
| Plotly / Folium | Interactive NYC map with color-coded zones, cascade animation, charts | Python |
| Matplotlib | Static charts for evaluation (confusion matrices, feature importance, ROC curves) | Python |
| SQLite (optional) | Lightweight database to store pipeline outputs for the app to read | Python |

**Languages used:** Python (PySpark for SQL, MLlib, Streaming, Dashboard) + Scala (GraphX only — required because GraphX doesn't have a Python API)

**Cluster setup:** 3+ node Hadoop cluster via Docker (or AWS EMR / Google Dataproc for cloud). Master node + 2-3 worker nodes. Each node needs 8GB+ RAM ideally. Tanay's existing Docker Hadoop setup at `C:\Users\tanay\Big_Data` can be used as starting point.

---

## 8. ML models — detailed specifications

### Model 1: Isolation Forest (anomaly/disruption detection)

- **Purpose:** Detect subway disruptions from cross-modal behavioral signatures
- **Input features (per zone per hour):**
  - `turnstile_exits` — number of subway exits at nearest station
  - `taxi_pickups` — number of taxi pickups in this zone
  - `bike_checkouts` — number of bike checkouts at nearest docks
  - `temperature` — current temperature (weather affects transport patterns)
  - `is_rush_hour` — binary (1 if 7-10 AM or 5-8 PM)
  - `day_of_week` — 0-6 (Monday-Sunday)
  - `is_weekend` — binary
  - `node_centrality` — PageRank score from the transport graph (how critical this zone is)
- **Training data:** Months of NORMAL operation only (no disruptions). The model learns what "normal" looks like.
- **Key parameter:** `contamination` — expected fraction of anomalies (set to 0.01-0.02, meaning 1-2% of data is expected to be anomalous)
- **Output:** Anomaly score between 0 and 1. Scores > 0.8 are flagged as disruptions.
- **How the detection works:** When turnstile exits DROP and taxi pickups SPIKE and bike checkouts SPIKE simultaneously in the same zone, the feature vector is very different from any normal pattern → high anomaly score → disruption detected.

### Model 2: GBT Classifier (cascade demand prediction)

- **Purpose:** Once a disruption is detected, predict which zones will be affected and by how much
- **Input features:**
  - `hour` — time of day
  - `day_of_week` — which day
  - `temperature` — weather conditions
  - `line_affected` — which subway line is disrupted (encoded)
  - `stations_affected` — how many stations are affected
  - `current_taxi_demand` — current taxi demand in the zone
  - `current_bike_demand` — current bike demand
  - `node_centrality` — how critical this zone is in the graph
  - `graph_distance` — how many hops away from the disruption in the graph
- **Training data:** Historical disruption events from MTA service alerts, matched against actual demand surges in surrounding zones
- **Output:** Predicted surge level (classification: low/medium/high/critical, or regression: percentage surge)
- **Key parameters:** `maxIter=100`, `maxDepth=5`, `stepSize=0.1`
- **Evaluation metrics:** Accuracy, precision, recall, F1-score, RMSE (if regression), confusion matrix

---

## 9. The L-train example — the story that explains everything

This is the scenario you use in presentations and viva to explain the system:

It's Tuesday 8:47 AM. The L train in Manhattan stops — signal failure.

**Without NeuroTraffic:** The MTA takes 15-20 minutes to issue an official alert. By then, bike docks are empty, taxi surge pricing is at 3x, and the 6 train is dangerously overcrowded. Everyone reacts after the damage is done.

**With NeuroTraffic:**

- **Minute 0-2:** Turnstile exits at L train stations drop to near zero. Taxi pickups near 14th Street surge 280%. Bike checkouts at 1st Ave dock spike 460%.
- **Minute 2:** Isolation Forest scores this zone at 0.94 (highly anomalous). The cross-modal pattern (turnstiles DOWN + taxis UP + bikes UP) matches the signature of a subway failure.
- **Minute 2-3:** GBT predicts the cascade:
  - "1st Ave bike dock will empty in 12 minutes"
  - "Taxi surge pricing in East Village will peak at 3.2x in 8 minutes"
  - "6 train at Astor Place will hit 120% capacity in 20 minutes"
  - "Second-order taxi surges near Astor Place in 35 minutes"
- **Minute 3:** Dashboard lights up — affected zones turn from green to yellow to red in sequence showing the predicted cascade.
- **Action:** Bike rebalancing trucks dispatched to 1st Ave dock. 6 train alerted to run extra trains. Taxi dispatchers send more cars to East Village.

Total time from failure to prediction: ~3 minutes (vs 15-20 minutes for official MTA alert).

---

## 10. Three novelty layers

### Novelty 1: Multi-modal transport graph (Spark GraphX)
All 263 taxi zones + 472 subway stations + 2,000+ bike docks become nodes in a single graph. Edges represent trips weighted by volume and travel time. PageRank identifies the most critical nodes. Label Propagation finds transport communities. The graph changes shape by time of day — rush hour has different critical nodes than midnight. Nobody has built this unified cross-modal transport graph before.

### Novelty 2: Cross-modal disruption detection without official alerts
Instead of waiting for the MTA to announce a failure, the system detects it from the behavioral signature — the simultaneous pattern of turnstile exits dropping AND taxi pickups spiking AND bike checkouts surging in the same area. This is how companies like Google detect traffic incidents — from behavior patterns, not official reports. Isolation Forest on cross-modal feature vectors makes this automatic.

### Novelty 3: Cascade demand prediction via graph propagation
Once a disruption is detected, the graph structure tells us where demand will shift. If historically 60% of Station A's morning commuters route through Station B, we predict Station B gets a 60% demand surge within 15 minutes. This cascade prediction — predicting the chain reaction of consequences across the entire network — is something single-mode systems fundamentally cannot do.

---

## 11. Literature review — 5 papers and the research gap

| # | Paper | Journal | What they did | Limitation (gap we fill) |
|---|---|---|---|---|
| 1 | Zhang et al. (2024) | Transportation Research Part A, Vol. 183 | Dynamic resilience assessment framework for multi-modal transport under metro disruptions | Analytical framework only — no real-time big data processing, no cross-modal data fusion at scale, no graph modeling |
| 2 | Chen et al. (2025) | Physica A: Statistical Mechanics | Cascading failure processes of interdependent multi-modal transit networks | Theoretical simulation — doesn't process real transport data at scale, no streaming, no anomaly detection from actual records |
| 3 | Ma et al. (2025) | Transportation Research Part A | Resilience of multi-modal transport with urban air mobility integration | Static network topology — no dynamic data-driven detection, no real-time cascade prediction |
| 4 | Ding et al. (2025) | Transportation Research Part E, Vol. 202 | Multilayer network cascading failure with data calibration for rail transit | Rail only — no cross-modal fusion with taxis/bikes, no Spark/Hadoop distributed processing |
| 5 | Saputri et al. (2025) | Int. J. Elec. & Comp. Eng., Vol. 15(1) | NYC taxi big data with Spark — revenue prediction using OLS and L-BFGS | Single mode (taxi only) — no graph, no multi-modal fusion, no disruption analysis |

**Common research gap:** No existing system combines (a) real multi-modal transport data at billion-record scale, (b) graph-based modeling of cross-modal dependencies, (c) data-driven disruption detection from behavioral signatures, and (d) real-time cascade demand prediction — all within a single distributed big data pipeline (Hadoop + Spark).

---

## 12. Work division — who does what

### Tanay (CB.AI.U4AID24059) — System architect + Graph engine
- Design full system architecture and integration plan
- Build the multi-modal transport graph in Scala using Spark GraphX
- Define graph schema (nodes: taxi zones, subway stations, bike docks; edges: trips between them)
- Run PageRank (critical node identification) and Label Propagation (community detection)
- Build time-windowed graphs (different edge weights for different hours)
- Extract graph features for ML models (node centrality, degree, community membership)
- Final integration of all 4 modules into the complete pipeline
- **Deliverables:** GraphX Scala code, graph schema, PageRank results, community detection results, final integrated pipeline

### Navaneeth (CB.AI.U4AID24044) — Data pipeline
- Download all 6 datasets from their respective sources
- Upload to HDFS with proper folder structure and partitioning
- Clean all 6 sources using PySpark and Spark SQL:
  - Taxi: remove GPS outliers (coords outside NYC), zero-passenger/zero-distance trips
  - Bike: remove duplicate ride_ids, standardize station names, drop missing coordinates
  - Turnstile: compute per-period counts from cumulative counters, fix counter resets (negative diffs)
  - Weather: interpolate missing readings, standardize units to metric
  - 311: filter relevant complaint types (street condition, traffic signal, water main), parse locations
  - Events: basic NLP on free-text street_closures to extract affected streets and dates
- Join all 6 sources by zone_id + hourly_timestamp into one combined table
- Write combined table to HDFS as Parquet
- **Deliverables:** All datasets on HDFS, cleaning PySpark code for all 6 sources, final joined Parquet table, data quality report

### Harsada (CB.AI.U4AID24019) — ML models
- Feature engineering from the combined table (build feature vectors for each zone/hour)
- Normalize features (StandardScaler)
- Train Isolation Forest on normal (non-disrupted) data for anomaly detection
- Tune contamination parameter (1-2%)
- Get MTA service alert data for ground truth disruption labels
- Train GBT Classifier on historical disruption events for cascade prediction
- Evaluate both models: precision, recall, F1, RMSE, confusion matrices
- Feature importance analysis
- **Deliverables:** Feature engineering code, trained Isolation Forest model, trained GBT model, evaluation report with metrics and charts

### Sasmitha (CB.AI.U4AID24051) — Streaming pipeline + Web App
- Build data replay simulator (replays historical data as real-time stream)
- Build Spark Structured Streaming pipeline:
  - Read incoming batches from streaming directory
  - Apply Isolation Forest for anomaly scoring
  - If anomaly detected, apply GBT for cascade prediction
  - Write predictions to SQLite/CSV for the app to read
- Build the multi-page Streamlit web app (the main user-facing deliverable):
  - **Page 1 — Live Monitor:** NYC map (Plotly/Folium) with color-coded zones updating in real time as the streaming pipeline detects anomalies and predicts cascades
  - **Page 2 — Disruption Simulator:** User picks a subway line + time of day, clicks "Simulate disruption," app shows the predicted cascade with zones turning red in sequence
  - **Page 3 — Historical Explorer:** User selects a date range, sees actual disruptions, how demand shifted, and model predicted vs actual surge comparison
  - **Page 4 — Analytics Dashboard:** Model evaluation metrics (accuracy, confusion matrix, feature importance), graph analytics (PageRank top nodes, community map)
  - App reads pre-computed results from the Spark pipeline (Parquet/CSV/SQLite) — it does NOT run Spark directly
- Record demo video of the full app in action
- **Deliverables:** Replay simulator, streaming pipeline code, complete Streamlit web app (4 pages), demo video

---

## 13. Shared tasks (all 4 members together)

### Weeks 1-2: Foundation
- ALL: Read and summarize the 5 research papers (1 paragraph each)
- ALL: Install and test Hadoop cluster on Docker
- ALL: Understand the full architecture (everyone should be able to explain the whole system)
- ALL: Set up a shared GitHub repository with folders for each module

### Weeks 7-8: Integration
- ALL: Sit together and connect all 4 modules
- Connect Navaneeth's cleaned data → Tanay's graph → Harsada's models → Sasmitha's streaming pipeline + web app
- Ensure pipeline outputs (model results, graph features, predictions) are written in a format the Streamlit app can read (Parquet/CSV/SQLite)
- Test the full pipeline end-to-end with a small sample of data first
- Fix integration bugs (this ALWAYS takes longer than expected)

### Weeks 14-15: Final deliverables
- ALL: Each person writes the report section for their own module
- ALL: Prepare the final presentation together
- ALL: Practice the demo and prepare for viva questions

---

## 14. Week-by-week timeline

| Week | Tanay | Navaneeth | Harsada | Sasmitha |
|---|---|---|---|---|
| 1-2 | Architecture design, cluster setup | Download all 6 datasets | Read ML papers, understand IF & GBT | Learn Spark Streaming, set up Plotly |
| 3-4 | Help with HDFS setup, start graph schema | Upload to HDFS, start cleaning | Start feature engineering on samples | Build data replay simulator |
| 5 | Graph schema finalized | Complete cleaning + join all sources | Continue feature engineering | Replay simulator working |
| 6-7 | Build GraphX graph in Scala | Support — fix data issues found | Train Isolation Forest, tune params | Build Streamlit app skeleton (4 pages) + NYC map |
| 8 | Run PageRank + community detection | Support — add features if needed | Train GBT on disruption data | Build streaming pipeline + Live Monitor page |
| 9-10 | Extract graph features for ML | Data validation, edge case testing | Evaluate both models, metrics | Build Disruption Simulator + Historical Explorer pages |
| 11 | Integration: connect all 4 modules | Integration support | Integration support | Connect streaming → app, build Analytics page |
| 12-13 | End-to-end testing, fix bugs | Test with edge cases | Model refinement | Full app testing, demo prep, record video |
| 14-15 | Final report (arch section), PPT | Final report (data section), PPT | Final report (ML section), PPT | Final report (app + streaming section), PPT |

---

## 15. Project objectives (for report/presentation)

1. To design a multi-modal transport graph architecture that unifies taxi trip data, subway turnstile data, and bike-share data into a single Spark GraphX network with dynamic edge weights.
2. To implement a distributed data ingestion and cleaning pipeline using Hadoop HDFS and Spark SQL to process ~350 GB of heterogeneous transport data (structured, semi-structured, and unstructured).
3. To develop a cross-modal disruption detection model using Isolation Forest that identifies subway failures from simultaneous anomaly patterns in taxi, bike, and turnstile data streams.
4. To build a cascade demand prediction engine that uses graph propagation on the transport graph to forecast which stations, zones, and docks will be affected by a detected disruption.
5. To implement a Spark Structured Streaming pipeline for near real-time scoring of disruption cascades.
6. To build a multi-page Streamlit web application with live monitoring, disruption simulation, historical exploration, and analytics dashboarding capabilities.
7. To evaluate the system's detection accuracy and prediction performance using historical disruption events from NYC MTA service alerts as ground truth.

---

## 16. Viva preparation — likely questions and answers

**Q: Why not just use the MTA's official service alerts?**
A: MTA alerts can take 15-20 minutes to be published. Our system detects disruptions in 2-3 minutes from behavioral signatures — before any official announcement.

**Q: Why do you need graph analytics? Can't you just use regular ML?**
A: Regular ML treats each zone independently. The graph captures how zones are connected — it tells us that when Station A fails, demand cascades to Station B because historically 60% of A's commuters reroute through B. Without the graph, you can't predict cascades.

**Q: How do you handle the 350 GB of data?**
A: Hadoop HDFS distributes the data across multiple cluster nodes. Spark processes it in parallel across all nodes. No single machine handles all 350 GB alone.

**Q: Why Isolation Forest and not another anomaly detection method?**
A: Isolation Forest is designed for high-dimensional anomaly detection with unsupervised learning. We don't need labeled "disruption" data to train it — it learns normal patterns and flags anything that deviates. It's also efficient on large datasets in Spark MLlib.

**Q: What if two disruptions happen simultaneously?**
A: Each zone is scored independently, so multiple disruptions in different areas produce multiple high anomaly scores. The cascade predictions would overlap, and the dashboard shows all affected zones regardless of how many root causes exist.

**Q: Why these specific 6 data sources?**
A: They cover the three major transport modes (taxi, subway, bike) plus three contextual sources (weather affects demand, 311 complaints indicate road disruptions, events cause planned disruptions). Together they give a complete picture of NYC transport.

---

*End of project reference. Keep this file alongside any Claude chat to maintain full project context.*
