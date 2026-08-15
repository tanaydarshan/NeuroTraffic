# NeuroTraffic — Sasmitha's Module Reference

> **Sasmitha S (CB.AI.U4AID24051)**
> **Role:** Streaming pipeline + Web App Developer
> **Module:** Spark Structured Streaming + 4-page Streamlit Web Application

> **How to use this file:** Upload this along with `NeuroTraffic_Project_Reference.md` to a new Claude chat and paste the starter prompt at the bottom of this file.

---

## 1. What you're building — the simple version

You're building two things that connect to each other:

**Thing 1 — The streaming pipeline.** This is a Spark Structured Streaming program that reads incoming transport data (taxi trips, bike checkouts, turnstile counts) in near real-time, runs it through the trained ML models (Isolation Forest for detection, GBT for prediction), and writes the results (anomaly scores, cascade predictions) to a file/database.

**Thing 2 — The web app.** This is a multi-page Streamlit application that reads those results and displays them in a browser. It has 4 pages: a live map, a disruption simulator, a historical explorer, and an analytics dashboard. This is what your professor sees when you demo the project.

The streaming pipeline is the backend. The web app is the frontend. They connect through shared files (CSV/Parquet) or a SQLite database.

---

## 2. What you receive from your teammates

You don't build everything from scratch. Your three teammates produce outputs that you consume:

### From Navaneeth (Data Pipeline):
- **Clean, joined Parquet table on HDFS** — all 6 data sources combined by zone and hour
- You use this historical data for the "Historical Explorer" page and to build the data replay simulator

### From Tanay (Graph Engine):
- **Graph features file** — a CSV/Parquet with columns: `node_id`, `node_type`, `latitude`, `longitude`, `pagerank_score`, `community_id`, `degree`
- You use this on the map (to show which nodes are most critical) and on the Analytics page

### From Harsada (ML Models):
- **Trained Isolation Forest model** — saved as a Spark ML pipeline model on HDFS
- **Trained GBT model** — saved as a Spark ML pipeline model on HDFS
- **Evaluation metrics** — accuracy, precision, recall, F1, confusion matrix as CSV/JSON
- You load these models in the streaming pipeline and display the metrics on the Analytics page

---

## 3. Your two deliverables in detail

### Deliverable A: Data Replay Simulator + Spark Streaming Pipeline

Since you can't connect to live NYC turnstiles from Coimbatore, you simulate real-time data by replaying historical records.

#### A1. Data Replay Simulator

A Python script that:
1. Reads the historical combined Parquet table (from Navaneeth)
2. Sorts it by timestamp
3. Pushes rows into a "streaming input" directory on HDFS (or local filesystem) in small batches
4. Waits a few seconds between batches to simulate real-time arrival

```
Historical data (Parquet) → Replay script → /data/streaming/input/ (new CSV every 5 seconds)
```

The streaming pipeline watches this directory and processes each new file as it arrives.

**Key decisions:**
- Batch size: push 1 hour of data per batch (all zones for one hour)
- Interval: one batch every 5 seconds (so 1 hour of city data arrives every 5 seconds — 24 hours of data replays in 2 minutes)
- File format: small CSV files named with timestamp, e.g., `batch_2023-03-14_08-00.csv`

#### A2. Spark Structured Streaming Pipeline

A PySpark script that:
1. Watches the streaming input directory for new files
2. Reads each new batch
3. Applies the trained Isolation Forest model → computes anomaly score for each zone
4. If any zone's anomaly score > 0.8 → applies the trained GBT model → predicts cascade surges
5. Writes results to an output directory (CSV/Parquet) or SQLite database

```python
# Pseudocode structure
stream = spark.readStream.csv("streaming/input/")
scored = isolation_forest_model.transform(stream)
anomalies = scored.filter(col("anomaly_score") > 0.8)
predictions = gbt_model.transform(anomalies)
predictions.writeStream.csv("streaming/output/")
```

**Output format (what the app reads):**

File: `streaming/output/predictions_2023-03-14_08-00.csv`

| zone_id | timestamp | anomaly_score | is_disruption | predicted_surge_pct | affected_line | cascade_zone_ids | time_to_peak_min |
|---|---|---|---|---|---|---|---|
| 79 | 2023-03-14 08:47 | 0.94 | true | 280 | L | 79,113,234,45 | 8 |
| 113 | 2023-03-14 08:47 | 0.87 | true | 160 | L | 113,234 | 15 |
| 45 | 2023-03-14 08:47 | 0.72 | false | 0 | - | - | - |

---

### Deliverable B: Streamlit Web App (4 pages)

A multi-page Streamlit application that runs locally (`streamlit run app.py` → opens in browser at `localhost:8501`).

#### Page 1: Live Monitor (`pages/1_live_monitor.py`)

**What the user sees:**
- A full-screen map of NYC (Plotly Mapbox or Folium) showing ~2,700 nodes
- Each node is a colored dot: green (normal), yellow (anomaly score 0.5-0.8), red (anomaly score > 0.8)
- When a disruption is detected, affected zones animate from green → yellow → red
- A sidebar showing:
  - Current system status ("All normal" or "DISRUPTION DETECTED — L train")
  - Top 5 highest anomaly scores right now
  - Predicted cascade: which zones will be affected, when, by how much
  - A live-updating line chart of anomaly scores over the last hour

**How it works technically:**
- The page reads the latest file from `streaming/output/` every 5 seconds (using `st.experimental_rerun` or a timer)
- It merges the prediction data with the graph features file (to get lat/lon for each zone)
- It renders the map with Plotly's `scattermapbox` or Folium, color-coding by anomaly score
- The sidebar reads the same prediction data and displays top anomalies

**Key Streamlit components:**
```python
import streamlit as st
import plotly.express as px
import pandas as pd
import time

st.set_page_config(layout="wide")
st.title("NeuroTraffic — Live Monitor")

# Read latest predictions
predictions = pd.read_csv("streaming/output/latest.csv")
graph_nodes = pd.read_csv("data/graph_nodes.csv")
merged = predictions.merge(graph_nodes, on="zone_id")

# Map
fig = px.scatter_mapbox(merged, lat="latitude", lon="longitude",
                         color="anomaly_score", size="anomaly_score",
                         color_continuous_scale=["green", "yellow", "red"],
                         mapbox_style="open-street-map",
                         zoom=10, center={"lat": 40.75, "lon": -73.98})
st.plotly_chart(fig, use_container_width=True)

# Sidebar
with st.sidebar:
    disruptions = merged[merged["is_disruption"] == True]
    if len(disruptions) > 0:
        st.error(f"DISRUPTION DETECTED — {disruptions.iloc[0]['affected_line']} train")
        st.dataframe(disruptions[["zone_id", "predicted_surge_pct", "time_to_peak_min"]])
    else:
        st.success("All systems normal")
```

#### Page 2: Disruption Simulator (`pages/2_simulator.py`)

**What the user sees:**
- A dropdown to select a subway line (L, 6, A, C, E, 1, 2, 3, etc.)
- A slider to pick time of day (6 AM to midnight)
- A slider to pick day of week (Monday-Sunday)
- A "Simulate Disruption" button
- When clicked: the map shows the predicted cascade — zones turning red in sequence with surge percentages

**How it works technically:**
- When the user clicks "Simulate," the page creates a synthetic input matching the selected parameters
- It loads the trained GBT model (pre-saved as a pickle/joblib file for fast loading — Harsada provides this)
- It runs the model to predict surges for all nearby zones
- It displays the results on a map, with a table showing: zone, predicted surge %, estimated time to peak
- Optionally: an animated sequence showing zones changing color over time (using Plotly animation frames)

**This is the demo-killer feature.** When your professor asks "show me how it works," you open this page, select "L train" + "8:47 AM" + "Tuesday," click simulate, and the cascade appears on screen.

#### Page 3: Historical Explorer (`pages/3_explorer.py`)

**What the user sees:**
- A date range picker (start date, end date)
- A chart showing anomaly scores over that period (time-series line chart)
- Highlighted spikes where actual disruptions occurred
- A comparison table: predicted surge vs actual surge (model accuracy visualization)
- A heatmap showing which zones had the most disruptions

**How it works technically:**
- Reads the full historical combined data (Navaneeth's Parquet, pre-aggregated to daily/hourly summaries for speed)
- Reads historical MTA service alerts (ground truth disruptions) if available
- Filters by the selected date range
- Plots anomaly scores over time using Plotly line chart
- Overlays ground truth disruption events as vertical markers

#### Page 4: Analytics Dashboard (`pages/4_analytics.py`)

**What the user sees:**
- Model evaluation metrics in big number cards: Accuracy, Precision, Recall, F1
- Confusion matrix heatmap (Isolation Forest: how many disruptions correctly detected vs false alarms)
- Feature importance bar chart (which features matter most for detection: turnstile_exits > taxi_pickups > bike_checkouts > ...)
- Graph analytics section:
  - Top 10 most critical nodes by PageRank (bar chart)
  - Community map (nodes colored by community membership)
  - Network statistics: total nodes, total edges, average degree, graph density

**How it works technically:**
- Reads Harsada's evaluation metrics (CSV/JSON)
- Reads Tanay's graph features file (CSV)
- Renders everything with Plotly charts and Streamlit metrics cards

---

## 4. File/folder structure for the app

```
neurotraffic-app/
├── app.py                          # Main entry point (home page)
├── pages/
│   ├── 1_live_monitor.py           # Page 1: Live monitor with map
│   ├── 2_simulator.py              # Page 2: Disruption simulator
│   ├── 3_explorer.py               # Page 3: Historical explorer
│   └── 4_analytics.py              # Page 4: Analytics dashboard
├── data/
│   ├── graph_nodes.csv             # From Tanay: node_id, type, lat, lon, pagerank, community
│   ├── evaluation_metrics.json     # From Harsada: accuracy, precision, recall, F1
│   ├── confusion_matrix.csv        # From Harsada: confusion matrix data
│   ├── feature_importance.csv      # From Harsada: feature names and importance scores
│   ├── historical_summary.parquet  # From Navaneeth: pre-aggregated historical data
│   └── mta_service_alerts.csv      # Ground truth disruption events (download separately)
├── models/
│   ├── isolation_forest.pkl        # Trained IF model (exported by Harsada as pickle/joblib)
│   └── gbt_model.pkl               # Trained GBT model (exported by Harsada as pickle/joblib)
├── streaming/
│   ├── input/                      # Replay simulator writes batches here
│   ├── output/                     # Streaming pipeline writes predictions here
│   └── replay_simulator.py         # The data replay script
├── streaming_pipeline.py           # Spark Structured Streaming script
├── requirements.txt                # streamlit, plotly, folium, pandas, pyspark, joblib
└── README.md
```

---

## 5. Tech stack for your module

| Tool | What you use it for | Install |
|---|---|---|
| Streamlit | Web app framework — creates the 4-page app | `pip install streamlit` |
| Plotly | Interactive NYC map (scattermapbox) and all charts | `pip install plotly` |
| Folium | Alternative for the NYC map (if Plotly Mapbox needs token) | `pip install folium` |
| streamlit-folium | Embeds Folium maps in Streamlit | `pip install streamlit-folium` |
| PySpark | Spark Structured Streaming pipeline | Already installed with Spark |
| Pandas | Data manipulation in the app | `pip install pandas` |
| Joblib | Loading pre-trained ML models (pickle) | `pip install joblib` |
| SQLite | Optional — store streaming outputs for the app to query | Built into Python |

**`requirements.txt`:**
```
streamlit>=1.30.0
plotly>=5.18.0
folium>=0.15.0
streamlit-folium>=0.15.0
pandas>=2.0.0
joblib>=1.3.0
pyarrow>=14.0.0
```

---

## 6. Week-by-week plan for Sasmitha

| Week | What to do | Deliverable by end of week |
|---|---|---|
| 1-2 | Learn Streamlit basics (do a tutorial app), learn Spark Streaming concepts, read the 5 research papers | Working "hello world" Streamlit app, Spark Streaming tutorial completed |
| 3-4 | Build the data replay simulator, test it with a small sample of Navaneeth's data | `replay_simulator.py` working — pushes CSV batches to a directory |
| 5 | Get replay simulator fully working with proper batch sizes and timing | Simulator replays 24 hours of data in ~2 minutes |
| 6-7 | Build Streamlit app skeleton with 4 page stubs, build the NYC map for Page 1 | App runs with 4 pages, Page 1 shows a static map of NYC with colored dots |
| 8 | Build the Spark Streaming pipeline, connect replay simulator → pipeline → output files, wire Page 1 to read live outputs | Live Monitor page updates as streaming pipeline runs |
| 9 | Build Page 2 (Disruption Simulator) — dropdown + slider + "Simulate" button + results map | Simulator page works: user selects line + time, sees predicted cascade on map |
| 10 | Build Page 3 (Historical Explorer) — date picker + time-series chart + disruption overlay | Explorer page works: user picks dates, sees anomaly scores and disruptions |
| 11 | Build Page 4 (Analytics Dashboard) — metrics cards + confusion matrix + feature importance + graph stats. Full integration with streaming pipeline | All 4 pages working end-to-end |
| 12-13 | Polish the app (styling, loading states, error handling), test with full-scale data, record demo video | Polished app, demo video recorded |
| 14-15 | Write your section of the final report, prepare for presentation and viva | Report section complete, ready to present |

---

## 7. Key technical details to know

### How Spark Structured Streaming works (simple version)
Think of it like a conveyor belt in a factory. New files arrive in a directory. Spark watches that directory. Every time a new file appears, Spark reads it, processes it through your code (apply models, compute scores), and writes the result to an output directory. This happens automatically in a loop — you don't manually trigger it.

```
New CSV arrives in input/ → Spark reads it → Applies IF model → Applies GBT → Writes prediction to output/
(this repeats every few seconds, automatically)
```

### How the app connects to the pipeline
The streaming pipeline writes CSV files to `streaming/output/`. The Streamlit app reads the latest file from that directory. They don't talk to each other directly — they communicate through files. This is called "file-based decoupling" and it's the simplest approach.

Alternatively, you can use SQLite: the pipeline writes predictions to a SQLite database, and the app queries it. This is slightly more robust but adds complexity.

### Plotly Mapbox vs Folium for the NYC map
- **Plotly scattermapbox**: prettier, more interactive, but needs a free Mapbox access token (sign up at mapbox.com, free tier is enough). Better for animations.
- **Folium**: no token needed, works out of the box, slightly less interactive but perfectly good. Easier to get started.

Start with Folium to get something working fast. Switch to Plotly Mapbox later if you want nicer visuals.

### How the Disruption Simulator works without Spark
The simulator page doesn't run Spark in real time — that would be too slow for a button click. Instead, Harsada exports the trained GBT model as a scikit-learn compatible pickle file (using `joblib`). The Streamlit app loads this lightweight model and runs predictions directly in Python. This gives instant results when the user clicks "Simulate."

Ask Harsada to export the model in two formats:
1. Spark ML pipeline format (for the streaming pipeline)
2. Scikit-learn/joblib pickle (for the Streamlit app's simulator page)

---

## 8. Common pitfalls to avoid

1. **Don't try to run Spark inside Streamlit.** The app and the streaming pipeline are separate processes. The app is pure Python (pandas, plotly, streamlit). The streaming pipeline is PySpark running on the Hadoop cluster.

2. **Don't hardcode file paths.** Use a config file or environment variables for paths to data, models, and streaming directories. This makes it work on any machine.

3. **Start with fake/sample data.** Don't wait for Navaneeth and Harsada to finish. Create a small fake dataset (100 rows, 10 zones) with random anomaly scores and build your entire app on that. When the real data arrives, just swap the file paths.

4. **Test each page independently.** Build and test Page 1 completely before starting Page 2. Don't try to build all 4 pages at once.

5. **The demo video matters.** Your professor may not run the app themselves. A clean 3-5 minute screen recording showing: open app → show live monitor → simulate a disruption → show cascade → show historical analysis → show metrics — that's what sells the project.

---

## 9. Starter prompt for Claude

Copy this prompt and paste it into a new Claude chat after uploading both `NeuroTraffic_Project_Reference.md` and this file (`NeuroTraffic_Sasmitha_Module.md`):

```
I'm Sasmitha S (CB.AI.U4AID24051), the streaming pipeline and web app developer for the NeuroTraffic project. I've uploaded two reference documents — the full project reference and my personal module reference. Please read both completely before we start.

I need to build:
1. A data replay simulator that replays historical transport data as a real-time stream
2. A Spark Structured Streaming pipeline that reads incoming batches, applies Isolation Forest for anomaly detection, and applies GBT for cascade prediction
3. A 4-page Streamlit web app:
   - Page 1: Live Monitor (NYC map with color-coded zones updating in real time)
   - Page 2: Disruption Simulator (user picks subway line + time, sees predicted cascade)
   - Page 3: Historical Explorer (date range picker, anomaly score time-series, predicted vs actual)
   - Page 4: Analytics Dashboard (model metrics, confusion matrix, feature importance, graph stats)

I want to start by setting up the project structure and building a working Page 1 with a static NYC map using sample/fake data. I don't have the real data yet — my teammates are still working on their modules. So help me create realistic sample data (maybe 50 zones with random anomaly scores and lat/lon coordinates) and build Page 1 on that.

After Page 1 works, we'll build the other pages one by one, then the streaming pipeline, then connect everything.

I'm comfortable with Python. I've used basic Plotly before but not Streamlit or Spark Streaming. Explain new concepts clearly.
```

---

*End of Sasmitha's module reference.*
