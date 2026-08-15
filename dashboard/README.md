# NeuroTraffic Dashboard

**Owner:** Sasmitha S (CB.AI.U4AID24051) · Big Data Analytics (23AID302)

A 4-page Streamlit web application for visualising NYC multi-modal transport disruptions in near real-time.

---

## Quick Start

```bash
# 1. Install dependencies
cd dashboard
pip install -r requirements.txt

# 2. Generate sample data (run once — ~15 seconds)
python data/generate_sample_data.py

# 3. Launch the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Pages

| Page | File | Description |
|------|------|-------------|
| 🏠 Home | `app.py` | Project overview, architecture diagram, data status |
| 📡 Live Monitor | `pages/1_live_monitor.py` | Real-time NYC map, auto-refresh every 5 s |
| ⚡ Simulator | `pages/2_simulator.py` | Pick any subway line + time → see cascade map |
| 📊 Explorer | `pages/3_explorer.py` | Date range picker, 6-month trend charts, heatmap |
| 🧠 Analytics | `pages/4_analytics.py` | Model metrics, confusion matrix, feature importance, graph stats |

---

## Live Streaming Demo

Open **two terminals:**

**Terminal 1 — Launch the app:**
```bash
cd dashboard
streamlit run app.py
```

**Terminal 2 — Start the replay simulator:**
```bash
cd dashboard
python streaming/replay_simulator.py --speed 5 --loop
```

The Live Monitor page will refresh every 5 seconds showing the replaying data.

### Simulator options
```
python streaming/replay_simulator.py --help

--speed N         Seconds between batches (default: 5)
--loop            Replay infinitely when data ends
--from YYYY-MM-DD Start from a specific date (default: 2023-01-01)
```

---

## Spark Streaming Pipeline (requires PySpark)

```bash
# Run INSTEAD of replay_simulator.py when PySpark is available
spark-submit --master local[*] streaming_pipeline.py

# Then in a separate terminal, push batches with the replay simulator:
python streaming/replay_simulator.py --speed 5 --loop
```

---

## File Structure

```
dashboard/
├── app.py                          ← Home page (run this with streamlit)
├── config.py                       ← All paths and constants
├── streaming_pipeline.py           ← Spark Structured Streaming (PySpark)
├── requirements.txt
├── pages/
│   ├── 1_live_monitor.py           ← Page 1: Live NYC map
│   ├── 2_simulator.py              ← Page 2: Disruption cascade simulator
│   ├── 3_explorer.py               ← Page 3: Historical trend explorer
│   └── 4_analytics.py              ← Page 4: Model metrics + graph analytics
├── data/
│   ├── generate_sample_data.py     ← Run this first to generate sample data
│   ├── graph_nodes.csv             ← 263 NYC taxi zones (from Tanay)
│   ├── evaluation_metrics.json     ← Model metrics (from Harsada)
│   ├── feature_importance.csv      ← GBT feature weights (from Harsada)
│   ├── confusion_matrix.csv        ← 2×2 confusion matrix (from Harsada)
│   └── historical_predictions.parquet  ← 6-month history (from Navaneeth)
├── models/
│   └── (put gbt_model.pkl here when Harsada exports it)
└── streaming/
    ├── replay_simulator.py         ← Pushes historical data as a stream
    ├── input/                      ← Simulator writes batch CSVs here
    └── output/
        ├── latest.csv              ← Live Monitor reads this
        └── history/                ← Archived predictions
```

---

## Swapping in Real Data

When your teammates finish their modules:

1. **Tanay's graph features** → replace `data/graph_nodes.csv` with the Parquet from `real_data/graph_features/features.parquet` (convert to CSV or update `config.py` to read Parquet directly)

2. **Navaneeth's combined table** → update `config.HISTORICAL_PARQUET` in `config.py` to point to the HDFS path

3. **Harsada's model outputs** → replace the JSON/CSV files in `data/` with the real evaluation outputs; put the exported `gbt_model.pkl` in `models/`

No other code changes needed — the app reads everything through `config.py`.

---

## Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Streamlit | ≥ 1.35 | Web app framework |
| Plotly | ≥ 5.22 | Interactive maps and charts |
| Pandas | ≥ 2.0 | Data manipulation |
| PyArrow | ≥ 14.0 | Parquet file reading |
| NumPy | ≥ 1.24 | Numerical computation |
| scikit-learn | ≥ 1.3 | Lightweight GBT for simulator |
| PySpark | 3.5.x | Spark Structured Streaming pipeline |

Map tiles: **Plotly carto-darkmatter** (free, no API token required)
