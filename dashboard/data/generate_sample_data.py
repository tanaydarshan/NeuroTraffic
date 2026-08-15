"""
NeuroTraffic — Sample Data Generator
=====================================
Owner: Sasmitha S (CB.AI.U4AID24051)

Generates synthetic data files in the exact same schemas as the real pipeline
outputs. Run this script once before launching the Streamlit app.

Usage:
    python dashboard/data/generate_sample_data.py

Outputs (all in dashboard/data/):
    graph_nodes.csv            — 263 NYC taxi zones with graph features
    evaluation_metrics.json    — model performance metrics
    feature_importance.csv     — GBT feature importance scores
    confusion_matrix.csv       — 2x2 confusion matrix
    historical_predictions.parquet — 6 months of hourly zone predictions

Schema matches Navaneeth's combined_v3 (38 cols) + Harsada's ML outputs.
When Docker HDFS is available, use integrate_real_data.py instead.
"""

import pandas as pd
import numpy as np
import json
import os
import random
from datetime import datetime, timedelta

# Reproducibility
np.random.seed(42)
random.seed(42)

OUT_DIR = os.path.dirname(__file__)

# ── Real NYC Taxi Zone names (all 263 zones) ──
NYC_ZONE_NAMES = [
    "Newark Airport", "Jamaica Bay", "Allerton/Pelham Gardens", "Alphabet City",
    "Arden Heights", "Arrochar/Fort Wadsworth", "Astoria", "Astoria Park",
    "Auburndale", "Baisley Park", "Bath Beach", "Battery Park",
    "Battery Park City", "Bay Ridge", "Bay Terrace/Fort Totten", "Bayside",
    "Bedford", "Bedford Park", "Bellerose", "Belmont",
    "Bensonhurst East", "Bensonhurst West", "Bloomfield/Emerson Hill", "Bloomingdale",
    "Boerum Hill", "Borough Park", "Breezy Point/Fort Tilden/Riis Beach", "Briarwood/Jamaica Hills",
    "Brighton Beach", "Broad Channel", "Bronx Park", "Bronxdale",
    "Brooklyn Heights", "Brooklyn Navy Yard", "Brownsville", "Bushwick North",
    "Bushwick South", "East Elmhurst", "East Flatbush/Farragut", "East Flatbush/Remsen Village",
    "East Flushing", "East Harlem North", "East Harlem South", "East New York",
    "East New York/Pennsylvania Avenue", "East Tremont", "East Village", "East Williamsburg",
    "Eastchester", "Elmhurst", "Elmhurst/Maspeth", "Eltingville/Annadale/Prince's Bay",
    "Erasmus", "Far Rockaway", "Flatbush/Ditmas Park", "Flatlands",
    "Flushing", "Flushing Meadows-Corona Park", "Fordham South", "Forest Hills",
    "Forest Park/Highland Park", "Fort Greene", "Fresh Meadows", "Freshkills Park",
    "Garment District", "Glen Oaks", "Glendale", "Governor's Island/Ellis Island/Liberty Island",
    "Gowanus", "Gramercy", "Gravesend", "Great Kills",
    "Great Kills Park", "Greenwich Village North", "Greenwich Village South", "Greenpoint",
    "Heartland Village/Todt Hill", "Highbridge", "Highbridge Park", "Hillcrest/Pomonok",
    "Hollis", "Homecrest", "Howard Beach", "Hudson Sq",
    "Hunts Point", "Inwood", "Inwood Hill Park", "Jackson Heights",
    "Jamaica", "Jamaica Estates", "JFK Airport", "Kensington",
    "Kew Gardens", "Kew Gardens Hills", "Kingsbridge Heights", "LaGuardia Airport",
    "Laurelton", "Lenox Hill East", "Lenox Hill West", "Lincoln Square East",
    "Lincoln Square West", "Little Italy/NoLiTa", "Long Island City/Hunters Point", "Long Island City/Queens Plaza",
    "Longwood", "Lower East Side", "Madison", "Manhattan Beach",
    "Manhattan Valley", "Manhattanville", "Marine Park/Floyd Bennett Field", "Marine Park/Mill Basin",
    "Maspeth", "Meatpacking/West Village West", "Melrose South", "Middle Village",
    "Midtown Center", "Midtown East", "Midtown North", "Midtown South",
    "Midwood", "Mill Basin", "Morningside Heights", "Morrisania/Melrose",
    "Mott Haven/Port Morris", "Mount Hope", "Murray Hill", "Murray Hill-Queens",
    "New Dorp/Midland Beach", "New Springville/Bloomfield/Travis", "North Corona", "Norwood",
    "Oakland Gardens", "Oakwood", "Ocean Hill", "Ocean Parkway South",
    "Old Astoria", "Old Howard Beach", "Ozone Park", "Park Slope",
    "Parkchester", "Pelham Bay", "Pelham Bay Park", "Pelham Pkwy",
    "Penn Station/Madison Sq West", "Port Richmond", "Prospect-Lefferts Gardens", "Prospect Heights",
    "Prospect Park", "Queens Village", "Queensboro Hill", "Queensbridge/Ravenswood/Long Island City",
    "Randalls Island", "Red Hook", "Rego Park", "Richmond Hill",
    "Ridgewood", "Rikers Island", "Riverdale/North Riverdale/Fieldston", "Rockaway Park",
    "Roosevelt Island", "Rosedale", "Rossville/Woodrow", "Saint Albans",
    "Saint George/New Brighton", "Saint Michaels Cemetery/Woodside", "Schuylerville/Edgewater Park", "Seaport",
    "Sheepshead Bay", "SoHo", "Soundview/Bruckner", "Soundview/Castle Hill",
    "South Beach/Dongan Hills", "South Jamaica", "South Ozone Park", "South Williamsburg",
    "Southeast Bronx", "Springfield Gardens North", "Springfield Gardens South", "Spuyten Duyvil/Kingsbridge",
    "St. Albans", "Stapleton", "Starrett City", "Steinway",
    "Stuyvesant Heights", "Stuyvesant Town/Peter Cooper Village", "Sunset Park East", "Sunset Park West",
    "Sunnyside", "Sutton Place/Turtle Bay North", "Times Sq/Theatre District", "TriBeCa/Civic Center",
    "Two Bridges/Seward Park", "Union Sq", "University Heights/Morris Heights", "Upper East Side North",
    "Upper East Side South", "Upper West Side North", "Upper West Side South", "Van Cortlandt Park",
    "Van Cortlandt Village", "Van Nest/Morris Park", "Vinegar Hill/DUMBO/Fulton Ferry", "Wakefield/Edenwald/Pratt",
    "Washington Heights North", "Washington Heights South", "West Brighton", "West Concourse",
    "West Farms/Bronx River", "West Village", "Westchester Village/Unionport", "Whitestone",
    "Willets Point", "Williamsbridge/Olinville", "Williamsburg (North Side)", "Williamsburg (South Side)",
    "Windsor Terrace", "Woodhaven", "Woodlawn/Wakefield", "Woodside",
    "World Trade Center", "Yorkville East", "Yorkville West", "NV",
    "Outside of NYC", "Unknown", "N/A", "NA",
    "East New York Zone 2", "Brownsville Zone 2", "Canarsie", "Flatbush North",
    "Crown Heights North", "Crown Heights South", "Carroll Gardens", "Cobble Hill",
    "Clinton Hill", "Bed-Stuy West", "Bed-Stuy East", "Bushwick Central",
    "Ridgewood South", "Maspeth South", "Corona", "Jackson Heights South",
    "Rego Park South", "Forest Hills South", "Jamaica South", "Richmond Hill South"
]

# Pad to exactly 263 zones if the list is shorter
while len(NYC_ZONE_NAMES) < 263:
    NYC_ZONE_NAMES.append(f"Zone {len(NYC_ZONE_NAMES) + 1}")
NYC_ZONE_NAMES = NYC_ZONE_NAMES[:263]

# Boroughs mapped by zone ranges (approximate real mapping)
def get_borough(zone_id):
    if zone_id <= 5:     return "EWR"
    if zone_id <= 10:    return "Queens"
    if zone_id <= 80:    return "Manhattan"
    if zone_id <= 130:   return "Brooklyn"
    if zone_id <= 180:   return "Queens"
    if zone_id <= 230:   return "Bronx"
    return "Staten Island"

# Approximate lat/lon centroids per borough
BOROUGH_CENTERS = {
    "Manhattan":     (40.7831, -73.9712, 0.025, 0.025),
    "Brooklyn":      (40.6501, -73.9496, 0.055, 0.045),
    "Queens":        (40.7282, -73.7949, 0.070, 0.060),
    "Bronx":         (40.8448, -73.8648, 0.040, 0.040),
    "Staten Island": (40.5795, -74.1502, 0.040, 0.040),
    "EWR":           (40.6895, -74.1745, 0.005, 0.005),
}

# Subway lines per zone (rough assignment)
SUBWAY_LINES = ["L", "A", "C", "E", "1", "2", "3", "4", "5", "6", "N", "Q", "R", "W", "J", "Z", "7", "G", "B", "D", "F", "M"]

def get_primary_line(zone_id):
    return SUBWAY_LINES[zone_id % len(SUBWAY_LINES)]

# Community IDs (8 communities like real Label Propagation result)
def get_community(zone_id):
    return zone_id % 8

# ============================================================
# 1. graph_nodes.csv
# ============================================================
print("[1/5] Generating graph_nodes.csv ...")

nodes = []
for zone_id in range(1, 264):
    borough = get_borough(zone_id)
    clat, clon, dlat, dlon = BOROUGH_CENTERS[borough]
    lat = clat + np.random.uniform(-dlat, dlat)
    lon = clon + np.random.uniform(-dlon, dlon)
    
    # Busier zones (Midtown, Downtown) get higher PageRank
    is_central = borough == "Manhattan" and zone_id in range(40, 90)
    page_rank = np.random.beta(2 if is_central else 1, 5) * (0.05 if is_central else 0.02)
    
    nodes.append({
        "zone_id":      zone_id,
        "vertex_id":    zone_id,
        "node_type":    "taxi_zone",
        "name":         NYC_ZONE_NAMES[zone_id - 1],
        "lat":          round(lat, 6),
        "lon":          round(lon, 6),
        "borough":      borough,
        "capacity":     random.randint(200, 5000),
        "page_rank":    round(page_rank, 6),
        "community_id": get_community(zone_id),
        "in_degree":    random.randint(5, 80),
        "out_degree":   random.randint(5, 80),
        "primary_line": get_primary_line(zone_id),
    })

nodes_df = pd.DataFrame(nodes)
nodes_df.to_csv(os.path.join(OUT_DIR, "graph_nodes.csv"), index=False)
print(f"    [OK] {len(nodes_df)} zones written to graph_nodes.csv")

# ============================================================
# 2. evaluation_metrics.json
# ============================================================
print("[2/5] Generating evaluation_metrics.json ...")

metrics = {
    "isolation_forest": {
        "model": "Z-Score Isolation Forest (PySpark)",
        "threshold": 0.80,
        "precision": 0.8743,
        "recall": 0.8219,
        "f1_score": 0.8473,
        "accuracy": 0.9612,
        "auc_roc": 0.9187,
        "true_positives": 1847,
        "true_negatives": 89341,
        "false_positives": 265,
        "false_negatives": 401,
        "total_samples": 91854,
        "anomaly_rate": 0.0243
    },
    "gbt_classifier": {
        "model": "Gradient Boosted Trees (PySpark MLlib)",
        "max_iter": 50,
        "max_depth": 5,
        "precision": 0.9021,
        "recall": 0.8654,
        "f1_score": 0.8834,
        "accuracy": 0.9738,
        "auc_roc": 0.9512,
        "true_positives": 1947,
        "true_negatives": 91234,
        "false_positives": 214,
        "false_negatives": 303,
        "total_samples": 93698,
        "anomaly_rate": 0.0243
    },
    "data_summary": {
        "total_zone_hours": 6900000,
        "zones": 263,
        "date_range": "2023-01-01 to 2023-06-30",
        "data_sources": ["Yellow Taxi", "Subway Turnstiles", "Citi Bike", "Weather", "311 Complaints", "NYC Events"]
    }
}

with open(os.path.join(OUT_DIR, "evaluation_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("    [OK] evaluation_metrics.json written")

# ============================================================
# 3. feature_importance.csv
# ============================================================
print("[3/5] Generating feature_importance.csv ...")

features = [
    ("z_subway",                 0.1872, "Anomaly"),
    ("taxi_to_subway_ratio",     0.1543, "Cross-modal"),
    ("page_rank",                0.1234, "Graph"),
    ("subway_ridership",         0.0987, "Transport"),
    ("z_pickups",                0.0876, "Anomaly"),
    ("taxi_pickups",             0.0754, "Transport"),
    ("bike_to_subway_ratio",     0.0643, "Cross-modal"),
    ("is_rush_hour",             0.0521, "Temporal"),
    ("z_bikes",                  0.0487, "Anomaly"),
    ("in_degree",                0.0432, "Graph"),
    ("out_degree",               0.0398, "Graph"),
    ("community_id",             0.0321, "Graph"),
    ("temperature_c",            0.0287, "Weather"),
    ("is_rain",                  0.0198, "Weather"),
    ("hour_of_day",              0.0187, "Temporal"),
    ("bike_starts",              0.0143, "Transport"),
    ("complaint_count",          0.0121, "External"),
    ("event_count",              0.0098, "External"),
    ("is_weekend",               0.0087, "Temporal"),
    ("precipitation_mm",         0.0076, "Weather"),
    ("complaint_density",        0.0065, "Cross-modal"),
    ("total_surface_transport",  0.0054, "Cross-modal"),
    ("is_snow",                  0.0043, "Weather"),
    ("wind_speed_kmh",           0.0032, "Weather"),
    ("humidity_pct",             0.0021, "Weather"),
    ("taxi_dropoffs",            0.0012, "Transport"),
    ("bike_ends",                0.0010, "Transport"),
    ("is_extreme_cold",          0.0008, "Weather"),
    ("is_extreme_heat",          0.0006, "Weather"),
    ("max_complaint_severity",   0.0004, "External"),
]

fi_df = pd.DataFrame(features, columns=["feature_name", "importance", "category"])
fi_df.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)
print(f"    [OK] {len(fi_df)} features written to feature_importance.csv")

# ============================================================
# 4. confusion_matrix.csv
# ============================================================
print("[4/5] Generating confusion_matrix.csv ...")

cm_data = [
    {"actual": "Normal",     "predicted": "Normal",     "count": 91234, "model": "GBT"},
    {"actual": "Normal",     "predicted": "Disruption", "count": 214,   "model": "GBT"},
    {"actual": "Disruption", "predicted": "Normal",     "count": 303,   "model": "GBT"},
    {"actual": "Disruption", "predicted": "Disruption", "count": 1947,  "model": "GBT"},
    {"actual": "Normal",     "predicted": "Normal",     "count": 89341, "model": "IsolationForest"},
    {"actual": "Normal",     "predicted": "Disruption", "count": 265,   "model": "IsolationForest"},
    {"actual": "Disruption", "predicted": "Normal",     "count": 401,   "model": "IsolationForest"},
    {"actual": "Disruption", "predicted": "Disruption", "count": 1847,  "model": "IsolationForest"},
]
cm_df = pd.DataFrame(cm_data)
cm_df.to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"), index=False)
print("    [OK] confusion_matrix.csv written")

# ============================================================
# 5. historical_predictions.parquet
# ============================================================
print("[5/5] Generating historical_predictions.parquet (this takes ~10 sec) ...")

START_DATE = datetime(2023, 1, 1)
END_DATE   = datetime(2023, 6, 30)

# We'll generate daily summaries (one row per zone per day) — not hourly —
# to keep the file small (~57k rows).  The app aggregates to hourly on demand.
all_rows = []
current_date = START_DATE
zone_ids = list(range(1, 264))

# Pre-load node info for realistic generation
node_info = {row["zone_id"]: row for row in nodes}

# Known disruption events (synthetic but realistic)
DISRUPTION_EVENTS = {
    "2023-01-17": {"lines": ["L"],    "zones": list(range(40, 60))},
    "2023-02-03": {"lines": ["A","C"],"zones": list(range(60, 90))},
    "2023-02-28": {"lines": ["4","5"],"zones": list(range(1, 30))},
    "2023-03-14": {"lines": ["L"],    "zones": list(range(40, 65))},
    "2023-04-07": {"lines": ["N","Q"],"zones": list(range(100, 130))},
    "2023-04-22": {"lines": ["1"],    "zones": list(range(50, 80))},
    "2023-05-11": {"lines": ["7"],    "zones": list(range(150, 180))},
    "2023-06-01": {"lines": ["J","Z"],"zones": list(range(120, 150))},
    "2023-06-15": {"lines": ["G"],    "zones": list(range(90, 115))},
}

print("    Generating rows (one per zone per day x 6 months) ...")
while current_date <= END_DATE:
    date_str = current_date.strftime("%Y-%m-%d")
    is_disruption_day = date_str in DISRUPTION_EVENTS
    disruption_info = DISRUPTION_EVENTS.get(date_str, {})
    disrupted_zones = set(disruption_info.get("zones", []))
    affected_line = disruption_info.get("lines", ["-"])[0] if disruption_info else "-"
    
    is_weekend = current_date.weekday() >= 5
    
    for zone_id in zone_ids:
        node = node_info[zone_id]
        borough = node["borough"]
        page_rank = node["page_rank"]
        
        # Base activity scaled by borough and time
        rush_mult  = 1.0 if not is_weekend else 0.65
        base_taxi  = int(np.random.poisson(max(1, page_rank * 50000 * rush_mult)))
        base_sub   = int(np.random.poisson(max(1, page_rank * 80000 * rush_mult)))
        base_bike  = int(np.random.poisson(max(1, page_rank * 20000 * rush_mult)))
        
        # Weather (random per day)
        temp_c = np.random.normal(10 if current_date.month < 4 else 20, 5)
        is_rain = random.random() < 0.25
        is_snow = random.random() < (0.10 if current_date.month <= 2 else 0.01)
        precip  = np.random.exponential(3) if (is_rain or is_snow) else 0.0
        
        # Z-scores (nominal)
        z_sub    = np.random.normal(0, 0.8)
        z_pick   = np.random.normal(0, 0.8)
        z_bikes  = np.random.normal(0, 0.8)
        anomaly_score = np.clip(abs(z_sub) / 6 + abs(z_pick) / 8, 0, 1)
        is_anomaly = 0
        prediction = 0
        
        # Disruption logic
        if is_disruption_day and zone_id in disrupted_zones:
            # Subway drops, taxi/bike surge
            base_sub  = int(base_sub * np.random.uniform(0.1, 0.35))
            base_taxi = int(base_taxi * np.random.uniform(2.2, 3.8))
            base_bike = int(base_bike * np.random.uniform(1.8, 3.0))
            z_sub   = np.random.normal(-3.5, 0.5)
            z_pick  = np.random.normal(3.2, 0.5)
            z_bikes = np.random.normal(2.8, 0.5)
            anomaly_score = np.clip(0.70 + np.random.uniform(0, 0.30), 0, 1)
            is_anomaly = 1
            prediction = 1
        
        # Cascade zones (zones adjacent to disrupted zones with partial effect)
        if is_disruption_day and zone_id in [z + 10 for z in disrupted_zones if z + 10 <= 263]:
            anomaly_score = np.clip(0.45 + np.random.uniform(0, 0.30), 0, 1)
            is_anomaly = int(anomaly_score > 0.8)
            prediction = is_anomaly
        
        surge_pct = int((anomaly_score - 0.5) * 400) if anomaly_score > 0.5 else 0
        
        wind_speed = round(np.random.uniform(5, 35), 1)
        humidity = round(np.random.uniform(40, 90), 1)
        visibility = round(np.random.uniform(5000, 16000), 0)
        is_heavy = int(is_rain and precip > 8)
        is_low_vis = int(visibility < 3000)
        is_high_wind_val = int(wind_speed > 25)

        all_rows.append({
            "zone_id":            zone_id,
            "date":               date_str,
            "hourly_timestamp":   f"{date_str} 08:00:00",
            "year":               current_date.year,
            "month":              current_date.month,
            "day_of_week":        current_date.weekday(),
            "hour_of_day":        8,
            "is_weekend":         int(is_weekend),
            "is_rush_hour":       1,
            "borough":            borough,
            "zone_name":          node["name"],
            "taxi_pickups":       max(0, base_taxi),
            "taxi_dropoffs":      max(0, int(base_taxi * np.random.uniform(0.8, 1.2))),
            "avg_fare":           round(np.random.uniform(8, 45), 2),
            "avg_trip_distance":  round(np.random.uniform(0.5, 12), 2),
            "avg_trip_duration_sec": round(np.random.uniform(300, 2400), 0),
            "subway_ridership":   max(0, base_sub),
            "subway_transfers":   max(0, int(base_sub * np.random.uniform(0.05, 0.15))),
            "subway_stations_active": random.randint(1, 8),
            "bike_starts":        max(0, base_bike),
            "bike_ends":          max(0, int(base_bike * np.random.uniform(0.85, 1.15))),
            "temperature_c":      round(temp_c, 1),
            "humidity_pct":       humidity,
            "precipitation_mm":   round(precip, 2),
            "wind_speed_kmh":     wind_speed,
            "visibility_m":       visibility,
            "is_rain":            int(is_rain),
            "is_heavy_rain":      is_heavy,
            "is_snow":            int(is_snow),
            "is_extreme_cold":    int(temp_c < -5),
            "is_extreme_heat":    int(temp_c > 35),
            "is_low_visibility":  is_low_vis,
            "is_high_wind":       is_high_wind_val,
            "complaint_count":    int(np.random.poisson(max(1, page_rank * 2000))),
            "max_complaint_severity": random.randint(1, 5),
            "event_count":        int(np.random.poisson(1.2)),
            "max_event_severity": random.randint(0, 3),
            "total_streets_affected": random.randint(0, 5),
            "page_rank":          round(page_rank, 6),
            "community_id":       node["community_id"],
            "in_degree":          node["in_degree"],
            "out_degree":         node["out_degree"],
            "z_subway":           round(z_sub, 4),
            "z_pickups":          round(z_pick, 4),
            "z_bikes":            round(z_bikes, 4),
            "anomaly_score":      round(anomaly_score, 4),
            "is_anomaly":         is_anomaly,
            "prediction":         prediction,
            "affected_line":      affected_line if is_anomaly else "-",
            "predicted_surge_pct": surge_pct,
            "time_to_peak_min":   random.randint(5, 25) if is_anomaly else 0,
        })
    
    current_date += timedelta(days=1)

hist_df = pd.DataFrame(all_rows)
hist_df["date"] = pd.to_datetime(hist_df["date"])
hist_df["hourly_timestamp"] = pd.to_datetime(hist_df["hourly_timestamp"])

parquet_path = os.path.join(OUT_DIR, "historical_predictions.parquet")
hist_df.to_parquet(parquet_path, index=False)
print(f"    [OK] {len(hist_df):,} rows written to historical_predictions.parquet")

# Also generate a latest.csv for the live monitor (streaming output mock)
print("\n[BONUS] Generating streaming/output/latest.csv ...")
latest_dir = os.path.join(os.path.dirname(OUT_DIR), "streaming", "output")
os.makedirs(latest_dir, exist_ok=True)

# Simulate a disruption right now
latest_zones = []
for zone_id in zone_ids:
    node = node_info[zone_id]
    # A few disrupted zones for demo purposes
    is_disrupted = zone_id in range(42, 58)
    is_cascade   = zone_id in range(58, 68)
    
    if is_disrupted:
        anom = round(np.random.uniform(0.82, 0.98), 3)
        line = "L"
        surge = random.randint(180, 320)
        ttl   = random.randint(5, 15)
    elif is_cascade:
        anom = round(np.random.uniform(0.52, 0.79), 3)
        line = "L"
        surge = random.randint(60, 150)
        ttl   = random.randint(12, 25)
    else:
        anom = round(np.random.uniform(0.02, 0.45), 3)
        line = "-"
        surge = 0
        ttl   = 0

    latest_zones.append({
        "zone_id":           zone_id,
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "anomaly_score":     anom,
        "is_disruption":     int(anom > 0.8),
        "predicted_surge_pct": surge,
        "affected_line":     line,
        "cascade_zone_ids":  ",".join(str(z) for z in range(zone_id, min(zone_id+4, 264))) if is_disrupted else "",
        "time_to_peak_min":  ttl,
        "z_subway":          round(np.random.normal(-3.2 if is_disrupted else 0, 0.5), 3),
        "z_pickups":         round(np.random.normal(3.1 if is_disrupted else 0, 0.5), 3),
        "taxi_pickups":      random.randint(100, 500),
        "subway_ridership":  random.randint(50, 800) if not is_disrupted else random.randint(5, 80),
        "bike_starts":       random.randint(20, 200),
    })

latest_df = pd.DataFrame(latest_zones)
latest_df.to_csv(os.path.join(latest_dir, "latest.csv"), index=False)
print(f"    [OK] latest.csv written ({len(latest_df)} zones)")

print("\n" + "=" * 55)
print("  [SUCCESS] All sample data generated successfully!")
print("=" * 55)
print(f"\n  Files in {OUT_DIR}:")
for f in os.listdir(OUT_DIR):
    fpath = os.path.join(OUT_DIR, f)
    if os.path.isfile(fpath):
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:8.1f} KB")
