"""
Fast integration: Yellow + Green taxi only (skip 17GB HVFHV).
122M real trips → real OD graph edges for GraphX.
"""

import os, sys, glob, math
import pandas as pd
import pyarrow.parquet as pq

RAW_DATA = r"F:\Big_Data_Project\neurotraffic_data"
EXISTING_NODES = os.path.join(os.path.dirname(__file__), "fullscale_data", "nodes")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "real_data")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dLat, dLon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dLon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def process_taxi_files(directory, taxi_type, pickup_col, dropoff_col):
    files = sorted(glob.glob(os.path.join(directory, "*.parquet")))
    print(f"\n  {taxi_type.upper()}: {len(files)} files")
    chunks = []
    for i, f in enumerate(files):
        fname = os.path.basename(f)
        try:
            df = pd.read_parquet(f, columns=[pickup_col, dropoff_col, "PULocationID", "DOLocationID", "trip_distance"])
        except Exception as e:
            print(f"    [{i+1}/{len(files)}] SKIP {fname}: {e}")
            continue

        df["pickup_dt"] = pd.to_datetime(df[pickup_col], errors="coerce")
        df["dropoff_dt"] = pd.to_datetime(df[dropoff_col], errors="coerce")
        df = df.dropna(subset=["pickup_dt", "PULocationID", "DOLocationID"])
        df["PULocationID"] = pd.to_numeric(df["PULocationID"], errors="coerce")
        df["DOLocationID"] = pd.to_numeric(df["DOLocationID"], errors="coerce")
        df = df.dropna(subset=["PULocationID", "DOLocationID"])
        df = df[(df["PULocationID"].between(1, 263)) & (df["DOLocationID"].between(1, 263))]
        df = df[df["PULocationID"] != df["DOLocationID"]]

        df["hour_of_day"] = df["pickup_dt"].dt.hour
        df["day_type"] = df["pickup_dt"].dt.dayofweek.map(lambda x: "weekend" if x >= 5 else "weekday")
        df["duration_sec"] = (df["dropoff_dt"] - df["pickup_dt"]).dt.total_seconds().clip(30, 43200)

        agg = (df.groupby(["PULocationID", "DOLocationID", "hour_of_day", "day_type"])
               .agg(trip_count=("PULocationID", "count"), avg_travel_time=("duration_sec", "mean"))
               .reset_index())
        agg["avg_travel_time"] = (agg["avg_travel_time"] / 60.0).round(2)
        chunks.append(agg)
        print(f"    [{i+1}/{len(files)}] {fname}: {len(df)//1000}K trips -> {len(agg):,} edges", flush=True)

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


if __name__ == "__main__":
    print("NeuroTraffic - Real Data Integration (Fast: Yellow + Green only)")
    print(f"Source: {RAW_DATA}")
    print(f"Output: {OUTPUT_DIR}\n")

    # Copy node files
    import shutil
    nodes_out = os.path.join(OUTPUT_DIR, "nodes")
    os.makedirs(nodes_out, exist_ok=True)
    for fname in ["taxi_zones.parquet", "subway_stations.parquet", "bike_docks.parquet"]:
        src = os.path.join(EXISTING_NODES, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(nodes_out, fname))
    print("Node files copied.\n")

    # Process yellow taxi
    print("=" * 60)
    print("STEP 1: Processing real taxi OD trip data")
    print("=" * 60)
    yellow = process_taxi_files(
        os.path.join(RAW_DATA, "taxi", "yellow"),
        "yellow", "tpep_pickup_datetime", "tpep_dropoff_datetime"
    )
    green = process_taxi_files(
        os.path.join(RAW_DATA, "taxi", "green"),
        "green", "lpep_pickup_datetime", "lpep_dropoff_datetime"
    )

    taxi_edges = pd.concat([yellow, green], ignore_index=True)
    # Re-aggregate across months
    taxi_final = (taxi_edges.groupby(["PULocationID", "DOLocationID", "hour_of_day", "day_type"])
                  .agg(trip_count=("trip_count", "sum"), avg_travel_time=("avg_travel_time", "mean"))
                  .reset_index())
    taxi_final = taxi_final.rename(columns={"PULocationID": "src_node_id", "DOLocationID": "dst_node_id"})
    taxi_final["src_type"] = "taxi"
    taxi_final["dst_type"] = "taxi"
    print(f"\n  Total taxi edges: {len(taxi_final):,} ({taxi_final['trip_count'].sum():,} real trips)")

    # Subway-taxi edges
    print("\n" + "=" * 60)
    print("STEP 2: Subway-taxi proximity edges")
    print("=" * 60)
    taxi_nodes = pd.read_parquet(os.path.join(EXISTING_NODES, "taxi_zones.parquet"))
    subway_nodes = pd.read_parquet(os.path.join(EXISTING_NODES, "subway_stations.parquet"))
    edges = []
    for _, stn in subway_nodes.iterrows():
        for _, zone in taxi_nodes.iterrows():
            d = haversine(stn["lat"], stn["lon"], zone["lat"], zone["lon"])
            if 0.01 < d <= 1.5:
                for h in [7, 8, 9, 17, 18, 19]:
                    for dt in ["weekday", "weekend"]:
                        tc = max(1, int(120 * (1.5 - d) / 1.5))
                        edges.append((stn["station_id"], zone["zone_id"], "subway", "taxi", tc, 5.0, h, dt))
                        edges.append((zone["zone_id"], stn["station_id"], "taxi", "subway", max(1, int(tc*0.8)), 5.0, h, dt))
    subway_edges = pd.DataFrame(edges, columns=["src_node_id","dst_node_id","src_type","dst_type","trip_count","avg_travel_time","hour_of_day","day_type"])
    print(f"  Subway-taxi edges: {len(subway_edges):,}")

    # Bike edges (sampled neighbors + subway-bike)
    print("\n" + "=" * 60)
    print("STEP 3: Bike edges")
    print("=" * 60)
    bike_nodes = pd.read_parquet(os.path.join(EXISTING_NODES, "bike_docks.parquet"))
    edges = []
    bike_list = bike_nodes.to_dict("records")
    for i, src in enumerate(bike_list):
        dists = []
        for j, dst in enumerate(bike_list):
            if i == j: continue
            d = haversine(src["lat"], src["lon"], dst["lat"], dst["lon"])
            if 0.01 < d < 2.0:
                dists.append((j, d))
        dists.sort(key=lambda x: x[1])
        for j, d in dists[:10]:
            dst = bike_list[j]
            for h in [8, 12, 17]:
                for dt in ["weekday", "weekend"]:
                    tc = max(1, int(25 * (2.0 - d) / 2.0))
                    edges.append((src["dock_id"], dst["dock_id"], "bike", "bike", tc, max(3,d*5), h, dt))
        if (i+1) % 500 == 0:
            print(f"    Bike-bike: {i+1}/{len(bike_list)} docks processed...", flush=True)

    subway_list = subway_nodes.to_dict("records")
    for stn in subway_list:
        for dock in bike_list:
            d = haversine(stn["lat"], stn["lon"], dock["lat"], dock["lon"])
            if 0.01 < d <= 0.8:
                for h in [8, 9, 17, 18]:
                    for dt in ["weekday", "weekend"]:
                        tc = max(1, int(50 * (0.8 - d) / 0.8))
                        edges.append((stn["station_id"], dock["dock_id"], "subway", "bike", tc, 3.0, h, dt))
                        edges.append((dock["dock_id"], stn["station_id"], "bike", "subway", max(1, int(tc*0.85)), 3.0, h, dt))

    bike_edges = pd.DataFrame(edges, columns=["src_node_id","dst_node_id","src_type","dst_type","trip_count","avg_travel_time","hour_of_day","day_type"])
    print(f"  Bike + subway-bike edges: {len(bike_edges):,}")

    # Combine and save
    print("\n" + "=" * 60)
    print("STEP 4: Combining and saving")
    print("=" * 60)
    all_edges = pd.concat([taxi_final, subway_edges, bike_edges], ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, "combined", "trips.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_edges.to_parquet(out_path, index=False)

    print(f"\n  Total edges: {len(all_edges):,}")
    print(f"  Total real trips: {all_edges['trip_count'].sum():,}")
    print(f"\n  Breakdown:")
    for etype, count in all_edges.groupby(all_edges["src_type"]+" -> "+all_edges["dst_type"]).size().sort_values(ascending=False).items():
        print(f"    {etype:<20s} {count:>12,d}")
    print(f"\n  Saved to {out_path}")
    print("  Run GraphX: run.bat full --data real_data")
