"""
NeuroTraffic — HDFS Helper
============================
Owner: Sasmitha S (CB.AI.U4AID24051)

Lightweight HDFS client using WebHDFS (HTTP REST API on port 9870) for Streamlit,
and falling back to standard HDFS RPC (port 9000) or local files.
Allows the dashboard to read/write from HDFS even when running on a host machine
where Java/Hadoop environment variables are not configured.
"""

import os
import sys
import pandas as pd
import requests
import io
import pyarrow.parquet as pq

# ── HDFS Configuration ──
# WebHDFS is exposed on port 9870 in docker-compose.yml.
# From the host machine (Windows), we connect to localhost:9870.
# From inside the cluster containers, we connect to namenode:9870.
WEBHDFS_HOST = os.environ.get("NEUROTRAFFIC_HDFS_HOST", "localhost")
WEBHDFS_PORT = 9870
WEBHDFS_BASE_URL = f"http://{WEBHDFS_HOST}:{WEBHDFS_PORT}/webhdfs/v1"

# RPC HDFS URL for PySpark (e.g. hdfs://namenode:9000)
HDFS_RPC_HOST = os.environ.get("NEUROTRAFFIC_RPC_HOST", "localhost")
HDFS_RPC_PORT = 9000
HDFS_RPC_URL = f"hdfs://{HDFS_RPC_HOST}:{HDFS_RPC_PORT}"

def is_hdfs_available():
    """Check if the WebHDFS namenode is reachable."""
    try:
        url = f"{WEBHDFS_BASE_URL}/?op=GETHOMEDIRECTORY"
        r = requests.get(url, timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False

def read_hdfs_csv(hdfs_path):
    """Read a CSV file from HDFS using WebHDFS."""
    try:
        url = f"{WEBHDFS_BASE_URL}{hdfs_path}?op=OPEN"
        # WebHDFS typically redirects to the datanode. We must follow redirects.
        r = requests.get(url, allow_redirects=True, timeout=5.0)
        if r.status_code == 200:
            return pd.read_csv(io.StringIO(r.text))
        else:
            print(f"[HDFS ERROR] Status {r.status_code} for {hdfs_path}")
            return None
    except Exception as e:
        print(f"[HDFS EXCEPTION] Failed to read {hdfs_path}: {e}")
        return None

def read_hdfs_parquet(hdfs_path):
    """Read a Parquet file from HDFS using WebHDFS."""
    try:
        url = f"{WEBHDFS_BASE_URL}{hdfs_path}?op=OPEN"
        r = requests.get(url, allow_redirects=True, timeout=10.0)
        if r.status_code == 200:
            return pd.read_parquet(io.BytesIO(r.content))
        else:
            print(f"[HDFS ERROR] Status {r.status_code} for {hdfs_path}")
            return None
    except Exception as e:
        print(f"[HDFS EXCEPTION] Failed to read {hdfs_path}: {e}")
        return None

def write_hdfs_csv(df, hdfs_path):
    """Write a DataFrame as CSV to HDFS using WebHDFS."""
    try:
        csv_data = df.to_csv(index=False)
        # Step 1: Initialize creation
        url = f"{WEBHDFS_BASE_URL}{hdfs_path}?op=CREATE&overwrite=true"
        r = requests.put(url, allow_redirects=False, timeout=5.0)
        
        # Step 2: WebHDFS redirects us to the Datanode to write the data
        if r.status_code == 307:
            redirect_url = r.headers["Location"]
            # Sometimes Datanode hostname is returned as container hostname (e.g. datanode1).
            # If we are on Windows, we need to map container hostnames to localhost/IP.
            # Replace datanode container names with WebHDFS host (localhost).
            for node in ["datanode1", "datanode2", "datanode3", "namenode"]:
                if f":{node}:" in redirect_url or f"//{node}:" in redirect_url:
                    redirect_url = redirect_url.replace(node, WEBHDFS_HOST)
            
            headers = {"Content-Type": "application/octet-stream"}
            r2 = requests.put(redirect_url, data=csv_data.encode("utf-8"), headers=headers, timeout=10.0)
            return r2.status_code in [200, 201]
        return False
    except Exception as e:
        print(f"[HDFS EXCEPTION] Failed to write {hdfs_path}: {e}")
        return False

def make_hdfs_directory(hdfs_dir):
    """Create a directory in HDFS using WebHDFS."""
    try:
        url = f"{WEBHDFS_BASE_URL}{hdfs_dir}?op=MKDIRS"
        r = requests.put(url, timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False

# ── High Level Wrapper ──
def load_dataframe(local_path, hdfs_path, use_hdfs=False):
    """Load dataframe from HDFS or local filesystem depending on flag."""
    if use_hdfs:
        print(f"Loading from HDFS: {hdfs_path}")
        # Detect extension
        if hdfs_path.endswith(".csv"):
            df = read_hdfs_csv(hdfs_path)
            if df is not None:
                return df
        elif hdfs_path.endswith(".parquet") or "/data/" in hdfs_path:
            df = read_hdfs_parquet(hdfs_path)
            if df is not None:
                return df
        print("[HDFS] Load failed, trying local fallback...")
    
    # Local fallback
    if os.path.exists(local_path):
        if local_path.endswith(".csv"):
            return pd.read_csv(local_path)
        elif local_path.endswith(".parquet"):
            return pd.read_parquet(local_path)
    return None

def read_hdfs_json(hdfs_path):
    """Read a JSON file from HDFS using WebHDFS."""
    try:
        url = f"{WEBHDFS_BASE_URL}{hdfs_path}?op=OPEN"
        r = requests.get(url, allow_redirects=True, timeout=5.0)
        if r.status_code == 200:
            import json
            return json.loads(r.text)
        else:
            print(f"[HDFS ERROR] Status {r.status_code} for {hdfs_path}")
            return None
    except Exception as e:
        print(f"[HDFS EXCEPTION] Failed to read {hdfs_path}: {e}")
        return None

def load_json(local_path, hdfs_path, use_hdfs=False):
    """Load JSON from HDFS or local filesystem depending on flag."""
    if use_hdfs:
        data = read_hdfs_json(hdfs_path)
        if data is not None:
            return data
    if os.path.exists(local_path):
        import json
        with open(local_path) as f:
            return json.load(f)
    return None
