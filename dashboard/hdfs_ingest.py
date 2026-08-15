"""
NeuroTraffic — HDFS Ingest Script
===================================
Owner: Sasmitha S (CB.AI.U4AID24051)
Course: Big Data Analytics (23AID302)

Bootstraps HDFS with all the data files that the dashboard and streaming
pipeline need.  Run this ONCE after the Hadoop cluster is up, or whenever
you want to refresh the HDFS data from local files.

Usage:
    python hdfs_ingest.py [--check] [--force]

Options:
    --check   : Just check what files are in HDFS (no upload)
    --force   : Re-upload even if the file already exists in HDFS

Requirements:
    pip install requests pandas pyarrow

The Hadoop cluster must be running:
    cd Navaneeth && docker-compose up -d
    (wait ~30 s for namenode to leave safe mode)
"""

import os
import sys
import time
import argparse
import requests
import io

# ── Path setup ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import config
import hdfs_helper

# ══════════════════════════════════════════════════════════
# Helper: upload a local file to HDFS via WebHDFS
# ══════════════════════════════════════════════════════════

def hdfs_file_exists(hdfs_path):
    """Return True if the path exists in HDFS."""
    try:
        url = f"{hdfs_helper.WEBHDFS_BASE_URL}{hdfs_path}?op=GETFILESTATUS"
        r = requests.get(url, timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def hdfs_mkdir(hdfs_dir):
    """Create a directory tree in HDFS (equivalent to mkdir -p)."""
    try:
        url = f"{hdfs_helper.WEBHDFS_BASE_URL}{hdfs_dir}?op=MKDIRS"
        r = requests.put(url, timeout=5.0)
        return r.status_code == 200
    except Exception as e:
        print(f"    [WARN] mkdir failed for {hdfs_dir}: {e}")
        return False


def upload_file(local_path, hdfs_path, force=False):
    """
    Upload a single local file to HDFS using WebHDFS two-step PUT.
    Returns True on success.
    """
    if not os.path.exists(local_path):
        print(f"    [SKIP] Local file not found: {local_path}")
        return False

    if not force and hdfs_file_exists(hdfs_path):
        size_kb = os.path.getsize(local_path) / 1024
        print(f"    [EXISTS] {os.path.basename(hdfs_path)}  ({size_kb:.0f} KB)  — skip (use --force to overwrite)")
        return True

    size_kb = os.path.getsize(local_path) / 1024
    print(f"    [UPLOAD] {os.path.basename(local_path)} → {hdfs_path}  ({size_kb:.0f} KB) ...", end="", flush=True)

    try:
        with open(local_path, "rb") as f:
            data = f.read()

        # Step 1: Initiate CREATE
        url = f"{hdfs_helper.WEBHDFS_BASE_URL}{hdfs_path}?op=CREATE&overwrite=true"
        r = requests.put(url, allow_redirects=False, timeout=10.0)

        if r.status_code == 307:
            # Step 2: Follow redirect to DataNode
            redirect_url = r.headers["Location"]
            # Fix container hostnames → localhost when running on the host machine
            for node in ["datanode1", "datanode2", "datanode3", "namenode"]:
                redirect_url = redirect_url.replace(f"//{node}:", f"//{hdfs_helper.WEBHDFS_HOST}:")

            headers = {"Content-Type": "application/octet-stream"}
            r2 = requests.put(redirect_url, data=data, headers=headers, timeout=60.0)
            if r2.status_code in [200, 201]:
                print(" OK")
                return True
            else:
                print(f" FAILED (datanode {r2.status_code})")
                return False
        else:
            print(f" FAILED (namenode {r.status_code}: {r.text[:120]})")
            return False

    except Exception as e:
        print(f" EXCEPTION: {e}")
        return False


def upload_bytes(data_bytes, hdfs_path, force=False, label=""):
    """Upload raw bytes to HDFS."""
    if not force and hdfs_file_exists(hdfs_path):
        print(f"    [EXISTS] {os.path.basename(hdfs_path)} — skip")
        return True

    print(f"    [UPLOAD] {label or os.path.basename(hdfs_path)} → {hdfs_path} ({len(data_bytes)/1024:.0f} KB) ...", end="", flush=True)
    try:
        url = f"{hdfs_helper.WEBHDFS_BASE_URL}{hdfs_path}?op=CREATE&overwrite=true"
        r = requests.put(url, allow_redirects=False, timeout=10.0)
        if r.status_code == 307:
            redirect_url = r.headers["Location"]
            for node in ["datanode1", "datanode2", "datanode3", "namenode"]:
                redirect_url = redirect_url.replace(f"//{node}:", f"//{hdfs_helper.WEBHDFS_HOST}:")
            headers = {"Content-Type": "application/octet-stream"}
            r2 = requests.put(redirect_url, data=data_bytes, headers=headers, timeout=60.0)
            if r2.status_code in [200, 201]:
                print(" OK")
                return True
            else:
                print(f" FAILED ({r2.status_code})")
                return False
        else:
            print(f" FAILED ({r.status_code})")
            return False
    except Exception as e:
        print(f" EXCEPTION: {e}")
        return False


# ══════════════════════════════════════════════════════════
# HDFS directory structure
# ══════════════════════════════════════════════════════════

HDFS_DIRS = [
    "/data",
    "/data/graph_features",
    "/data/predictions",
    "/data/combined",
    "/data/streaming",
    "/data/streaming/input",
    "/data/streaming/output",
    "/data/streaming/output/history",
    "/data/local",                    # mirror of the local mount
]

# Mapping: (local_path, hdfs_path)
FILE_MAP = [
    (config.GRAPH_NODES_CSV,         config.HDFS_GRAPH_NODES_CSV),
    (config.EVALUATION_METRICS_JSON,  config.HDFS_EVALUATION_METRICS_JSON),
    (config.FEATURE_IMPORTANCE_CSV,   config.HDFS_FEATURE_IMPORTANCE_CSV),
    (config.CONFUSION_MATRIX_CSV,     config.HDFS_CONFUSION_MATRIX_CSV),
    (config.HISTORICAL_PARQUET,       config.HDFS_HISTORICAL_PARQUET),
    (config.LATEST_CSV,               config.HDFS_LATEST_CSV),
]


# ══════════════════════════════════════════════════════════
# Check: list current HDFS contents
# ══════════════════════════════════════════════════════════

def list_hdfs(hdfs_dir):
    """Print a directory listing of an HDFS path."""
    try:
        url = f"{hdfs_helper.WEBHDFS_BASE_URL}{hdfs_dir}?op=LISTSTATUS"
        r = requests.get(url, timeout=5.0)
        if r.status_code == 200:
            files = r.json().get("FileStatuses", {}).get("FileStatus", [])
            if not files:
                print(f"    (empty)")
            for f in files:
                ftype  = "DIR " if f["type"] == "DIRECTORY" else "FILE"
                size   = f"{f['length']/1024:.1f} KB" if f["type"] == "FILE" else ""
                print(f"    {ftype}  {f['pathSuffix']:<40} {size}")
        else:
            print(f"    [ERROR] {r.status_code} — {hdfs_dir} may not exist")
    except Exception as e:
        print(f"    [EXCEPTION] {e}")


def run_check():
    """Check HDFS connectivity and list key directories."""
    print("\n" + "=" * 60)
    print("  NeuroTraffic — HDFS Status Check")
    print("=" * 60)

    if not hdfs_helper.is_hdfs_available():
        print("\n  [ERROR] Cannot reach HDFS WebHDFS at "
              f"{hdfs_helper.WEBHDFS_BASE_URL}")
        print("  Make sure the Hadoop cluster is running:")
        print("    cd Navaneeth && docker-compose up -d")
        sys.exit(1)

    print(f"\n  Connected to: {hdfs_helper.WEBHDFS_BASE_URL}")

    dirs_to_check = [
        "/data",
        "/data/graph_features",
        "/data/predictions",
        "/data/streaming/output",
    ]
    for d in dirs_to_check:
        print(f"\n  {d}/")
        list_hdfs(d)


# ══════════════════════════════════════════════════════════
# Main ingest
# ══════════════════════════════════════════════════════════

def run_ingest(force=False):
    print("\n" + "=" * 60)
    print("  NeuroTraffic — HDFS Ingest")
    print("=" * 60)

    # 1. Connectivity check
    print("\n[1/4] Checking HDFS connectivity ...")
    if not hdfs_helper.is_hdfs_available():
        print(f"\n  [ERROR] Cannot reach WebHDFS at {hdfs_helper.WEBHDFS_BASE_URL}")
        print("  Steps to fix:")
        print("    1. cd NeuroTraffic/Navaneeth")
        print("    2. docker-compose up -d")
        print("    3. Wait ~30 seconds for namenode to leave safe mode")
        print("    4. Re-run: python dashboard/hdfs_ingest.py")
        sys.exit(1)

    print(f"  Connected! Namenode: {hdfs_helper.WEBHDFS_HOST}:{hdfs_helper.WEBHDFS_PORT}")

    # 2. Check safe mode
    print("\n[2/4] Checking safe mode ...")
    try:
        url = f"{hdfs_helper.WEBHDFS_BASE_URL}/?op=GETHOMEDIRECTORY"
        r = requests.get(url, timeout=3)
        # Try a write — if in safe mode, this will fail
        test_dir_url = f"{hdfs_helper.WEBHDFS_BASE_URL}/tmp?op=MKDIRS"
        r2 = requests.put(test_dir_url, timeout=5)
        if r2.status_code == 200:
            print("  Safe mode: OFF — ready for writes.")
        else:
            print(f"  [WARN] Write test returned {r2.status_code}. Namenode may still be in safe mode.")
            print("  Waiting 15 seconds ...")
            time.sleep(15)
    except Exception as e:
        print(f"  [WARN] Safe-mode check failed: {e}")

    # 3. Create directory structure
    print("\n[3/4] Creating HDFS directory structure ...")
    for d in HDFS_DIRS:
        ok = hdfs_mkdir(d)
        status = "OK" if ok else "FAILED"
        print(f"    {status:<8} {d}")

    # 4. Upload data files
    print(f"\n[4/4] Uploading data files (force={force}) ...")
    ok_count = 0
    fail_count = 0

    for local_path, hdfs_path in FILE_MAP:
        result = upload_file(local_path, hdfs_path, force=force)
        if result:
            ok_count += 1
        else:
            fail_count += 1

    # Also upload any history CSVs in streaming/output/history/
    history_local = os.path.join(config.STREAMING_OUTPUT_DIR, "history")
    if os.path.isdir(history_local):
        print(f"\n    Uploading streaming history files ...")
        for fname in sorted(os.listdir(history_local)):
            if fname.endswith(".csv"):
                local_path = os.path.join(history_local, fname)
                hdfs_path  = f"{config.HDFS_STREAMING_OUTPUT_DIR}/history/{fname}"
                result = upload_file(local_path, hdfs_path, force=force)
                ok_count  += int(result)
                fail_count += int(not result)

    print(f"\n{'='*60}")
    print(f"  Ingest complete:  {ok_count} succeeded,  {fail_count} failed/skipped")
    print(f"  HDFS path prefix: hdfs://localhost:9000/data/")
    print(f"  WebHDFS UI:       http://localhost:9870")
    print(f"{'='*60}\n")

    if fail_count > 0:
        print("  TIP: Run `python data/generate_sample_data.py` first to create")
        print("  the missing local files, then re-run this ingest script.\n")


# ══════════════════════════════════════════════════════════
# Wait-for-HDFS helper (used by other scripts)
# ══════════════════════════════════════════════════════════

def wait_for_hdfs(timeout_seconds=120, poll_interval=5):
    """
    Block until HDFS WebHDFS is reachable, or raise TimeoutError.
    Useful for scripting: call this before ingest in automated pipelines.
    """
    print(f"[WAIT] Waiting for HDFS (timeout={timeout_seconds}s) ...", flush=True)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if hdfs_helper.is_hdfs_available():
            print("[WAIT] HDFS is up!")
            return
        print(f"[WAIT] Not ready yet — retrying in {poll_interval}s ...", flush=True)
        time.sleep(poll_interval)
    raise TimeoutError(f"HDFS did not become available within {timeout_seconds} seconds.")


# ══════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroTraffic HDFS Ingest")
    parser.add_argument("--check", action="store_true", help="Only check HDFS status")
    parser.add_argument("--force", action="store_true", help="Re-upload all files even if they exist")
    parser.add_argument("--wait",  type=int, default=0,
                        help="Wait this many seconds for HDFS to become available before ingesting")
    args = parser.parse_args()

    if args.wait > 0:
        wait_for_hdfs(timeout_seconds=args.wait)

    if args.check:
        run_check()
    else:
        run_ingest(force=args.force)
