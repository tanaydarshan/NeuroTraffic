@echo off
REM ========================================================
REM  NeuroTraffic — Full Hadoop Pipeline Launcher
REM  Owner: Sasmitha S (CB.AI.U4AID24051)
REM
REM  This script:
REM    1. Starts the Hadoop + Spark Docker cluster
REM    2. Waits for namenode to leave safe mode
REM    3. Generates sample data if needed
REM    4. Ingests data files into HDFS
REM    5. Starts the replay simulator (HDFS mode, background)
REM    6. Launches the Streamlit dashboard
REM
REM  Usage:
REM    start_hadoop_pipeline.bat
REM ========================================================

setlocal enabledelayedexpansion
set SCRIPT_DIR=%~dp0
set DASHBOARD_DIR=%SCRIPT_DIR%dashboard
set NAVANEETH_DIR=%SCRIPT_DIR%Navaneeth
set NEUROTRAFFIC_USE_HDFS=true

echo.
echo  ====================================================
echo   NeuroTraffic -- Full Hadoop Pipeline Launcher
echo  ====================================================

REM ── Step 1: Start Docker cluster ──
echo.
echo [1/6] Starting Hadoop + Spark Docker cluster ...
cd /d "%NAVANEETH_DIR%"
docker-compose up -d
if errorlevel 1 (
    echo  [ERROR] docker-compose failed. Is Docker Desktop running?
    pause
    exit /b 1
)

REM ── Step 2: Wait for namenode ──
echo.
echo [2/6] Waiting 30 seconds for namenode to start ...
timeout /t 30 /nobreak >nul

REM ── Step 3: Generate sample data if needed ──
echo.
echo [3/6] Checking local data files ...
cd /d "%DASHBOARD_DIR%"
if not exist "data\graph_nodes.csv" (
    echo   graph_nodes.csv missing -- generating sample data ...
    python data\generate_sample_data.py
    if errorlevel 1 (
        echo   [ERROR] Sample data generation failed.
        pause
        exit /b 1
    )
) else (
    echo   Local data files found -- skipping generation.
)

REM ── Step 4: Ingest into HDFS ──
echo.
echo [4/6] Ingesting data into HDFS ...
set NEUROTRAFFIC_USE_HDFS=true
python hdfs_ingest.py --wait 90
if errorlevel 1 (
    echo   [ERROR] HDFS ingest failed. Check cluster status.
    pause
    exit /b 1
)

REM ── Step 5: Start replay simulator in background (HDFS mode) ──
echo.
echo [5/6] Starting replay simulator (HDFS mode, looping) ...
start "NeuroTraffic Replay Simulator" cmd /k "cd /d %DASHBOARD_DIR% && set NEUROTRAFFIC_USE_HDFS=true && python streaming\replay_simulator.py --loop --hdfs --speed 5"

REM ── Step 6: Launch Streamlit dashboard ──
echo.
echo [6/6] Launching Streamlit dashboard ...
echo   URL: http://localhost:8501
echo   Enable the HDFS toggle in the sidebar to connect to Hadoop.
echo.
cd /d "%DASHBOARD_DIR%"
set NEUROTRAFFIC_USE_HDFS=true
streamlit run app.py --server.port 8501 --server.headless false

echo.
echo  NeuroTraffic pipeline stopped.
pause
