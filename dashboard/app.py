"""
NeuroTraffic — Home Page
=========================
Owner: Sasmitha S (CB.AI.U4AID24051)
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
import time

# ── Path setup ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import hdfs_helper
import requests

if "use_hdfs" not in st.session_state:
    st.session_state.use_hdfs = config.USE_HDFS

config.USE_HDFS = st.session_state.use_hdfs

# ── Page config ──
st.set_page_config(
    page_title="NeuroTraffic — NYC Transport Intelligence",
    page_icon="🚊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset & Base ── */
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #090e1a !important;
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1425 0%, #111827 100%) !important;
    border-right: 1px solid rgba(0, 212, 255, 0.12) !important;
}

[data-testid="stSidebarContent"] { padding: 1.5rem 1rem !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0, 212, 255, 0.15) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    backdrop-filter: blur(10px);
}
[data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 0.75rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 1.8rem !important; font-weight: 700 !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0ea5e9) !important;
    color: #090e1a !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.03em !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0, 212, 255, 0.35) !important;
}

/* ── Danger / Alert buttons ── */
.danger-btn > button {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    color: white !important;
}

/* ── Selects & sliders ── */
.stSelectbox > div > div, .stSlider { color: #e2e8f0 !important; }
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(0, 212, 255, 0.2) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
.stSlider [data-baseweb="slider"] { color: #00d4ff !important; }

/* ── DataFrames ── */
.stDataFrame { border-radius: 10px !important; overflow: hidden !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.03) !important; border-radius: 10px !important; padding: 4px !important; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #94a3b8 !important; border-radius: 8px !important; padding: 0.5rem 1rem !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] { background: rgba(0, 212, 255, 0.15) !important; color: #00d4ff !important; }

/* ── Expanders ── */
.streamlit-expanderHeader { background: rgba(255,255,255,0.04) !important; border-radius: 8px !important; }

/* ── Progress / spinner ── */
.stProgress > div > div { background: linear-gradient(90deg, #00d4ff, #0ea5e9) !important; }

/* ── Alert boxes ── */
.stAlert { border-radius: 10px !important; border: none !important; }
.stSuccess { background: rgba(34, 197, 94, 0.12) !important; border-left: 3px solid #22c55e !important; }
.stError   { background: rgba(239, 68, 68, 0.12) !important; border-left: 3px solid #ef4444 !important; }
.stWarning { background: rgba(250, 204, 21, 0.12) !important; border-left: 3px solid #facc15 !important; }
.stInfo    { background: rgba(0, 212, 255, 0.08) !important; border-left: 3px solid #00d4ff !important; }

/* ── Custom card ── */
.nt-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(0, 212, 255, 0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
.nt-card:hover {
    border-color: rgba(0, 212, 255, 0.35);
    box-shadow: 0 4px 32px rgba(0, 212, 255, 0.08);
}

/* ── Status badge ── */
.status-ok   { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; display: inline-block; }
.status-warn { background: rgba(250,204,21,0.15); color: #facc15; border: 1px solid rgba(250,204,21,0.3); padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; display: inline-block; }
.status-crit { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; display: inline-block; }

/* ── Page nav cards ── */
.nav-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.25s ease;
    height: 140px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}
.nav-card:hover {
    border-color: rgba(0, 212, 255, 0.4);
    background: rgba(0, 212, 255, 0.06);
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0, 212, 255, 0.12);
}
.nav-card .icon { font-size: 2rem; }
.nav-card .label { font-size: 0.9rem; font-weight: 600; color: #e2e8f0; }
.nav-card .desc  { font-size: 0.72rem; color: #64748b; }

/* ── Hero section ── */
.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00d4ff 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0;
}
.hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-top: 0.75rem;
    font-weight: 400;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.25);
    color: #00d4ff;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Section header ── */
.section-header {
    font-size: 0.7rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}

/* ── Team table ── */
.team-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.75rem 1rem;
    border-radius: 10px;
    margin-bottom: 6px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    transition: background 0.2s;
}
.team-row:hover { background: rgba(255,255,255,0.045); }
.team-avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700;
    flex-shrink: 0;
}

/* ── Plotly chart background fix ── */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1425; }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,212,255,0.5); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1.5rem;">
        <div style="font-size:2.5rem;">🚊</div>
        <div style="font-size:1.1rem; font-weight:800; color:#00d4ff; letter-spacing:-0.02em;">NeuroTraffic</div>
        <div style="font-size:0.65rem; color:#475569; margin-top:4px;">NYC Transport Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
    st.page_link("app.py",                    label="🏠  Home",                  help="Project overview")
    st.page_link("pages/1_live_monitor.py",   label="📡  Live Monitor",           help="Real-time disruption map")
    st.page_link("pages/2_simulator.py",      label="⚡  Disruption Simulator",   help="Simulate cascade events")
    st.page_link("pages/3_explorer.py",       label="📊  Historical Explorer",    help="6-month trend analysis")
    st.page_link("pages/4_analytics.py",      label="🧠  Analytics Dashboard",    help="Model metrics & graph stats")

    st.markdown('<div class="section-header">Hadoop / HDFS</div>', unsafe_allow_html=True)
    use_hdfs = st.toggle("Enable HDFS Mode", value=st.session_state.use_hdfs, key="hdfs_toggle")
    if use_hdfs != st.session_state.use_hdfs:
        st.session_state.use_hdfs = use_hdfs
        config.USE_HDFS = use_hdfs
        st.rerun()

    if use_hdfs:
        hdfs_ok = hdfs_helper.is_hdfs_available()
        if hdfs_ok:
            st.markdown('<span class="status-ok">● HDFS Connected</span>', unsafe_allow_html=True)
            st.caption(f"Namenode: {config.HDFS_HOST}:{config.HDFS_PORT_WEB}")
        else:
            st.markdown('<span class="status-crit">○ HDFS Offline</span>', unsafe_allow_html=True)
            st.caption("Using local fallback data")

    st.markdown('<div class="section-header">System</div>', unsafe_allow_html=True)

    # Live system status
    latest = hdfs_helper.load_dataframe(config.LATEST_CSV, config.HDFS_LATEST_CSV, use_hdfs=use_hdfs)
    if latest is not None:
        try:
            disruptions = latest[latest["is_disruption"] == 1]
            n_disrupt = len(disruptions)
            if n_disrupt > 0:
                top_line = disruptions.iloc[0]["affected_line"]
                st.markdown(f'<span class="status-crit">⚠ DISRUPTION — {top_line} line</span>', unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.72rem; color:#94a3b8; margin-top:6px;'>{n_disrupt} zones affected</div>", unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-ok">● All Systems Normal</span>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<span class="status-ok">● Data loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warn">○ No live data yet</span>', unsafe_allow_html=True)
        if use_hdfs:
            st.caption("Start streaming pipeline & replay simulator to write predictions to HDFS")
        else:
            st.caption("Run generate_sample_data.py first")

    st.markdown('<div class="section-header">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; color:#64748b; line-height:1.6;">
        Big Data Analytics (23AID302)<br>
        Amrita Vishwa Vidyapeetham<br><br>
        <b style="color:#94a3b8;">Sasmitha S</b> — CB.AI.U4AID24051<br>
        Streaming Pipeline · Web App
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main: Hero
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding: 2.5rem 0 1.5rem;">
    <div class="hero-badge">🏙 NYC · Big Data Analytics · 2023</div>
    <h1 class="hero-title">NeuroTraffic</h1>
    <p class="hero-sub">
        Multi-modal transport disruption intelligence for New York City.<br>
        Spark Structured Streaming · Graph Analytics · ML Anomaly Detection
    </p>
</div>
""", unsafe_allow_html=True)

# ── Quick stats ──
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Transport Nodes",  "3,219",  help="Taxi zones + subway stations + bike docks")
with col2: st.metric("Graph Edges",      "643K",   help="Trip-based + geographic proximity edges")
with col3: st.metric("Data Sources",     "6",      help="Taxi, Subway, Bike, Weather, 311, Events")
with col4: st.metric("GBT Accuracy",     "97.4%",  help="Gradient Boosted Trees on historical data")
with col5: st.metric("Streaming Lag",    "< 5 s",  help="Replay simulator → pipeline → dashboard")

# ─────────────────────────────────────────────
# Navigation Cards
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Pages</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""
    <a href="/1_live_monitor" style="text-decoration:none;">
    <div class="nav-card">
        <div class="icon">📡</div>
        <div class="label">Live Monitor</div>
        <div class="desc">Real-time NYC map · Auto-refreshes every 5 s</div>
    </div></a>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <a href="/2_simulator" style="text-decoration:none;">
    <div class="nav-card">
        <div class="icon">⚡</div>
        <div class="label">Disruption Simulator</div>
        <div class="desc">Pick a subway line · See cascade predictions</div>
    </div></a>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <a href="/3_explorer" style="text-decoration:none;">
    <div class="nav-card">
        <div class="icon">📊</div>
        <div class="label">Historical Explorer</div>
        <div class="desc">6-month trend analysis · Disruption timeline</div>
    </div></a>""", unsafe_allow_html=True)
with c4:
    st.markdown("""
    <a href="/4_analytics" style="text-decoration:none;">
    <div class="nav-card">
        <div class="icon">🧠</div>
        <div class="label">Analytics Dashboard</div>
        <div class="desc">Model metrics · Graph stats · Feature importance</div>
    </div></a>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Architecture diagram
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Pipeline Architecture</div>', unsafe_allow_html=True)
st.markdown("""
<div class="nt-card">
<div style="font-family: 'Courier New', monospace; font-size: 0.78rem; color: #94a3b8; line-height: 2; overflow-x: auto;">
<span style="color:#00d4ff; font-weight:700;">NYC Open Data</span>
&nbsp;──►&nbsp;
<span style="color:#a78bfa; font-weight:700;">Data Pipeline</span>&nbsp;(Navaneeth · Spark ETL)
&nbsp;──►&nbsp;
<span style="color:#34d399; font-weight:700;">Graph Engine</span>&nbsp;(Tanay · Spark GraphX)
&nbsp;──►&nbsp;
<span style="color:#fb923c; font-weight:700;">ML Pipeline</span>&nbsp;(Harsada · Isolation Forest + GBT)
&nbsp;──►&nbsp;
<span style="color:#f472b6; font-weight:700;">This Dashboard</span>&nbsp;(Sasmitha · Spark Streaming + Streamlit)
<br><br>
<span style="color:#475569;">Taxi · Subway · Bike · Weather · 311 · Events</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<span style="color:#475569;">Cleaned Parquet (HDFS)</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<span style="color:#475569;">PageRank · Communities</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<span style="color:#475569;">Anomaly Scores · Cascade</span>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<span style="color:#475569;">Live Map · Simulator · Analytics</span>
</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Team
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Team</div>', unsafe_allow_html=True)

team = [
    ("T",  "Tanay",     "CB.AI.U4AID24059", "Graph Engine (Spark GraphX · PageRank · Label Propagation)", "#00d4ff"),
    ("N",  "Navaneeth", "—",                "Data Pipeline (Spark ETL · 6 data sources · HDFS)",           "#a78bfa"),
    ("H",  "Harsada",   "—",                "ML Pipeline (Isolation Forest · GBT · PySpark MLlib)",         "#34d399"),
    ("S",  "Sasmitha",  "CB.AI.U4AID24051", "Streaming Pipeline · Streamlit Web App  ← you are here",       "#f472b6"),
]

for init, name, roll, role, color in team:
    is_you = "you are here" in role
    border = f"border: 1px solid {color}33;" if is_you else "border: 1px solid rgba(255,255,255,0.06);"
    bg     = f"background: {color}0a;" if is_you else "background: rgba(255,255,255,0.025);"
    st.markdown(f"""
    <div class="team-row" style="{bg} {border}">
        <div class="team-avatar" style="background:{color}22; color:{color};">{init}</div>
        <div style="flex:1;">
            <span style="color:#e2e8f0; font-weight:600; font-size:0.85rem;">{name}</span>
            <span style="color:#475569; font-size:0.72rem; margin-left:8px;">{roll}</span><br>
            <span style="color:#64748b; font-size:0.75rem;">{role}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Data status
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Data Status</div>', unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])
with col1:
    files_to_check = [
        (config.GRAPH_NODES_CSV,         config.HDFS_GRAPH_NODES_CSV,         "graph_nodes.csv",               "From Tanay"),
        (config.HISTORICAL_PARQUET,       config.HDFS_HISTORICAL_PARQUET,       "historical_predictions.parquet", "From Navaneeth"),
        (config.EVALUATION_METRICS_JSON,  config.HDFS_EVALUATION_METRICS_JSON,  "evaluation_metrics.json",        "From Harsada"),
        (config.FEATURE_IMPORTANCE_CSV,   config.HDFS_FEATURE_IMPORTANCE_CSV,   "feature_importance.csv",         "From Harsada"),
        (config.CONFUSION_MATRIX_CSV,     config.HDFS_CONFUSION_MATRIX_CSV,     "confusion_matrix.csv",           "From Harsada"),
        (config.LATEST_CSV,               config.HDFS_LATEST_CSV,               "latest.csv",                     "Streaming output"),
    ]
    
    missing = 0
    hdfs_active = st.session_state.use_hdfs and hdfs_helper.is_hdfs_available()
    
    for lpath, hpath, fname, source in files_to_check:
        if hdfs_active:
            url = f"{config.HDFS_WEB_PREFIX}{hpath}?op=GETFILESTATUS"
            try:
                r = requests.get(url, timeout=1.0)
                exists = r.status_code == 200
                if exists:
                    file_status = r.json().get("FileStatus", {})
                    size_bytes = file_status.get("length", 0)
                    size = f"{size_bytes/1024:.1f} KB"
                else:
                    size = "missing"
            except Exception:
                exists = False
                size = "conn error"
            source_lbl = f"{source} (HDFS)"
        else:
            exists = os.path.exists(lpath)
            size = f"{os.path.getsize(lpath)/1024:.0f} KB" if exists else "missing"
            source_lbl = f"{source} (Local)"
            
        icon = "✅" if exists else "❌"
        if not exists:
            missing += 1
            
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04);">
            <span style="font-size:1rem;">{icon}</span>
            <span style="color:#e2e8f0;font-size:0.8rem;font-weight:500;flex:1;">{fname}</span>
            <span style="color:#475569;font-size:0.72rem;">{source_lbl}</span>
            <span style="color:#64748b;font-size:0.72rem;">{size}</span>
        </div>
        """, unsafe_allow_html=True)

with col2:
    if missing == 0:
        st.success("✅ All data files present")
    else:
        st.warning(f"⚠ {missing} file(s) missing")
        st.info("Run `python data/generate_sample_data.py` to generate sample data")
        if st.button("🔄  Generate Sample Data"):
            import subprocess
            gen_script = os.path.join(config.DATA_DIR, "generate_sample_data.py")
            with st.spinner("Generating sample data ..."):
                result = subprocess.run(
                    [sys.executable, gen_script],
                    capture_output=True, text=True, cwd=config.BASE_DIR
                )
            if result.returncode == 0:
                st.success("Done! Refresh the page.")
            else:
                st.error(f"Error:\n{result.stderr[:500]}")
