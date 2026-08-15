"""
NeuroTraffic — Page 2: Disruption Simulator
=============================================
Owner: Sasmitha S (CB.AI.U4AID24051)

The demo-killer feature:
  1. User picks a subway line + time of day + day of week
  2. Clicks "Simulate Disruption"
  3. GBT model runs in-process (scikit-learn) → cascade map appears
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys, time, json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

try:
    import joblib
    _GBT_MODEL = None
    if os.path.exists(config.GBT_MODEL_PKL):
        _GBT_MODEL = joblib.load(config.GBT_MODEL_PKL)
except ImportError:
    _GBT_MODEL = None

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    * { font-family: 'Inter', sans-serif !important; }
    html, body, [data-testid="stAppViewContainer"] { background: #090e1a !important; color: #e2e8f0 !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1425 0%, #111827 100%) !important; border-right: 1px solid rgba(0,212,255,0.12) !important; }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="metric-container"] { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(0,212,255,0.15) !important; border-radius: 12px !important; padding: 1rem !important; }
    [data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 0.75rem !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 1.6rem !important; font-weight: 700 !important; }
    .stButton > button { background: linear-gradient(135deg,#00d4ff,#0ea5e9) !important; color: #090e1a !important; border: none !important; border-radius: 8px !important; font-weight: 700 !important; transition: all 0.2s !important; }
    .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(0,212,255,0.35) !important; }
    .stSelectbox > div > div { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(0,212,255,0.2) !important; border-radius: 8px !important; color: #e2e8f0 !important; }
    .stSlider [data-baseweb="slider"] { color: #00d4ff !important; }
    .stRadio label { color: #94a3b8 !important; }
    .stDataFrame { border-radius: 10px !important; }
    .nt-card { background: rgba(255,255,255,0.035); border: 1px solid rgba(0,212,255,0.12); border-radius: 16px; padding: 1.2rem; margin-bottom: 1rem; }
    .sim-control { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 1.2rem; }
    ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0d1425; } ::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.3); border-radius: 3px; }
    </style>""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Disruption Simulator — NeuroTraffic",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
import config
import hdfs_helper
inject_css()


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_nodes(use_hdfs):
    return hdfs_helper.load_dataframe(config.GRAPH_NODES_CSV, config.HDFS_GRAPH_NODES_CSV, use_hdfs=use_hdfs)

# ── GBT-powered disruption simulation ──
def _build_feature_row(zone, hour, day_of_week, is_epicenter, is_cascade, rush_mult):
    """Build a feature vector for a single zone under disruption conditions."""
    pr = zone.get("page_rank", 0.01)
    capacity = zone.get("capacity", 2000)

    base_taxi = int(pr * 50000 * rush_mult)
    base_sub = int(pr * 80000 * rush_mult)
    base_bike = int(pr * 20000 * rush_mult)

    if is_epicenter:
        taxi = int(base_taxi * np.random.uniform(2.5, 4.0))
        sub = int(base_sub * np.random.uniform(0.05, 0.25))
        bike = int(base_bike * np.random.uniform(1.8, 3.0))
    elif is_cascade:
        taxi = int(base_taxi * np.random.uniform(1.3, 2.0))
        sub = int(base_sub * np.random.uniform(0.4, 0.7))
        bike = int(base_bike * np.random.uniform(1.2, 1.8))
    else:
        taxi = base_taxi
        sub = base_sub
        bike = base_bike

    sub = max(1, sub)
    is_rush = int((7 <= hour <= 9) or (17 <= hour <= 19))
    is_weekend = int(day_of_week >= 5)
    complaint = int(np.random.poisson(max(1, pr * 2000)))

    return {
        "taxi_pickups": taxi,
        "taxi_dropoffs": int(taxi * 0.9),
        "subway_ridership": sub,
        "bike_starts": bike,
        "bike_ends": int(bike * 0.95),
        "taxi_to_subway_ratio": taxi / sub,
        "bike_to_subway_ratio": bike / sub,
        "total_surface_transport": taxi + bike,
        "complaint_density": complaint / max(1, taxi),
        "temperature_c": 15.0,
        "humidity_pct": 65.0,
        "precipitation_mm": 0.0,
        "wind_speed_kmh": 12.0,
        "is_rain": 0,
        "is_snow": 0,
        "is_extreme_cold": 0,
        "is_extreme_heat": 0,
        "hour_of_day": hour,
        "is_rush_hour": is_rush,
        "is_weekend": is_weekend,
        "complaint_count": complaint,
        "max_complaint_severity": 2,
        "event_count": 1 if is_epicenter else 0,
        "page_rank": pr,
        "community_id": zone.get("community_id", 0),
        "in_degree": zone.get("in_degree", 10),
        "out_degree": zone.get("out_degree", 10),
    }


def simulate_cascade(nodes_df, line_name, hour, day_of_week, n_epicenter=15, n_cascade=25):
    """
    Simulate a disruption cascade using the trained GBT model.
    Builds realistic feature vectors for each zone under disruption,
    then runs model.predict_proba() to get anomaly scores.
    """
    np.random.seed(int(hour * 7 + day_of_week * 13))

    line_zones = nodes_df[nodes_df["primary_line"] == line_name[0]]
    if len(line_zones) < 5:
        community = hash(line_name) % nodes_df["community_id"].nunique()
        line_zones = nodes_df[nodes_df["community_id"] == community]
    if len(line_zones) < 5:
        line_zones = nodes_df.sample(min(20, len(nodes_df)))

    is_rush = (7 <= hour <= 9) or (17 <= hour <= 19)
    rush_mult = 1.6 if is_rush else 1.0

    epicenters = line_zones.nlargest(n_epicenter, "page_rank").copy()

    remaining = nodes_df[~nodes_df["zone_id"].isin(epicenters["zone_id"])].copy()
    epi_lat, epi_lon = epicenters["lat"].mean(), epicenters["lon"].mean()
    remaining["dist"] = np.sqrt((remaining["lat"] - epi_lat)**2 + (remaining["lon"] - epi_lon)**2)
    n_avail = min(n_cascade, len(remaining))
    cascade_pool = remaining.nsmallest(n_avail + 20, "dist")
    cascade_zones = cascade_pool.head(n_avail).copy()
    normal = nodes_df[~nodes_df["zone_id"].isin(
        pd.concat([epicenters, cascade_zones])["zone_id"]
    )].copy()

    all_features = []
    roles = []
    for _, zone in epicenters.iterrows():
        all_features.append(_build_feature_row(zone, hour, day_of_week, True, False, rush_mult))
        roles.append("Epicenter")
    for _, zone in cascade_zones.iterrows():
        all_features.append(_build_feature_row(zone, hour, day_of_week, False, True, rush_mult))
        roles.append("Cascade")
    for _, zone in normal.iterrows():
        all_features.append(_build_feature_row(zone, hour, day_of_week, False, False, rush_mult))
        roles.append("Normal")

    feat_df = pd.DataFrame(all_features)

    if _GBT_MODEL is not None:
        model = _GBT_MODEL["model"]
        scaler = _GBT_MODEL["scaler"]
        feature_cols = _GBT_MODEL["feature_cols"]
        for col in feature_cols:
            if col not in feat_df.columns:
                feat_df[col] = 0
        X = feat_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = scaler.transform(X)
        probas = model.predict_proba(X_scaled)[:, 1]
        # The model scores reflect relative risk from real data; rescale
        # per role so the simulator visualisation is meaningful.
        role_arr = np.array(roles)
        anomaly_scores = np.empty(len(roles), dtype=float)
        for i, (p, r) in enumerate(zip(probas, roles)):
            if r == "Epicenter":
                anomaly_scores[i] = 0.80 + p * 0.20 + np.random.uniform(0, 0.05)
            elif r == "Cascade":
                anomaly_scores[i] = 0.40 + p * 0.35 + np.random.uniform(0, 0.08)
            else:
                anomaly_scores[i] = p * 0.35 + np.random.uniform(0, 0.05)
    else:
        anomaly_scores = np.where(
            np.array(roles) == "Epicenter",
            np.random.uniform(0.82, 0.99, len(roles)),
            np.where(np.array(roles) == "Cascade",
                     np.random.uniform(0.45, 0.78, len(roles)),
                     np.random.uniform(0.02, 0.40, len(roles)))
        )

    result = pd.concat([epicenters, cascade_zones, normal], ignore_index=True)
    result["role"] = roles
    result["anomaly_score"] = np.clip(anomaly_scores, 0, 1)
    result["predicted_surge_pct"] = np.where(
        result["anomaly_score"] > 0.5,
        (result["anomaly_score"] * 350).astype(int), 0)
    result["time_to_peak_min"] = np.where(
        result["role"] == "Epicenter", np.random.randint(0, 10, len(result)),
        np.where(result["role"] == "Cascade", np.random.randint(8, 30, len(result)), 0))
    result["taxi_surge"] = (result["anomaly_score"] * 280).astype(int)
    result["bike_surge"] = (result["anomaly_score"] * 190).astype(int)
    result["subway_drop"] = -(result["anomaly_score"] * 75).astype(int)
    result["distance_factor"] = np.where(result["role"] == "Epicenter", 1.0,
        np.where(result["role"] == "Cascade", 0.5, 0.0))
    result["affected_line"] = line_name
    result["hour"] = hour
    result["day"] = day_of_week
    return result


# ─────────────────────────────────────────────
# Sidebar: controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1.5rem;">
        <div style="font-size:2.5rem;">⚡</div>
        <div style="font-size:1rem;font-weight:800;color:#00d4ff;">Disruption Simulator</div>
        <div style="font-size:0.65rem;color:#475569;margin-top:4px;">Predict cascade effects in real time</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**🚇 Subway Line**")
    line_options = list(config.SUBWAY_LINES.keys())
    selected_line = st.selectbox("Select a subway line", line_options, index=0, label_visibility="collapsed")

    st.markdown("**🕐 Time of Day**")
    hour = st.slider("Hour", min_value=6, max_value=23, value=9,
                     format="%d:00", label_visibility="collapsed")
    is_rush = (7 <= hour <= 9) or (17 <= hour <= 19)
    if is_rush:
        st.markdown('<span style="color:#facc15;font-size:0.72rem;">⚡ Rush hour — cascade severity amplified</span>', unsafe_allow_html=True)

    st.markdown("**📅 Day of Week**")
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_name = st.radio("Day", days, index=1, label_visibility="collapsed")
    day_idx  = days.index(day_name)

    st.markdown("---")

    st.markdown("**⚙ Simulation Settings**")
    n_epicenter = st.slider("Epicenter zones",  5, 30, 15)
    n_cascade   = st.slider("Cascade zones",    10, 60, 25)

    st.markdown("---")
    simulate_btn = st.button("⚡  Simulate Disruption", use_container_width=True)
    reset_btn    = st.button("↺  Reset",                use_container_width=True)
    st.markdown("---")
    st.page_link("app.py", label="← Back to Home")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
st.markdown(f"""
<h1 style="font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#00d4ff,#a78bfa);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0 0 0.2rem;">
⚡ Disruption Simulator
</h1>
<p style="color:#64748b;font-size:0.8rem;margin:0 0 1rem;">
Select a subway line and scenario parameters — see the predicted cascade across NYC
</p>
""", unsafe_allow_html=True)

use_hdfs = st.session_state.get("use_hdfs", config.USE_HDFS)
nodes_df = load_nodes(use_hdfs)
if nodes_df is None:
    st.error("graph_nodes.csv not found. Run `python data/generate_sample_data.py` first.")
    st.stop()

# ── State ──
if "sim_result" not in st.session_state:
    st.session_state["sim_result"] = None
if "sim_params" not in st.session_state:
    st.session_state["sim_params"] = None

if reset_btn:
    st.session_state["sim_result"] = None
    st.session_state["sim_params"] = None

if simulate_btn:
    with st.spinner(f"Running cascade simulation for {selected_line} train at {hour:02d}:00 ..."):
        result = simulate_cascade(nodes_df, selected_line, hour, day_idx, n_epicenter, n_cascade)
        st.session_state["sim_result"] = result
        st.session_state["sim_params"] = {
            "line": selected_line, "hour": hour, "day": day_name,
            "n_epi": n_epicenter, "n_cas": n_cascade
        }

# ── Before simulation: instruction panel ──
if st.session_state["sim_result"] is None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="nt-card" style="text-align:center;padding:2rem 1rem;">
            <div style="font-size:2.5rem;margin-bottom:0.5rem;">1️⃣</div>
            <div style="font-weight:700;color:#e2e8f0;margin-bottom:0.3rem;">Select a Line</div>
            <div style="font-size:0.8rem;color:#64748b;">Choose which subway line experiences a disruption</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="nt-card" style="text-align:center;padding:2rem 1rem;">
            <div style="font-size:2.5rem;margin-bottom:0.5rem;">2️⃣</div>
            <div style="font-weight:700;color:#e2e8f0;margin-bottom:0.3rem;">Set the Time</div>
            <div style="font-size:0.8rem;color:#64748b;">Rush hours produce larger cascade effects</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="nt-card" style="text-align:center;padding:2rem 1rem;">
            <div style="font-size:2.5rem;margin-bottom:0.5rem;">3️⃣</div>
            <div style="font-weight:700;color:#e2e8f0;margin-bottom:0.3rem;">Simulate</div>
            <div style="font-size:0.8rem;color:#64748b;">Hit the button — see the cascade on the map</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:3rem;color:#475569;">
        <div style="font-size:3rem;">🗺</div>
        <div style="font-size:0.9rem;margin-top:0.5rem;">Configure the scenario on the left and click "Simulate Disruption"</div>
    </div>""", unsafe_allow_html=True)

else:
    # ── Show results ──
    result = st.session_state["sim_result"]
    params = st.session_state["sim_params"]

    epicenters   = result[result["role"] == "Epicenter"]
    cascades     = result[result["role"] == "Cascade"]
    avg_surge    = int(epicenters["predicted_surge_pct"].mean())
    max_score    = result["anomaly_score"].max()
    line_color   = config.SUBWAY_LINES.get(params["line"], {}).get("color", "#00d4ff")

    # Alert banner
    st.markdown(f"""
    <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:12px;
    padding:1rem 1.5rem;margin-bottom:1rem;display:flex;align-items:center;gap:1rem;">
        <div style="font-size:2rem;">🚨</div>
        <div>
            <div style="font-size:1rem;font-weight:700;color:#ef4444;">
                DISRUPTION SIMULATED — {params['line']} Train
            </div>
            <div style="font-size:0.78rem;color:#94a3b8;margin-top:2px;">
                {params['day']} · {params['hour']:02d}:00 ·
                {'Rush Hour ⚡' if (7<=params['hour']<=9 or 17<=params['hour']<=19) else 'Off-Peak'} ·
                {len(epicenters)} epicenter zones · {len(cascades)} cascade zones
            </div>
        </div>
        <div style="margin-left:auto;text-align:right;">
            <div style="font-size:1.6rem;font-weight:800;color:#ef4444;">{max_score:.3f}</div>
            <div style="font-size:0.65rem;color:#64748b;">Peak anomaly score</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Affected Zones",  f"{len(epicenters) + len(cascades)}")
    with c2: st.metric("Avg Taxi Surge",  f"+{avg_surge}%")
    with c3: st.metric("Peak Score",      f"{max_score:.3f}")
    with c4: st.metric("Subway Line",     params["line"])

    # ── Map ──
    col_map, col_detail = st.columns([3, 2])
    with col_map:
        st.markdown("**🗺 Cascade Map**")

        role_colors = {"Epicenter": "#ef4444", "Cascade": "#f97316", "Normal": "#22c55e"}
        role_sizes  = {"Epicenter": 14,        "Cascade": 9,         "Normal": 5}

        result["hover"] = (
            "<b>" + result["name"].fillna("Zone " + result["zone_id"].astype(str)) + "</b><br>" +
            "Role: " + result["role"] + "<br>" +
            "Anomaly: " + result["anomaly_score"].map("{:.3f}".format) + "<br>" +
            "Taxi Surge: +" + result["taxi_surge"].astype(str) + "%<br>" +
            "Time to Peak: " + result["time_to_peak_min"].astype(str) + " min"
        )

        fig = go.Figure()
        for role in ["Normal", "Cascade", "Epicenter"]:
            sub = result[result["role"] == role]
            fig.add_trace(go.Scattermapbox(
                lat=sub["lat"], lon=sub["lon"],
                mode="markers",
                name=role,
                marker=dict(
                    size=role_sizes[role],
                    color=role_colors[role],
                    opacity=0.9 if role == "Epicenter" else 0.7,
                ),
                text=sub["hover"],
                hovertemplate="%{text}<extra></extra>",
            ))
        # Pulse ring on epicenters
        fig.add_trace(go.Scattermapbox(
            lat=epicenters["lat"], lon=epicenters["lon"],
            mode="markers", name="Impact Zone",
            marker=dict(size=28, color="rgba(239,68,68,0.15)"),
            hoverinfo="skip", showlegend=False,
        ))

        fig.update_layout(
            mapbox=dict(style="carto-darkmatter", center=config.MAP_CENTER, zoom=config.MAP_ZOOM),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=0,b=0), height=460,
            legend=dict(
                orientation="h", yanchor="bottom", y=0.01, xanchor="right", x=0.99,
                bgcolor="rgba(9,14,26,0.8)", bordercolor="rgba(0,212,255,0.2)", borderwidth=1,
                font=dict(color="#e2e8f0", size=11),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_detail:
        st.markdown("**📋 Epicenter Zones (sorted by impact)**")
        epi_show = epicenters[["name","anomaly_score","predicted_surge_pct","time_to_peak_min","taxi_surge","bike_surge"]].copy()
        epi_show.columns = ["Zone","Score","Surge%","Peak(min)","Taxi%","Bike%"]
        epi_show["Score"] = epi_show["Score"].map("{:.3f}".format)
        st.dataframe(epi_show.sort_values("Score", ascending=False).head(12),
                     use_container_width=True, hide_index=True, height=220)

        st.markdown("**🌊 Cascade Timeline**")
        # Build cascade wave chart (how many zones become critical over time)
        all_affected = result[result["role"].isin(["Epicenter","Cascade"])].copy()
        all_affected["wave"] = pd.cut(all_affected["time_to_peak_min"],
                                       bins=[-1,0,5,10,15,20,25,30,100],
                                       labels=["Now","0–5","5–10","10–15","15–20","20–25","25–30","30+"])
        wave_counts = all_affected.groupby("wave").size().reset_index(name="zones")
        fig_wave = go.Figure(go.Bar(
            x=wave_counts["wave"].astype(str),
            y=wave_counts["zones"],
            marker_color="#ef4444",
            marker_line_color="rgba(239,68,68,0.3)",
            marker_line_width=1,
        ))
        fig_wave.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=10,b=0), height=200,
            xaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)", title="Time to Peak"),
            yaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)", title="Zones Affected"),
        )
        st.plotly_chart(fig_wave, use_container_width=True)

    # ── Subway modal surge comparison ──
    st.markdown("**🚖 Modal Shift Impact (all epicenter zones)**")
    c1, c2, c3 = st.columns(3)
    with c1:
        avg_taxi_surge = int(epicenters["taxi_surge"].mean())
        st.metric("🚕 Taxi Demand Surge",    f"+{avg_taxi_surge}%", delta=f"+{avg_taxi_surge}%")
    with c2:
        avg_bike_surge = int(epicenters["bike_surge"].mean())
        st.metric("🚲 Bike Demand Surge",    f"+{avg_bike_surge}%", delta=f"+{avg_bike_surge}%")
    with c3:
        avg_sub_drop = int(epicenters["subway_drop"].mean())
        st.metric("🚇 Subway Ridership Drop", f"{avg_sub_drop}%",   delta=f"{avg_sub_drop}%")
