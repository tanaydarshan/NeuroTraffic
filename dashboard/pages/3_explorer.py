"""
NeuroTraffic — Page 3: Historical Explorer
============================================
Owner: Sasmitha S (CB.AI.U4AID24051)

Date range picker → time-series anomaly chart → disruption overlay → zone heatmap
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, sys
from datetime import datetime, date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

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
    .stButton > button { background: linear-gradient(135deg,#00d4ff,#0ea5e9) !important; color: #090e1a !important; border: none !important; border-radius: 8px !important; font-weight: 700 !important; }
    .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px rgba(0,212,255,0.35) !important; }
    .stSelectbox > div > div { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(0,212,255,0.2) !important; border-radius: 8px !important; color: #e2e8f0 !important; }
    .stDateInput > div { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(0,212,255,0.2) !important; border-radius: 8px !important; }
    .stDataFrame { border-radius: 10px !important; }
    .stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.03) !important; border-radius: 10px !important; padding: 4px !important; }
    .stTabs [data-baseweb="tab"] { background: transparent !important; color: #94a3b8 !important; border-radius: 8px !important; }
    .stTabs [aria-selected="true"] { background: rgba(0,212,255,0.15) !important; color: #00d4ff !important; }
    .nt-card { background: rgba(255,255,255,0.035); border: 1px solid rgba(0,212,255,0.12); border-radius: 16px; padding: 1.2rem; margin-bottom: 1rem; }
    ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0d1425; } ::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.3); border-radius: 3px; }
    </style>""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Historical Explorer — NeuroTraffic",
    page_icon="📊",
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
def load_historical(use_hdfs):
    df = hdfs_helper.load_dataframe(config.HISTORICAL_PARQUET, config.HDFS_HISTORICAL_PARQUET, use_hdfs=use_hdfs)
    if df is not None and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=300)
def load_nodes(use_hdfs):
    return hdfs_helper.load_dataframe(config.GRAPH_NODES_CSV, config.HDFS_GRAPH_NODES_CSV, use_hdfs=use_hdfs)

# Known disruption events
DISRUPTION_EVENTS = {
    "2023-01-17": "L train — Full suspension",
    "2023-02-03": "A/C — Signal failure",
    "2023-02-28": "4/5 — Track fire",
    "2023-03-14": "L train — Power outage",
    "2023-04-07": "N/Q — Signal malfunction",
    "2023-04-22": "1 train — Suspension",
    "2023-05-11": "7 train — Derailment",
    "2023-06-01": "J/Z — Service disruption",
    "2023-06-15": "G train — Track work",
}

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1.5rem;">
        <div style="font-size:2.5rem;">📊</div>
        <div style="font-size:1rem;font-weight:800;color:#00d4ff;">Historical Explorer</div>
        <div style="font-size:0.65rem;color:#475569;margin-top:4px;">Jan – Jun 2023 · 263 zones</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("**📅 Date Range**")
    date_start = st.date_input("Start date", value=date(2023, 1, 1),
                               min_value=date(2023, 1, 1), max_value=date(2023, 6, 30),
                               label_visibility="collapsed")
    date_end   = st.date_input("End date",   value=date(2023, 6, 30),
                               min_value=date(2023, 1, 1), max_value=date(2023, 6, 30),
                               label_visibility="collapsed")

    st.markdown("**🏙 Filter by Borough**")
    boroughs = ["All", "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    selected_borough = st.selectbox("Borough", boroughs, label_visibility="collapsed")

    st.markdown("**🚇 Filter by Subway Line**")
    all_lines = ["All"] + list(config.SUBWAY_LINES.keys())
    selected_line = st.selectbox("Line", all_lines, label_visibility="collapsed")

    st.markdown("**📐 Aggregation**")
    agg_level = st.radio("Aggregate by", ["Day", "Week", "Month"], index=0, horizontal=True)

    show_events = st.checkbox("Show disruption events", value=True)
    show_trend  = st.checkbox("Show trend line", value=True)

    st.markdown("---")
    st.page_link("app.py", label="← Back to Home")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:1.8rem;font-weight:800;background:linear-gradient(135deg,#00d4ff,#a78bfa);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0 0 0.2rem;">
📊 Historical Explorer
</h1>
<p style="color:#64748b;font-size:0.8rem;margin:0 0 1rem;">
6-month NYC transport anomaly history · disruption timeline · zone-level heatmap
</p>""", unsafe_allow_html=True)

use_hdfs = st.session_state.get("use_hdfs", config.USE_HDFS)
hist_df  = load_historical(use_hdfs)
nodes_df = load_nodes(use_hdfs)

if hist_df is None:
    st.error("historical_predictions.parquet not found. Run `python data/generate_sample_data.py` first.")
    st.stop()

# ── Filter ──
mask = (hist_df["date"] >= pd.Timestamp(date_start)) & (hist_df["date"] <= pd.Timestamp(date_end))
filtered = hist_df[mask].copy()

if selected_borough != "All":
    if "borough" in filtered.columns:
        filtered = filtered[filtered["borough"] == selected_borough]
    elif nodes_df is not None:
        borough_zones = nodes_df[nodes_df["borough"] == selected_borough]["zone_id"].tolist()
        filtered = filtered[filtered["zone_id"].isin(borough_zones)]

if selected_line != "All" and nodes_df is not None:
    line_zones = nodes_df[nodes_df["primary_line"] == selected_line[0]]["zone_id"].tolist()
    filtered = filtered[filtered["zone_id"].isin(line_zones)]

# ── Aggregate by time ──
agg_col = {"Day": "date", "Week": "week", "Month": "month_label"}
if agg_level == "Week":
    filtered["week"] = filtered["date"].dt.to_period("W").dt.start_time
    agg_key = "week"
elif agg_level == "Month":
    filtered["month_label"] = filtered["date"].dt.to_period("M").dt.start_time
    agg_key = "month_label"
else:
    agg_key = "date"

daily = filtered.groupby(agg_key).agg(
    avg_anomaly=("anomaly_score",    "mean"),
    max_anomaly=("anomaly_score",    "max"),
    n_disruptions=("is_anomaly",     "sum"),
    avg_taxi=("taxi_pickups",        "mean"),
    avg_subway=("subway_ridership",  "mean"),
    avg_bike=("bike_starts",         "mean"),
).reset_index()
daily.rename(columns={agg_key: "period"}, inplace=True)

# ── Summary metrics ──
n_days  = (pd.Timestamp(date_end) - pd.Timestamp(date_start)).days + 1
n_zones = filtered["zone_id"].nunique()
total_anomalies = int(filtered["is_anomaly"].sum())
peak_score = filtered["anomaly_score"].max()

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Days in range",     f"{n_days}")
with c2: st.metric("Zones analysed",    f"{n_zones}")
with c3: st.metric("Total disruptions", f"{total_anomalies:,}")
with c4: st.metric("Peak anomaly score",f"{peak_score:.3f}")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Anomaly Timeline", "🔥 Zone Heatmap", "📋 Disruption Log", "🔄 Predicted vs Actual"])

# ── Tab 1: Time-series ──
with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.06)

    # Main anomaly line
    fig.add_trace(go.Scatter(
        x=daily["period"], y=daily["avg_anomaly"],
        name="Avg Anomaly Score",
        line=dict(color="#00d4ff", width=2),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
        hovertemplate="%{x|%b %d}<br>Avg Score: %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=daily["period"], y=daily["max_anomaly"],
        name="Peak Score",
        line=dict(color="#f97316", width=1.5, dash="dot"),
        hovertemplate="%{x|%b %d}<br>Peak: %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    # Trend line
    if show_trend and len(daily) > 3:
        xs = np.arange(len(daily))
        ys = daily["avg_anomaly"].values
        z  = np.polyfit(xs, ys, 1)
        p  = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=daily["period"], y=p(xs),
            name="Trend",
            line=dict(color="rgba(167,139,250,0.6)", width=1.5, dash="dash"),
            hoverinfo="skip",
        ), row=1, col=1)

    # Disruption event markers
    if show_events:
        for event_date, event_desc in DISRUPTION_EVENTS.items():
            evt = pd.Timestamp(event_date)
            if pd.Timestamp(date_start) <= evt <= pd.Timestamp(date_end):
                fig.add_vline(x=evt, line_dash="dash", line_color="rgba(239,68,68,0.4)",
                              line_width=1.5, row=1, col=1)
                fig.add_annotation(
                    x=evt, y=0.95, yref="y",
                    text=event_desc.split("—")[0].strip(),
                    font=dict(size=9, color="#ef4444"),
                    showarrow=False, textangle=-90,
                    xshift=6, row=1, col=1,
                )

    # Critical threshold band
    fig.add_hrect(y0=0.8, y1=1.05, fillcolor="rgba(239,68,68,0.05)",
                  line_width=0, row=1, col=1)
    fig.add_hline(y=0.8, line_dash="dash", line_color="rgba(239,68,68,0.4)",
                  annotation_text="Critical (0.8)", row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(250,204,21,0.35)",
                  annotation_text="Anomaly (0.5)", row=1, col=1)

    # Disruption count bar
    fig.add_trace(go.Bar(
        x=daily["period"], y=daily["n_disruptions"],
        name="Disruption Count",
        marker_color="#ef4444",
        marker_opacity=0.7,
        hovertemplate="%{x|%b %d}<br>Disruptions: %{y:,}<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=10,b=0), height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    bgcolor="rgba(9,14,26,0.8)", font=dict(color="#e2e8f0",size=11)),
        xaxis2=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)"),
        yaxis= dict(color="#475569", gridcolor="rgba(255,255,255,0.04)", range=[0,1.05], title="Anomaly Score"),
        yaxis2=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)", title="# Disruptions"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Modal transport trend
    if nodes_df is not None:
        st.markdown("**🚖 Modal Transport Trends**")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=daily["period"], y=daily["avg_taxi"],   name="Avg Taxi Pickups",     line=dict(color="#facc15", width=1.5)))
        fig2.add_trace(go.Scatter(x=daily["period"], y=daily["avg_subway"], name="Avg Subway Ridership", line=dict(color="#00d4ff", width=1.5)))
        fig2.add_trace(go.Scatter(x=daily["period"], y=daily["avg_bike"],   name="Avg Bike Starts",      line=dict(color="#34d399", width=1.5)))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=10,b=0), height=220,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                        bgcolor="rgba(9,14,26,0.8)", font=dict(color="#e2e8f0",size=11)),
            xaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)", title="Daily avg (zone)"),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ── Tab 2: Zone heatmap ──
with tab2:
    st.markdown("**🔥 Disruption Heatmap by Zone (top 40 most anomalous zones)**")

    zone_summary = filtered.groupby("zone_id").agg(
        avg_score=("anomaly_score","mean"),
        n_disruptions=("is_anomaly","sum"),
    ).reset_index()

    if "zone_name" in filtered.columns:
        zone_names = filtered[["zone_id","zone_name"]].drop_duplicates().rename(columns={"zone_name":"name"})
        zone_summary = zone_summary.merge(zone_names, on="zone_id", how="left")
    if "name" not in zone_summary.columns and nodes_df is not None:
        zone_summary = zone_summary.merge(nodes_df[["zone_id","name"]], on="zone_id", how="left")
    if "name" not in zone_summary.columns:
        zone_summary["name"] = "Zone " + zone_summary["zone_id"].astype(str)
    if "borough" not in zone_summary.columns:
        if "borough" in filtered.columns:
            bor = filtered[["zone_id","borough"]].drop_duplicates()
            zone_summary = zone_summary.merge(bor, on="zone_id", how="left")
        elif nodes_df is not None:
            zone_summary = zone_summary.merge(nodes_df[["zone_id","borough"]], on="zone_id", how="left")
        else:
            zone_summary["borough"] = "Unknown"

    top40 = zone_summary.nlargest(40, "avg_score")

    # Pivot for heatmap (zones × metric)
    fig_heat = px.bar(
        top40.sort_values("avg_score", ascending=True),
        x="avg_score", y="name",
        color="n_disruptions",
        color_continuous_scale=[[0,"#1e293b"],[0.5,"#f97316"],[1,"#ef4444"]],
        labels={"avg_score":"Avg Anomaly Score","name":"Zone","n_disruptions":"# Disruptions"},
        orientation="h",
        hover_data=["borough","n_disruptions"],
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=10,b=0), height=600,
        xaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(color="#e2e8f0", tickfont=dict(size=10)),
        coloraxis_colorbar=dict(title="# Disruptions", tickfont=dict(color="#94a3b8")),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Borough breakdown
    st.markdown("**🏙 Disruptions by Borough**")
    if nodes_df is not None:
        borough_summary = zone_summary.groupby("borough").agg(
            avg_score=("avg_score","mean"),
            total_disruptions=("n_disruptions","sum"),
        ).reset_index().sort_values("total_disruptions", ascending=False)

        fig_bor = px.bar(
            borough_summary,
            x="borough", y="total_disruptions",
            color="avg_score",
            color_continuous_scale=[[0,"#1e293b"],[0.5,"#f97316"],[1,"#ef4444"]],
            labels={"borough":"Borough","total_disruptions":"Total Disruptions","avg_score":"Avg Score"},
        )
        fig_bor.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=10,b=0), height=260,
            xaxis=dict(color="#475569"), yaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)"),
        )
        st.plotly_chart(fig_bor, use_container_width=True)


# ── Tab 3: Disruption log ──
with tab3:
    st.markdown("**📋 Disruption Event Log**")
    disrupt_log = []
    for event_date, event_desc in DISRUPTION_EVENTS.items():
        evt = pd.Timestamp(event_date)
        if pd.Timestamp(date_start) <= evt <= pd.Timestamp(date_end):
            day_data = filtered[filtered["date"] == evt]
            n_aff = int(day_data["is_anomaly"].sum())
            peak  = float(day_data["anomaly_score"].max()) if len(day_data) > 0 else 0
            disrupt_log.append({
                "Date":        event_date,
                "Event":       event_desc,
                "Zones Affected": n_aff,
                "Peak Score":  f"{peak:.3f}",
                "Day":         evt.strftime("%A"),
            })
    if disrupt_log:
        log_df = pd.DataFrame(disrupt_log)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.info("No disruption events in selected date range")

    st.markdown("---")
    st.markdown("**📥 Export filtered data**")
    csv_data = filtered[["zone_id","date","anomaly_score","is_anomaly","taxi_pickups",
                          "subway_ridership","bike_starts","affected_line"]].to_csv(index=False)
    st.download_button("⬇ Download CSV", csv_data, file_name="neurotraffic_export.csv", mime="text/csv")


# ── Tab 4: Predicted vs Actual ──
with tab4:
    st.markdown("**🔄 Model Accuracy: Predicted vs Actual Disruptions**")

    # Compare prediction vs is_anomaly (ground truth from z-scores)
    comparison = filtered[filtered["is_anomaly"].isin([0,1])].copy()

    tp = int(((comparison["prediction"] == 1) & (comparison["is_anomaly"] == 1)).sum())
    fp = int(((comparison["prediction"] == 1) & (comparison["is_anomaly"] == 0)).sum())
    fn = int(((comparison["prediction"] == 0) & (comparison["is_anomaly"] == 1)).sum())
    tn = int(((comparison["prediction"] == 0) & (comparison["is_anomaly"] == 0)).sum())

    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    f1        = 2 * precision * recall / max(0.001, precision + recall)
    acc       = (tp + tn) / max(1, tp + fp + fn + tn)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Precision", f"{precision:.3f}")
    with c2: st.metric("Recall",    f"{recall:.3f}")
    with c3: st.metric("F1 Score",  f"{f1:.3f}")
    with c4: st.metric("Accuracy",  f"{acc:.3f}")

    # Scatter: predicted_surge_pct vs actual anomaly_score
    scatter_data = comparison[comparison["predicted_surge_pct"] > 0].copy()
    if len(scatter_data) > 0:
        fig_scatter = px.scatter(
            scatter_data.sample(min(2000, len(scatter_data))),
            x="predicted_surge_pct", y="anomaly_score",
            color="is_anomaly",
            color_discrete_map={0:"#22c55e", 1:"#ef4444"},
            labels={"predicted_surge_pct":"Predicted Surge %","anomaly_score":"Actual Anomaly Score","is_anomaly":"Ground Truth"},
            title="Predicted Surge % vs Actual Anomaly Score",
            opacity=0.6,
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=40,b=0), height=380,
            xaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(color="#475569", gridcolor="rgba(255,255,255,0.04)"),
            font=dict(color="#94a3b8"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
