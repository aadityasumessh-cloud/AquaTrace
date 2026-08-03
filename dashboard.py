import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import numpy as np
import webbrowser
import os

# ---------------- COLORS ----------------
PRIMARY   = "#14BFF3"
SECONDARY = "#4CC9F0"
ACCENT    = "#E91717"
BG        = "#050816"
CARD      = "#0F172A"

# ---------------- LOAD DATA ----------------
df = pd.read_csv("water_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

block_map = {"A": "MH1", "B": "MH2", "C": "MH3", "D": "MH4"}
df["block"] = df["block"].map(block_map)

# FIX 2: create leak_detected column if it doesn't exist in the CSV
if "leak_detected" not in df.columns:
    df["leak_detected"] = 0

# ---------------- ML MODEL ----------------
model = IsolationForest(contamination=0.05, random_state=42)
df["ml_anomaly"] = model.fit_predict(df[["total_liters", "flow_rate"]])
df["ml_anomaly"] = df["ml_anomaly"].map({1: 0, -1: 1})

# ---------------- APP ----------------
app = dash.Dash(__name__)

# ---------------- UI CSS ----------------
app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    {%css%}
    <style>
        body {
            background: linear-gradient(135deg, #050816, #0F172A);
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }

        .card {
            background: rgba(255,255,255,0.04);
            border-radius: 14px;
            padding: 18px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.6);
            border: 1px solid rgba(255,255,255,0.08);
            transition: all 0.3s ease-in-out;
        }

        .card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 15px 40px rgba(0,255,255,0.25);
            border: 1px solid #00F5D4;
        }

        .title {
            text-align: center;
            font-size: 34px;
            font-weight: bold;
            margin-bottom: 30px;
            color: #00F5D4;
        }

        .kpi-unit {
            font-size: 12px;
            color: #888;
            margin-top: 2px;
        }

        .Select-control {
            background-color: #0F172A !important;
            border: none !important;
            border-radius: 10px !important;
        }

        .Select-menu-outer {
            background-color: #0F172A !important;
            color: white !important;
        }

        .Select-value-label {
            color: white !important;
        }
    </style>
</head>
<body>
    {%app_entry%}
    {%config%}
    {%scripts%}
    {%renderer%}
</body>
</html>
"""

# ---------------- LAYOUT ----------------
app.layout = html.Div(style={"padding": "30px"}, children=[

    html.Div("💧 Smart Water Governance System", className="title"),

    # FIX 4: interval removed — data is static CSV, interval was pointless.
    # If you switch to a live source, re-add: dcc.Interval(id="interval", interval=3000, n_intervals=0)

    html.Div(id="kpi_cards", style={
        "display": "flex",
        "gap": "15px",
        "marginBottom": "30px"
    }),

    html.Div([
        dcc.Dropdown(
            id="block",
            options=[{"label": f"🏢 {b}", "value": b}
                     for b in sorted(df["block"].dropna().unique())],
            value="MH1",
            style={"width": "48%", "display": "inline-block",
                   "color": "#000"}
        ),
        dcc.Dropdown(
            id="room",
            style={"width": "48%", "display": "inline-block",
                   "marginLeft": "4%", "color": "#000"}
        )
    ], style={"marginBottom": "30px"}),

    html.Div([
        html.Div([
            html.H3("📊 Water Usage Trend", style={"color": SECONDARY}),
            dcc.Graph(id="usage_graph")
        ], className="card", style={"width": "48%"}),

        html.Div([
            html.H3("🧠 Digital Twin", style={"color": PRIMARY}),
            dcc.Graph(id="twin_graph")
        ], className="card", style={"width": "48%"})
    ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "30px"}),

    html.Div([
        html.Div([
            html.H3("🔮 Prediction", style={"color": ACCENT}),
            dcc.Graph(id="prediction_graph")
        ], className="card", style={"width": "48%"}),

        html.Div([
            html.H3("⚙️ Flow Rate", style={"color": PRIMARY}),
            dcc.Graph(id="gauge")
        ], className="card", style={"width": "48%"})
    ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "30px"}),

    html.Div(id="alerts")
])

# ---------------- CALLBACK ----------------
@app.callback(
    [Output("room", "options"),
     Output("room", "value"),
     Output("usage_graph", "figure"),
     Output("twin_graph", "figure"),
     Output("prediction_graph", "figure"),
     Output("gauge", "figure"),
     Output("alerts", "children"),
     Output("kpi_cards", "children")],

    [Input("block", "value"),
     Input("room", "value")]
    # FIX 4: removed interval Input since data is static
)
def update(block, room):

    filtered_block = df[df["block"] == block]

    room_options = [{"label": f"{block} - Room {r}", "value": r}
                    for r in sorted(filtered_block["room_id"].unique())]

    # FIX 3: safely handle None room (first load) and invalid room for block
    valid_rooms = filtered_block["room_id"].values
    if room is None or room not in valid_rooms:
        room = filtered_block["room_id"].iloc[0]

    filtered = filtered_block[filtered_block["room_id"] == room].sort_values("timestamp")

    avg     = round(filtered["total_liters"].mean(), 2)
    max_val = round(filtered["total_liters"].max(),  2)
    min_val = round(filtered["total_liters"].min(),  2)
    flow    = round(filtered["flow_rate"].mean(),    2)

    # FIX 6: KPI cards now show units
    kpis = [
        html.Div([
            html.H4("Avg Usage", style={"margin": 0, "color": "#aaa", "fontSize": "13px"}),
            html.H2(avg, style={"margin": "4px 0 0"}),
            html.Div("liters / reading", className="kpi-unit")
        ], className="card", style={"flex": 1, "textAlign": "center"}),

        html.Div([
            html.H4("Peak Usage", style={"margin": 0, "color": "#aaa", "fontSize": "13px"}),
            html.H2(max_val, style={"margin": "4px 0 0", "color": "#ff6b6b"}),
            html.Div("liters max", className="kpi-unit")
        ], className="card", style={"flex": 1, "textAlign": "center"}),

        html.Div([
            html.H4("Min Usage", style={"margin": 0, "color": "#aaa", "fontSize": "13px"}),
            html.H2(min_val, style={"margin": "4px 0 0", "color": "#00e5b4"}),
            html.Div("liters min", className="kpi-unit")
        ], className="card", style={"flex": 1, "textAlign": "center"}),

        html.Div([
            html.H4("Avg Flow Rate", style={"margin": 0, "color": "#aaa", "fontSize": "13px"}),
            html.H2(flow, style={"margin": "4px 0 0", "color": PRIMARY}),
            html.Div("L / min", className="kpi-unit")
        ], className="card", style={"flex": 1, "textAlign": "center"}),
    ]

    # ── Usage trend ──────────────────────────────────────────────────────────
    fig1 = px.line(filtered, x="timestamp", y="total_liters")
    fig1.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=10, r=10, t=10, b=30),
    )
    fig1.update_traces(line=dict(color=SECONDARY, width=3))

    # ── Digital twin ─────────────────────────────────────────────────────────
    fig2 = px.line(filtered, x="timestamp", y="total_liters")
    fig2.add_hline(
        y=filtered["total_liters"].mean(),
        line_dash="dash",
        line_color="rgba(255,255,255,0.3)",
        annotation_text="Mean",
        annotation_font_color="white",
    )
    fig2.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=10, r=10, t=10, b=30),
    )
    fig2.update_traces(line=dict(color=PRIMARY, width=3))

    # ── Prediction ───────────────────────────────────────────────────────────
    X = np.arange(len(filtered)).reshape(-1, 1)
    y_vals = filtered["total_liters"].values
    model_lr = LinearRegression()
    model_lr.fit(X, y_vals)
    future_X = np.arange(len(filtered), len(filtered) + 10).reshape(-1, 1)
    pred     = model_lr.predict(future_X)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        y=y_vals, name="Actual",
        line=dict(color=SECONDARY, width=2)))
    fig3.add_trace(go.Scatter(
        x=list(range(len(filtered), len(filtered) + 10)),
        y=pred, name="Forecast",
        line=dict(color=ACCENT, dash="dash", width=2)))
    fig3.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=10, b=30),
    )

    # ── Gauge — FIX 5: bg colours now match dark theme ───────────────────────
    flow_max = max(filtered["flow_rate"].max() * 1.3, 10)
    fig4 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=flow,
        number=dict(suffix=" L/min", font=dict(color="white")),
        title={"text": "Avg Flow Rate", "font": {"color": "white"}},
        gauge={
            "axis": {"range": [0, flow_max],
                     "tickcolor": "rgba(255,255,255,0.4)",
                     "tickfont": {"color": "rgba(255,255,255,0.4)"}},
            "bar": {"color": PRIMARY},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,           flow_max * 0.4], "color": "rgba(0,200,100,0.15)"},
                {"range": [flow_max*0.4, flow_max*0.7], "color": "rgba(255,200,0,0.15)"},
                {"range": [flow_max*0.7, flow_max],     "color": "rgba(233,23,23,0.15)"},
            ],
            "threshold": {
                "line": {"color": ACCENT, "width": 2},
                "thickness": 0.75,
                "value": flow_max * 0.7,
            },
        }
    ))
    fig4.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",   # FIX 5: was missing → caused white bg
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=60, b=10),
        height=250,
    )

    # ── Alerts ───────────────────────────────────────────────────────────────
    anomaly_count = int(df["ml_anomaly"].sum())
    leak_count    = int(df["leak_detected"].sum())
    total_issues  = anomaly_count + leak_count

    if total_issues > 0:
        parts = []
        if anomaly_count:
            parts.append(f"{anomaly_count} usage anomal{'ies' if anomaly_count!=1 else 'y'}")
        if leak_count:
            parts.append(f"{leak_count} leak{'s' if leak_count!=1 else ''}")
        alert_box = html.Div(
            f"🚨  {' · '.join(parts)} detected across all blocks",
            style={"background": "rgba(233,23,23,0.15)",
                   "border": f"1px solid {ACCENT}",
                   "borderLeft": f"4px solid {ACCENT}",
                   "padding": "14px 20px",
                   "borderRadius": "10px",
                   "textAlign": "center",
                   "color": "#ff9999"})
    else:
        alert_box = html.Div(
            "✅  All systems stable — no anomalies or leaks detected",
            style={"background": "rgba(0,200,100,0.08)",
                   "border": "1px solid rgba(0,200,100,0.3)",
                   "borderLeft": "4px solid #00e5b4",
                   "padding": "14px 20px",
                   "borderRadius": "10px",
                   "textAlign": "center",
                   "color": "#00e5b4"})

    return room_options, room, fig1, fig2, fig3, fig4, alert_box, kpis


# ---------------- OPEN ONCE ----------------
if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        webbrowser.open("http://127.0.0.1:8050")

    app.run(debug=True)   # FIX 1: removed stray 's' after run()