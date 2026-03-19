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
PRIMARY = "#14BFF3"
SECONDARY = "#4CC9F0"
ACCENT = "#E91717"
BG = "#050816"
CARD = "#0F172A"

# ---------------- LOAD DATA ----------------
df = pd.read_csv("water_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

block_map = {"A": "MH1", "B": "MH2", "C": "MH3", "D": "MH4"}
df["block"] = df["block"].map(block_map)

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

    dcc.Interval(id="interval", interval=3000, n_intervals=0),

    html.Div(id="kpi_cards", style={
        "display": "flex",
        "gap": "15px",
        "marginBottom": "30px"
    }),

    html.Div([
        dcc.Dropdown(
            id="block",
            options=[{"label": f"🏢 {b}", "value": b} for b in df["block"].dropna().unique()],
            value="MH1",
            style={"width": "48%", "display": "inline-block"}
        ),

        dcc.Dropdown(
            id="room",
            style={"width": "48%", "display": "inline-block", "marginLeft": "4%"}
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
     Input("room", "value"),
     Input("interval", "n_intervals")]
)
def update(block, room, _):

    filtered_block = df[df["block"] == block]

    room_options = [{"label": f"{block} - Room {r}", "value": r}
                    for r in filtered_block["room_id"].unique()]

    if room not in filtered_block["room_id"].values:
        room = filtered_block["room_id"].iloc[0]

    filtered = filtered_block[filtered_block["room_id"] == room].sort_values("timestamp")

    avg = round(filtered["total_liters"].mean(), 2)
    max_val = round(filtered["total_liters"].max(), 2)
    min_val = round(filtered["total_liters"].min(), 2)
    flow = round(filtered["flow_rate"].mean(), 2)

    kpis = [
        html.Div([html.H4("Avg"), html.H2(avg)], className="card"),
        html.Div([html.H4("Max"), html.H2(max_val)], className="card"),
        html.Div([html.H4("Min"), html.H2(min_val)], className="card"),
        html.Div([html.H4("Flow"), html.H2(flow)], className="card"),
    ]

    fig1 = px.line(filtered, x="timestamp", y="total_liters")
    fig1.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font=dict(color="white"))
    fig1.update_traces(line=dict(color=SECONDARY, width=3))

    fig2 = px.line(filtered, x="timestamp", y="total_liters")
    fig2.add_hline(y=filtered["total_liters"].mean(), line_dash="dash")
    fig2.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font=dict(color="white"))
    fig2.update_traces(line=dict(color=PRIMARY, width=3))

    X = np.arange(len(filtered)).reshape(-1, 1)
    y = filtered["total_liters"].values

    model_lr = LinearRegression()
    model_lr.fit(X, y)

    future_X = np.arange(len(filtered), len(filtered)+10).reshape(-1,1)
    pred = model_lr.predict(future_X)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(y=y, line=dict(color=SECONDARY)))
    fig3.add_trace(go.Scatter(x=list(range(len(filtered), len(filtered)+10)),
                              y=pred, line=dict(color=ACCENT, dash="dash")))
    fig3.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, font=dict(color="white"))

    fig4 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=flow,
        title={'text': "Flow Rate"},
        gauge={
            'axis': {'range': [0, 150]},
            'bar': {'color': PRIMARY},
            'steps': [
                {'range': [0, 50], 'color': "green"},
                {'range': [50, 100], 'color': "yellow"},
                {'range': [100, 150], 'color': "red"}
            ]
        }
    ))

    alerts = df[(df["ml_anomaly"] == 1) | (df["leak_detected"] == 1)]

    if len(alerts) > 0:
        alert_box = html.Div(f"🚨 {len(alerts)} anomalies detected",
                             style={"background": ACCENT, "padding": "15px",
                                    "borderRadius": "10px", "textAlign": "center"})
    else:
        alert_box = html.Div("✅ System Stable",
                             style={"background": PRIMARY, "padding": "15px",
                                    "borderRadius": "10px", "textAlign": "center"})

    return room_options, room, fig1, fig2, fig3, fig4, alert_box, kpis


# ---------------- OPEN ONCE ----------------
if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        webbrowser.open("http://127.0.0.1:8050")

    app.run(debug=True)