import base64
import html as html_module
import time
import random
from functools import lru_cache
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

_APP_DIR = Path(__file__).resolve().parent
_BRAND_LOGO_PATH = _APP_DIR / "static" / "aquatrace_brand_logo.png"

st.set_page_config(
    page_title="AquaTrace · Operations",
    page_icon=str(_BRAND_LOGO_PATH) if _BRAND_LOGO_PATH.is_file() else "💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  THEME (Light / Dark)
# ─────────────────────────────────────────────────────────────────────────────

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"  # "dark" | "light"

def _is_dark() -> bool:
    return st.session_state.theme_mode == "dark"


@lru_cache(maxsize=1)
def _brand_logo_data_uri() -> str | None:
    if not _BRAND_LOGO_PATH.is_file():
        return None
    raw = _BRAND_LOGO_PATH.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _aquatrace_brand_logo_html(max_height_px: int) -> str:
    """PNG wordmark (falls back to SVG chip if file missing)."""
    uri = _brand_logo_data_uri()
    if uri:
        return (
            f'<img src="{uri}" alt="AquaTrace" role="img" draggable="false" '
            f'style="display:block;height:{max_height_px}px;width:auto;max-width:260px;'
            "object-fit:contain;border-radius:10px;background:#ffffff;padding:5px 10px;"
            'box-sizing:content-box;" />'
        )
    return _aquatrace_logo_svg(min(max_height_px, 36), "aq-fallback")


def _aquatrace_logo_svg(size_px: int, grad_id: str) -> str:
    """Inline wordless mark — droplet + telemetry stroke (single-color-safe IDs)."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size_px}" height="{size_px}" '
        f'viewBox="0 0 40 40" fill="none" role="img" aria-label="AquaTrace">'
        "<defs>"
        f'<linearGradient id="{grad_id}" x1="10" y1="4" x2="28" y2="36" gradientUnits="userSpaceOnUse">'
        '<stop stop-color="#e0f2fe"/><stop offset="0.45" stop-color="#38bdf8"/><stop offset="1" stop-color="#1d4ed8"/>'
        "</linearGradient>"
        "</defs>"
        f'<path fill="url(#{grad_id})" '
        'd="M20 6.2c7.2 11.8 11.2 18.7 11.2 23.2a11.2 11.2 0 01-22.4 0c0-4.5 4.1-11.6 11.2-23.2Z"/>'
        '<path stroke="rgba(255,255,255,0.6)" stroke-width="1.35" stroke-linecap="round" '
        'stroke-linejoin="round" fill="none" '
        'd="M11 26q4.3-6.2 9-6.2 4.9 0 9.2 8"/>'
        "</svg>"
    )


def _sidebar_live_clock_html(tok: dict) -> str:
    """Client-side clock (ticks every second) — survives long Streamlit rerun intervals."""
    inp = tok["sb_input_bg"]
    bdr = tok["sb_card_border"]
    mut = tok["muted"]
    acc = tok["accent"]
    return f"""<div style="margin-top:10px;box-sizing:border-box;border-radius:14px;border:1px solid {bdr};
background:{inp};padding:11px 13px 12px;width:100%;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:600;
              letter-spacing:.09em;text-transform:uppercase;color:{mut};opacity:.94;margin-bottom:5px;">
    Live time · local clock
  </div>
  <div id="aq_sidebar_clock_disp" style="font-family:'JetBrains Mono',monospace;font-size:1rem;
              font-weight:600;color:{acc};letter-spacing:.02em;">--:--:--</div>
</div>
<script>
(function(){{
  const el=document.getElementById("aq_sidebar_clock_disp");
  if(!el)return;
  const pad=n=>String(n).padStart(2,"0");
  function tick(){{
    const d=new Date();
    el.textContent=pad(d.getHours())+":"+pad(d.getMinutes())+":"+pad(d.getSeconds());
  }}
  tick();
  setInterval(tick,1000);
}})();
</script>"""


def _theme_tokens():
    if _is_dark():
        return dict(
            bg="#141c28",
            sidebar_bg="#1a2535",
            text="#f1f5f9",
            muted="#93b4d4",
            border="rgba(255,255,255,.11)",
            sidebar_border="rgba(96,165,250,.28)",
            grid="rgba(255,255,255,.085)",
            tick="#6085a8",
            paper="rgba(0,0,0,0)",
            plot="rgba(0,0,0,0)",
            panel_bg="rgba(255,255,255,.072)",
            sb_card_bg="rgba(56,189,248,0.08)",
            sb_card_border="rgba(147,197,253,0.22)",
            sb_input_bg="rgba(255,255,255,.08)",
            sb_shade="rgba(15,23,42,0.28)",
            accent="#3b82f6",
            mesh_top="rgba(56,189,248,0.20)",
            mesh_bl="rgba(96,165,250,0.14)",
        )
    # Light mode: airy, slightly cooler white
    return dict(
        bg="#f0f7fc",
        sidebar_bg="#ffffff",
        text="#0f172a",
        muted="#4a688a",
        border="rgba(15, 23, 42, .095)",
        sidebar_border="rgba(59,130,246,0.22)",
        grid="rgba(15, 23, 42, .07)",
        tick="#385c7e",
        paper="rgba(0,0,0,0)",
        plot="rgba(0,0,0,0)",
        panel_bg="rgba(255,255,255,0.88)",
        sb_card_bg="rgba(59,130,246,0.07)",
        sb_card_border="rgba(59,130,246,0.18)",
        sb_input_bg="#f8fafc",
        sb_shade="rgba(15,23,42,0.05)",
        accent="#2563eb",
        mesh_top="rgba(56,189,248,0.14)",
        mesh_bl="rgba(59,130,246,0.10)",
    )

# Note: sidebar toggle updates theme on rerun; TOK is evaluated per run.
TOK = _theme_tokens()

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────

_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    background: __BG__;
    color: __TEXT__;
    -webkit-font-smoothing: antialiased;
}
.stApp { background: transparent; }
section[data-testid="stSidebar"] {
    background: __SIDEBAR_BG__ !important;
    border-right: 1px solid __SIDEBAR_BORDER__;
}

/* ── Quiet canvas (no playful motion) ───────────────────────────────────── */
#aquatrace-fish-bg{
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
    background: __BG__;
}
#aquatrace-fish-bg .prof-bg-mesh {
    position: absolute;
    inset: 0;
    background:
      radial-gradient(ellipse 92% 70% at 12% -8%, __MESH_TOP__, transparent 55%),
      radial-gradient(ellipse 75% 55% at 92% 100%, __MESH_BL__, transparent 48%);
}
.stApp, header, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"]{
    position: relative;
    z-index: 1;
}

/* ── Status dot ── */
@keyframes pulse {
    0%,100% { opacity:.95; transform:scale(1); }
    50%      { opacity:.55; transform:scale(1); }
}
.live-dot {
    display:inline-block; width:8px; height:8px; border-radius:50%;
    background:#10b981; margin-right:7px;
    box-shadow:0 0 10px rgba(16,185,129,.65);
    vertical-align: middle;
    animation:pulse 2.4s ease-in-out infinite;
}

/* ── Surfaces ── */
.kpi-card {
    transition: border-color .2s ease, box-shadow .2s ease;
}
.kpi-card:hover {
    border-color: rgba(37,99,235,.22);
    box-shadow: 0 4px 14px rgba(15,23,42,.06);
}
.kpi-card--interactive {
    cursor: default;
    outline: none;
    transition: transform .16s ease, box-shadow .2s ease, border-color .2s ease;
    min-height: 118px;
    box-sizing: border-box;
}
.kpi-card--interactive:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 28px rgba(15,23,42,.14);
    border-color: rgba(59,130,246,.35);
}
.kpi-card--interactive:focus-visible {
    border-color: rgba(59,130,246,.5);
    box-shadow: 0 0 0 2px rgba(59,130,246,.35), 0 8px 24px rgba(15,23,42,.12);
}

/* ── Forecast rows ── */
.forecast-ticker { }
.step-arrow{ display:inline-block; }

/* ── Header ── */
.header-strip {
    display:flex; align-items:center; gap:16px;
    margin-bottom:1.5rem; padding-bottom:1.1rem;
    border-bottom:1px solid __BORDER__;
}
.header-monogram {
    width:48px; height:48px; flex-shrink:0;
    background:linear-gradient(145deg, #1e40af 0%, __ACCENT__ 55%, #0ea5e9 100%);
    border-radius:12px;
    display:flex; align-items:center; justify-content:center;
    overflow:hidden;
    border:1px solid rgba(255,255,255,.22);
    box-shadow:0 6px 22px rgba(37,99,235,.38), inset 0 1px 0 rgba(255,255,255,.28);
}
.header-monogram.header-monogram--brand {
    width:auto;
    height:auto;
    min-height:0;
    padding:0;
    background:transparent;
    border:none;
    box-shadow:none;
    overflow:visible;
}
.header-monogram svg { display:block; }
.header-monogram.header-monogram--brand img {
    display:block;
    box-shadow:0 6px 18px rgba(15,23,42,.22);
}
.header-title { font-size:1.45rem; font-weight:700; letter-spacing:-.03em; color:__TEXT__; }
.header-sub   { font-size:.7rem; color:__MUTED__; letter-spacing:.09em;
                text-transform:uppercase; font-family:'JetBrains Mono',monospace; margin-top:4px;
                font-weight:500; }

/* ── KPI cards ── */
.kpi-card {
    background:__PANEL_BG__;
    border:1px solid __BORDER__;
    border-radius:16px; padding:20px 22px;
    position:relative; overflow:hidden;
}
.kpi-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px; border-radius:16px 16px 0 0;
}
.kpi-blue::before  { background:linear-gradient(90deg,__ACCENT__,#1d4ed8); }
.kpi-green::before { background:linear-gradient(90deg,#10b981,#059669); }
.kpi-amber::before { background:linear-gradient(90deg,#b45309,#92400e); }
.kpi-red::before   { background:linear-gradient(90deg,#dc2626,#991b1b); }
.kpi-slate::before { background:linear-gradient(90deg,#64748b,#475569); }
.kpi-label { font-size:.68rem; font-family:'JetBrains Mono',monospace; letter-spacing:.08em;
             text-transform:uppercase; color:__MUTED__; margin-bottom:8px; font-weight:500; }
.kpi-value { font-size:2rem; font-weight:700; letter-spacing:-.03em; line-height:1; color:__TEXT__; }
.kpi-unit  { font-size:.72rem; color:__MUTED__; font-family:'JetBrains Mono',monospace; margin-top:6px; }
.kpi-icon  { position:absolute; top:18px; right:18px; font-size:1.4rem; opacity:.2; }

/* ── Section labels ── */
.section-label { font-size:.62rem; letter-spacing:.11em; text-transform:uppercase;
                 color:__MUTED__; font-family:'JetBrains Mono',monospace; margin-bottom:2px;
                 font-weight:600; }
.section-title { font-size:1.08rem; font-weight:600; color:__TEXT__; margin-bottom:.75rem; letter-spacing:-.02em; }
.section-desc {
    font-size:.74rem; font-family:'JetBrains Mono',monospace; color:__MUTED__;
    letter-spacing:.04em; font-weight:500; line-height:1.45;
    margin:-0.45rem 0 .85rem 0;
}

/* KPI row: equal column stretch so cards align */
section.main div[data-testid="stHorizontalBlock"]:has(.kpi-card--interactive) {
    align-items: stretch;
}
section.main div[data-testid="stHorizontalBlock"]:has(.kpi-card--interactive) [data-testid="column"] {
    display: flex;
    flex-direction: column;
}
section.main div[data-testid="stHorizontalBlock"]:has(.kpi-card--interactive) [data-testid="column"] > div {
    flex: 1 1 auto;
    min-width: 0;
}

/* Charts: full-width inside columns (avoids slight misalignment between split panes) */
section.main [data-testid="stPlotlyChart"] { width: 100% !important; }

/* ── Info / warn boxes ── */
.info-box { background:rgba(37,99,235,.06); border:1px solid rgba(37,99,235,.14);
            border-left:3px solid __ACCENT__; border-radius:8px; padding:11px 15px;
            font-size:.84rem; color:__TEXT__; font-family:'JetBrains Mono',monospace;
            opacity:0.95; }
.warn-box { background:rgba(180,83,9,.06); border:1px solid rgba(180,83,9,.18);
            border-left:3px solid #b45309; border-radius:8px; padding:11px 15px;
            font-size:.84rem; color:__TEXT__; font-family:'JetBrains Mono',monospace; }
.alert-box{ background:rgba(220,38,38,.07); border:1px solid rgba(220,38,38,.2);
            border-left:3px solid #dc2626; border-radius:8px; padding:11px 15px;
            font-size:.84rem; color:__TEXT__; font-family:'JetBrains Mono',monospace; }

/* ── Prediction rows ── */
.pred-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:9px 14px; border-radius:6px; margin-bottom:5px;
    background:__SB_INPUT_BG__; border:1px solid __BORDER__;
    font-family:'JetBrains Mono',monospace; font-size:.8rem;
}
.pred-step { color:__MUTED__; }
.pred-val  { color:__ACCENT__; font-weight:600; }

/* ── Status chips ── */
.chip {
    background:__PANEL_BG__; border:1px solid __BORDER__;
    border-radius:10px; padding:10px 14px; text-align:center;
    transition: box-shadow .25s ease, transform .25s ease, border-color .25s ease;
}
.chip:hover{
    border-color: rgba(37,99,235,.22);
    box-shadow: 0 2px 10px rgba(15,23,42,.06);
}
.chip-val   { font-size:1.05rem; font-weight:600; letter-spacing:-.01em; }
.chip-label { font-size:.63rem; font-family:'JetBrains Mono',monospace; color:__MUTED__;
              text-transform:uppercase; letter-spacing:.08em; margin-top:3px;
              font-weight:500; }

/* ── Sidebar ── */
.sb-brand {
    padding:.25rem .5rem 1.25rem;
    margin:0 -.5rem 1rem;
    border-bottom:1px solid __SB_CARD_BORDER__;
    background:linear-gradient(180deg, __SB_CARD_BG__ 0%, transparent 100%);
}
/* Brand inside a bordered panel: no extra outer rule */
.sb-brand.sb-brand--panel {
    margin:0 0 .15rem 0;
    padding:.15rem .25rem .6rem;
    border-bottom:none;
    background:transparent;
}
.sb-brand-inner { display:flex; align-items:center; gap:12px; }
.sb-brand-inner.sb-brand-inner--brand { align-items:center; gap:14px; }

.sb-panel-heading {
    font-family:'JetBrains Mono',monospace;
    font-size:.6rem;
    letter-spacing:.11em;
    text-transform:uppercase;
    color:__TEXT__;
    font-weight:600;
    margin:0 0 12px 0;
    padding-bottom:8px;
    border-bottom:1px solid __BORDER__;
    opacity:0.92;
}
/* Streamlit bordered sidebar sections (st.container(border=True)) */
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stSidebar"] [data-testid="StyledVerticalBlockBorderWrapper"] {
    background:linear-gradient(165deg, __SB_CARD_BG__, __SB_INPUT_BG__) !important;
    border:1px solid __SB_CARD_BORDER__ !important;
    border-radius:14px !important;
    padding:12px 14px 14px !important;
    margin-bottom:13px !important;
    box-shadow:0 5px 22px __SB_SHADE__ !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:last-of-type,
section[data-testid="stSidebar"] [data-testid="StyledVerticalBlockBorderWrapper"]:last-of-type {
    margin-bottom:10px !important;
}
.sb-logo {
    flex-shrink:0; width:48px; height:48px;
    border-radius:12px;
    background:linear-gradient(145deg, #1e40af 0%, __ACCENT__ 55%, #0ea5e9 100%);
    display:flex; align-items:center; justify-content:center;
    overflow:hidden;
    border:1px solid rgba(255,255,255,.22);
    box-shadow:0 6px 22px rgba(37,99,235,.34), inset 0 1px 0 rgba(255,255,255,.28);
}
.sb-logo svg { display:block; }
.sb-logo.sb-logo--brand {
    width:auto;
    height:auto;
    padding:0;
    border:none;
    background:transparent;
    box-shadow:none;
    border-radius:0;
    overflow:visible;
    display:flex;
    align-items:center;
    justify-content:flex-start;
}
.sb-name  { font-size:1.08rem; font-weight:700; letter-spacing:-.02em; color:__ACCENT__; line-height:1.15; }
.sb-tag   { font-size:.61rem; font-family:'JetBrains Mono',monospace; color:__MUTED__;
            letter-spacing:.09em; text-transform:uppercase; line-height:1.45; margin-top:4px;
            font-weight:500; }
.sb-live-card {
    margin-top:2px;
    border-radius:14px;
    border:1px solid __SB_CARD_BORDER__;
    background:__SB_CARD_BG__;
    box-shadow:0 8px 28px __SB_SHADE__;
    overflow:hidden;
}
.sb-live-head {
    display:flex; align-items:center; gap:8px;
    padding:10px 14px;
    font-family:'JetBrains Mono',monospace;
    font-size:.56rem; letter-spacing:.09em; text-transform:uppercase;
    color:__MUTED__;
    font-weight:600;
    border-bottom:1px solid __BORDER__;
    background:__SB_INPUT_BG__;
}
.sb-stat-grid { padding:10px 12px 12px; display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.sb-stat {
    border-radius:10px;
    padding:10px 11px;
    background:__SB_INPUT_BG__;
    border:1px solid __BORDER__;
}
.sb-stat-wide { grid-column:1 / -1; }
.sb-stat-label {
    font-family:'JetBrains Mono',monospace; font-size:.56rem;
    letter-spacing:.07em; text-transform:uppercase; color:__MUTED__;
    margin-bottom:4px;
    font-weight:500;
}
.sb-stat-val {
    font-family:'JetBrains Mono',monospace;
    font-size:.98rem; font-weight:600;
    letter-spacing:-.02em; color:__ACCENT__;
    line-height:1.15;
}
.sb-stat-val.sb-accent-green { color:#10b981; }
.sb-stat-val.sb-accent-alert { color:#dc2626; font-weight:700; }

section[data-testid="stSidebar"] .block-container { padding-left: 1rem; padding-right: 1rem; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    border-radius:12px !important;
    border-color:__SIDEBAR_BORDER__ !important;
    background:__SB_INPUT_BG__ !important;
    transition: box-shadow .2s ease, border-color .2s ease !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"]:focus-within > div {
    box-shadow:0 0 0 2px rgba(37,99,235,.2) !important;
    border-color:rgba(37,99,235,.45) !important;
}
section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {
    font-size:.7rem !important;
    letter-spacing:.08em !important;
    text-transform:uppercase !important;
    font-family:'JetBrains Mono',monospace !important;
    color:__MUTED__ !important;
    font-weight:600 !important;
}
hr { border:none; border-top:1px solid __BORDER__ !important; margin:1.25rem 0 !important; }
</style>
"""

_css = (
    _css.replace("__BG__", TOK["bg"])
        .replace("__SIDEBAR_BG__", TOK["sidebar_bg"])
        .replace("__TEXT__", TOK["text"])
        .replace("__MUTED__", TOK["muted"])
        .replace("__BORDER__", TOK["border"])
        .replace("__SIDEBAR_BORDER__", TOK["sidebar_border"])
        .replace("__PANEL_BG__", TOK["panel_bg"])
        .replace("__SB_CARD_BG__", TOK["sb_card_bg"])
        .replace("__SB_CARD_BORDER__", TOK["sb_card_border"])
        .replace("__SB_INPUT_BG__", TOK["sb_input_bg"])
        .replace("__SB_SHADE__", TOK["sb_shade"])
        .replace("__ACCENT__", TOK["accent"])
        .replace("__MESH_TOP__", TOK["mesh_top"])
        .replace("__MESH_BL__", TOK["mesh_bl"])
)

st.markdown(_css, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  BACKGROUND — static mesh (professional, zero motion)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    "<div id='aquatrace-fish-bg'><div class='prof-bg-mesh' aria-hidden='true'></div></div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
#  DATA ENGINE
# ─────────────────────────────────────────────────────────────────────────────

BLOCKS   = ["MH1", "MH2", "MH3", "MH4"]
FLOORS   = list(range(1, 11))
MAX_ROWS = 5000   # keep up to 5000 rows so history accumulates

# Fixed auto-refresh for the Streamlit rerun loop (no UI control).
REFRESH_INTERVAL_SEC = 120


def _base_usage(hour: int) -> float:
    morning = 60 * np.exp(-0.5 * ((hour - 7.5) / 1.5) ** 2)
    evening = 50 * np.exp(-0.5 * ((hour - 20)  / 2.0) ** 2)
    return max(5 + morning + evening, 1.0)


def _generate_row(ts: datetime, block=None, floor=None) -> dict:
    block    = block or random.choice(BLOCKS)
    floor    = floor or random.choice(FLOORS)
    base     = _base_usage(ts.hour) * random.uniform(0.85, 1.15)
    is_spike = random.random() < 0.08
    liters   = base * (random.uniform(3, 5) if is_spike else 1.0)
    flow     = liters / random.uniform(3.5, 6.5)
    return {
        "timestamp":    ts,
        "block":        block,
        "floor":        floor,
        "total_liters": round(liters, 2),
        "flow_rate":    round(flow, 3),
    }


def _seed_history() -> list:
    """Seed 70 readings per block+floor combo spread over the last 7 hours
    so users can scroll back through meaningful history on startup."""
    now  = datetime.now()
    rows = []
    for b in BLOCKS:
        for f in FLOORS:
            for i in range(70, 0, -1):
                ts = now - timedelta(minutes=i * 6)   # one reading every 6 min
                rows.append(_generate_row(ts, block=b, floor=f))
    rows.sort(key=lambda r: r["timestamp"])
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE — shared store between the UI and background thread
# ─────────────────────────────────────────────────────────────────────────────

if "rows" not in st.session_state:
    st.session_state.rows = _seed_history()

if "loop_counter" not in st.session_state:
    st.session_state.loop_counter = 0

if "live_idx" not in st.session_state:
    # Index into (block,floor) combos for live generation.
    st.session_state.live_idx = 0
if "last_live_ts" not in st.session_state:
    # Timestamp of last live generation tick.
    st.session_state.last_live_ts = time.time()

_ISO_MODEL_VER = 6  # bumped: removed forced spikes, tuned 50/50 window
if st.session_state.get("_iso_model_ver") != _ISO_MODEL_VER:
    # Train once on synthetic data from the same generator as live readings,
    # so most points count as normal and spikes are occasional outliers.
    rng = np.random.default_rng(42)
    anchor = datetime.now().replace(second=0, microsecond=0)
    n = 6000
    X = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        ts = anchor - timedelta(
            hours=int(rng.integers(0, 96)),
            minutes=int(rng.integers(0, 60)),
        )
        row = _generate_row(ts)
        X[i, 0] = row["total_liters"]
        X[i, 1] = row["flow_rate"]
    _iso = IsolationForest(contamination=0.07, random_state=42, n_estimators=150)
    _iso.fit(X)
    st.session_state.ui_anomaly_detector = _iso
    # Pre-compute the score threshold from training data:
    # Use the 7th-percentile score (top ~7% most anomalous during training)
    # as a stable cut-off that doesn't shift with each filtered subset.
    train_scores = _iso.decision_function(X)
    st.session_state.iso_score_threshold = float(np.percentile(train_scores, 7))
    st.session_state._iso_model_ver = _ISO_MODEL_VER

if "hour_dive_snap" not in st.session_state:
    # Last stable hour-inspector values (avoids empty-state flicker on rapid reruns).
    st.session_state.hour_dive_snap = {}

# ─────────────────────────────────────────────────────────────────────────────
#  PLOTLY THEME HELPER
# ─────────────────────────────────────────────────────────────────────────────

PLOT_BG   = TOK["plot"]
GRID_COL  = TOK["grid"]
TICK_COL  = TOK["tick"]
FONT_MONO = "JetBrains Mono, monospace"
FONT_SANS = "Inter, system-ui, sans-serif"


def _theme(fig, height=300, margin=None):
    m = margin or dict(l=10, r=10, t=30, b=30)
    fig.update_layout(
        height=height,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PLOT_BG,
        font=dict(family=FONT_MONO, color=TICK_COL, size=11),
        margin=m,
        xaxis=dict(gridcolor=GRID_COL, linecolor="rgba(255,255,255,.07)",
                   tickcolor=GRID_COL, zeroline=False),
        yaxis=dict(gridcolor=GRID_COL, linecolor="rgba(255,255,255,.07)",
                   zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,.08)",
                    font=dict(size=11, color=TICK_COL)),
        transition=dict(duration=450, easing="cubic-in-out"),
    )
    # Hint plotly.js to animate between updates.
    for tr in fig.data:
        try:
            tr.update(transition=dict(duration=450, easing="cubic-in-out"))
        except Exception:
            pass
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_trend(filtered: pd.DataFrame) -> go.Figure:
    """Show full history with a range slider; default view = last 3 hours."""
    filtered = filtered.sort_values("timestamp")
    norm = filtered[filtered["anomaly"] == 0]
    anom = filtered[filtered["anomaly"] == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=norm["timestamp"], y=norm["total_liters"],
        mode="lines", name="Normal",
        line=dict(color="#2563eb", width=1.6, shape="spline", smoothing=0.6),
        fill="tozeroy", fillcolor="rgba(37,99,235,.08)",
    ))
    if not anom.empty:
        fig.add_trace(go.Scatter(
            x=anom["timestamp"], y=anom["total_liters"],
            mode="markers", name="Anomaly",
            marker=dict(color="#dc2626", size=7, symbol="circle-open",
                        line=dict(width=1.5, color="#dc2626")),
        ))

    fig.update_layout(title=None)
    _theme(fig, height=300, margin=dict(l=10, r=10, t=10, b=60))

    # Range slider so older data is scrollable
    if len(filtered) > 0:
        end_ts   = filtered["timestamp"].max()
        start_ts = end_ts - timedelta(hours=3)   # default view: last 3 h
        fig.update_xaxes(
            range=[start_ts, end_ts],
            rangeslider=dict(visible=True, thickness=0.08,
                             bgcolor=TOK["panel_bg"],
                             bordercolor=TOK["border"], borderwidth=1),
        )
    return fig


def build_flow_rate(filtered: pd.DataFrame) -> go.Figure:
    """Show full flow-rate history with a range slider; default view = last 3 hours."""
    filtered = filtered.sort_values("timestamp")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered["timestamp"], y=filtered["flow_rate"],
        mode="lines", name="Flow rate",
        line=dict(color="#0d9488", width=1.5, shape="spline", smoothing=0.7),
        fill="tozeroy", fillcolor="rgba(13,148,136,.07)",
    ))

    _theme(fig, height=240, margin=dict(l=10, r=10, t=10, b=60))

    if len(filtered) > 0:
        end_ts   = filtered["timestamp"].max()
        start_ts = end_ts - timedelta(hours=3)
        fig.update_xaxes(
            range=[start_ts, end_ts],
            rangeslider=dict(visible=True, thickness=0.08,
                             bgcolor=TOK["panel_bg"],
                             bordercolor=TOK["border"], borderwidth=1),
        )
    return fig


def build_hourly_bar(filtered: pd.DataFrame) -> go.Figure:
    hourly = filtered.groupby("hour")["total_liters"].mean().reindex(range(24), fill_value=0)
    mx = max(float(hourly.max()), 1e-6)
    colors = [
        f"rgba(37,99,235,{round(0.22 + 0.58 * float(v) / mx, 3)})"
        for v in hourly.values
    ]
    fig = go.Figure(go.Bar(
        x=hourly.index, y=hourly.values,
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="Hour %{x}:00<br>Avg %{y:.1f} L<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Hour", yaxis_title="Avg L", showlegend=False)
    _theme(fig, height=250, margin=dict(l=10, r=10, t=10, b=40))
    return fig


def build_block_bar(df: pd.DataFrame, selected_block: str) -> go.Figure:
    usage  = df.groupby("block")["total_liters"].mean().sort_values(ascending=True)
    colors = ["#dc2626" if b == usage.idxmax() else
              ("#2563eb" if b == selected_block else "#475569")
              for b in usage.index]
    fig = go.Figure(go.Bar(
        x=usage.values, y=usage.index, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="%{y}: %{x:.1f} L<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Avg Liters", showlegend=False)
    _theme(fig, height=230, margin=dict(l=10, r=10, t=10, b=30))
    return fig


def build_gauge(value: float, max_val: float, title: str, unit: str,
                lo: float, hi: float) -> go.Figure:
    if value <= lo:
        bar_color, status, sc = "#2563eb", "NORMAL",   "#10b981"
    elif value <= hi:
        bar_color, status, sc = "#b45309", "ELEVATED", "#b45309"
    else:
        bar_color, status, sc = "#dc2626", "CRITICAL", "#dc2626"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 2),
        number=dict(suffix=f" {unit}", font=dict(size=22, color=TICK_COL, family=FONT_SANS)),
        gauge=dict(
            axis=dict(range=[0, max_val], tickwidth=1,
                      tickcolor="rgba(255,255,255,.15)",
                      tickfont=dict(color=TICK_COL, size=9, family=FONT_MONO),
                      nticks=5),
            bar=dict(color=bar_color, thickness=0.5),
            bgcolor=PLOT_BG, borderwidth=0,
            steps=[
                dict(range=[0, lo],       color="rgba(37,99,235,.06)"),
                dict(range=[lo, hi],      color="rgba(180,83,9,.06)"),
                dict(range=[hi, max_val], color="rgba(220,38,38,.07)"),
            ],
            threshold=dict(line=dict(color="#dc2626", width=2),
                           thickness=0.75, value=hi),
        ),
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:.6em;color:{sc};letter-spacing:.12em;'>{status}</span>",
            font=dict(size=11, color=TICK_COL, family=FONT_SANS),
        ),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        margin=dict(l=20, r=20, t=60, b=10), height=210,
        font=dict(family=FONT_MONO),
    )
    return fig


def live_gauge_component(
    gauge_id: str,
    title: str,
    value: float,
    unit: str,
    max_val: float,
    status: str,
    accent: str,
):
    # Browser-side 60fps animation so the gauge is always "alive".
    # No external JS libs; safe for Streamlit components.
    v = float(value) if value is not None else 0.0
    mv = float(max_val) if max_val and max_val > 0 else 1.0
    bg = TOK["bg"]
    border = TOK["border"]
    text = TOK["text"]
    muted = TOK["muted"]
    panel = TOK["panel_bg"]

    html = f"""
    <div style="width:100%; height:220px; border-radius:12px; position:relative;
                background: {panel}; box-shadow: inset 0 0 0 1px {border};">
      <div style="position:absolute; left:14px; top:10px; right:14px;">
        <div style="font-family:'Inter',system-ui,sans-serif; font-weight:600; font-size:12px; color:{muted}; letter-spacing:.01em;">
          {title}
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:9px; letter-spacing:.1em; text-transform:uppercase;
                    color:{accent}; margin-top:3px; font-weight:600;">
          {status}
        </div>
      </div>
      <canvas id="{gauge_id}" width="520" height="220"
        style="width:100%; height:220px; border-radius:12px; display:block;"></canvas>
      <div id="{gauge_id}_num" style="
          position:absolute; left:0; right:0; bottom:26px;
          text-align:center;
          font-family:'Inter',system-ui,sans-serif; font-weight:700;
          font-size:28px; color:{text};
          letter-spacing:-.02em;
        "></div>
    </div>
    <script>
      (function(){{
        const canvas = document.getElementById('{gauge_id}');
        const num = document.getElementById('{gauge_id}_num');
        const ctx = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;

        const maxV = {mv:.6f};
        const target = {v:.6f};
        const unit = "{unit}";
        const accent = "{accent}";
        const bg = "{bg}";

        // live state
        let cur = target;
        let vel = 0;
        let t0 = performance.now();

        const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
        const lerp=(a,b,t)=>a+(b-a)*t;

        function hexToRgb(h){{
          const m = h.replace('#','');
          const n = parseInt(m, 16);
          return [ (n>>16)&255, (n>>8)&255, n&255 ];
        }}
        const rgb = hexToRgb(accent);

        function draw(t){{
          const dt = Math.min(0.032, (t - t0)/1000);
          t0 = t;

          // spring toward target (smooth, always moving)
          const k = 18.0;
          const c = 9.5;
          const x = cur - target;
          const a = -k*x - c*vel;
          vel += a * dt;
          cur += vel * dt;

          // micro fluctuation (keeps it alive) — scales with value
          const wob = (0.018 + 0.035*(target/maxV)) * maxV;
          const micro = wob * Math.sin(t*0.0022) + wob*0.6*Math.sin(t*0.0011 + 1.2);
          const shown = clamp(cur + micro, 0, maxV);

          // clear
          ctx.clearRect(0,0,W,H);

          // glow background
          const gx = W*0.5, gy = H*0.95;
          const gr = ctx.createRadialGradient(gx,gy,0,gx,gy,W*0.75);
          gr.addColorStop(0, 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',0.18)');
          gr.addColorStop(1, 'rgba(0,0,0,0)');
          ctx.fillStyle = gr;
          ctx.fillRect(0,0,W,H);

          // arc base
          const cx = W/2, cy = H*0.90;
          const R = Math.min(W*0.40, H*0.75);
          const a0 = Math.PI*1.05, a1 = Math.PI*1.95;
          ctx.lineCap = 'round';
          ctx.lineWidth = 18;
          ctx.strokeStyle = 'rgba(255,255,255,0.06)';
          ctx.beginPath();
          ctx.arc(cx, cy, R, a0, a1);
          ctx.stroke();

          // value arc
          const p = clamp(shown/maxV, 0, 1);
          const ang = lerp(a0, a1, p);
          const lg = ctx.createLinearGradient(cx-R,0,cx+R,0);
          lg.addColorStop(0, 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',0.55)');
          lg.addColorStop(1, 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',0.95)');
          ctx.strokeStyle = lg;
          ctx.shadowColor = 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',0.35)';
          ctx.shadowBlur = 22;
          ctx.beginPath();
          ctx.arc(cx, cy, R, a0, ang);
          ctx.stroke();
          ctx.shadowBlur = 0;

          // tick sparkle
          const sx = cx + Math.cos(ang)*R;
          const sy = cy + Math.sin(ang)*R;
          const sp = ctx.createRadialGradient(sx,sy,0,sx,sy,18);
          sp.addColorStop(0, 'rgba(255,255,255,0.90)');
          sp.addColorStop(1, 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',0.0)');
          ctx.fillStyle = sp;
          ctx.beginPath();
          ctx.arc(sx, sy, 10, 0, Math.PI*2);
          ctx.fill();

          // number (avoid template literals to keep Python f-strings safe)
          const decimals = (unit === '%' || unit === 'L' || unit === 'L/min') ? 1 : 2;
          num.textContent = shown.toFixed(decimals) + " " + unit;

          requestAnimationFrame(draw);
        }}
        requestAnimationFrame(draw);
      }})();
    </script>
    """
    components.html(html, height=230, scrolling=False)


def build_scatter(filtered: pd.DataFrame) -> go.Figure:
    norm = filtered[filtered["anomaly"] == 0]
    anom = filtered[filtered["anomaly"] == 1]
    fig  = go.Figure()
    fig.add_trace(go.Scatter(
        x=norm["flow_rate"], y=norm["total_liters"],
        mode="markers", name="Normal",
        marker=dict(color="rgba(37,99,235,.42)", size=5),
    ))
    if not anom.empty:
        fig.add_trace(go.Scatter(
            x=anom["flow_rate"], y=anom["total_liters"],
            mode="markers", name="Anomaly",
            marker=dict(color="#dc2626", size=7, symbol="x", line=dict(width=1.5)),
        ))
    fig.update_layout(xaxis_title="Flow Rate (L/min)", yaxis_title="Total Liters")
    _theme(fig, height=240, margin=dict(l=10, r=10, t=10, b=40))
    return fig


def build_floor_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.groupby(["block", "floor"])["total_liters"].mean().reset_index()
    matrix = pivot.pivot(index="floor", columns="block", values="total_liters").fillna(0)
    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=matrix.columns.tolist(),
        y=[f"F{f}" for f in matrix.index.tolist()],
        colorscale=[[0, "#0f172a"], [0.45, "#1e40af"], [1, "#3b82f6"]],
        showscale=True,
        hovertemplate="Block %{x} · %{y}<br>Avg %{z:.1f} L<extra></extra>",
    ))
    _theme(fig, height=280, margin=dict(l=40, r=10, t=10, b=30))
    return fig


def build_anomaly_timeline(filtered: pd.DataFrame) -> go.Figure:
    """Show full anomaly history with a range slider; default view = last 3 hours."""
    filtered = filtered.sort_values("timestamp").copy()
    anomalies = filtered[filtered["anomaly"] == 1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=filtered["timestamp"],
        y=filtered["total_liters"],
        marker_color="rgba(37,99,235,.09)",
        hovertemplate="%{x}<br>%{y:.1f} L<extra></extra>",
        showlegend=False,
    ))
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["timestamp"],
            y=anomalies["total_liters"],
            mode="markers",
            name="Anomaly",
            marker=dict(color="#dc2626", size=8, symbol="circle-open",
                        line=dict(width=1.5, color="#dc2626")),
            hovertemplate="%{x}<br>%{y:.1f} L<extra></extra>",
        ))

    fig.update_layout(showlegend=False, title=None)
    _theme(fig, height=230, margin=dict(l=10, r=10, t=10, b=60))

    if len(filtered) > 0:
        end_ts   = filtered["timestamp"].max()
        start_ts = end_ts - timedelta(hours=3)
        fig.update_xaxes(
            range=[start_ts, end_ts],
            rangeslider=dict(visible=True, thickness=0.08,
                             bgcolor=TOK["panel_bg"],
                             bordercolor=TOK["border"], borderwidth=1),
        )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  KPI / CHIP HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def kpi_html(cls, icon, label, value, unit):
    return f"""
    <div class="kpi-card {cls}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-unit">{unit}</div>
    </div>"""


def kpi_metric_box(card_class: str, title: str, unit_label: str, value_display: str, accent_hex: str) -> str:
    """Boxed KPI with hover/focus affordances (keyboard: Tab into card)."""
    safe_title = html_module.escape(title, quote=True)
    safe_val = html_module.escape(value_display, quote=False)
    return f"""
<div class="kpi-card {card_class} kpi-card--interactive" tabindex="0" role="group" aria-label="{safe_title}">
  <div class="kpi-label">{title}</div>
  <div class="kpi-value" style="color:{accent_hex};">{safe_val}</div>
  <div class="kpi-unit">{unit_label}</div>
</div>"""


def chip_html(label, value, unit, color):
    return f"""
    <div class="chip" style="border-top:2px solid {color};">
        <div class="chip-val" style="color:{color};">{value} <span style="font-size:.68rem;color:{TOK['muted']};">{unit}</span></div>
        <div class="chip-label">{label}</div>
    </div>"""


def pred_html(step, direction, val, col):
    return f"""
    <div class="pred-row">
        <span class="pred-step">Step +{step}</span>
        <span class="step-arrow" style="color:{col};font-size:.8rem;">{direction}</span>
        <span class="pred-val">{val:.2f} L</span>
    </div>"""


def _hour_circular_dist(h1: int, h2: int) -> int:
    d = abs(int(h1) - int(h2))
    return min(d, 24 - d)


def digital_twin_component(df: pd.DataFrame):
    """2D digital twin: block × floor demand heatmap (no WebGL 3D) — calmer updates for Streamlit reruns."""
    if df is None or df.empty:
        st.info("No readings available yet for the digital twin.")
        return

    pivot = df.groupby(["block", "floor"])["total_liters"].mean().reset_index()
    matrix = pivot.pivot(index="floor", columns="block", values="total_liters").fillna(0)
    matrix = matrix.reindex(index=FLOORS, columns=BLOCKS).fillna(0)

    z = matrix.values.astype(float)
    z_txt = [[f"{v:.1f}" if v > 0 else "" for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(BLOCKS),
            y=[f"F{f}" for f in FLOORS],
            text=z_txt,
            texttemplate="%{text}",
            textfont=dict(size=10, color="rgba(240,246,255,0.92)"),
            colorscale=[
                [0.0, "#0f172a"],
                [0.35, "#1e3a8a"],
                [0.65, "#2563eb"],
                [1.0, "#38bdf8"],
            ],
            showscale=True,
            colorbar=dict(
                title=dict(text="Avg L", side="right"),
                tickfont=dict(size=10, color=TICK_COL),
                thickness=12,
            ),
            hovertemplate="Block %{x} · %{y}<br>Avg %{z:.1f} L<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=20, t=8, b=36),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_MONO, color=TICK_COL, size=11),
        xaxis=dict(title="Block", gridcolor=GRID_COL, linecolor="rgba(255,255,255,.07)", side="bottom"),
        yaxis=dict(title="Floor", gridcolor=GRID_COL, linecolor="rgba(255,255,255,.07)", autorange="reversed"),
        # Keeps hover/zoom stable across rapid Streamlit reruns (reduces visible flicker).
        uirevision="aquatrace_twin_2d",
        transition=dict(duration=0),
    )

    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key="twin_2d")


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR — only rendered once (static controls)
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    with st.container(border=True):
        st.markdown(f"""
        <div class="sb-brand sb-brand--panel">
          <div class="sb-brand-inner sb-brand-inner--brand">
            <div class="sb-logo sb-logo--brand">{_aquatrace_brand_logo_html(78)}</div>
            <div>
              <div class="sb-tag">Facility water · operations console</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<p class="sb-panel-heading">Display</p>', unsafe_allow_html=True)
        is_dark_ui = st.toggle("Dark interface", value=_is_dark())
        st.session_state.theme_mode = "dark" if is_dark_ui else "light"

    with st.container(border=True):
        st.markdown('<p class="sb-panel-heading">Location filter</p>', unsafe_allow_html=True)
        sel_block = st.selectbox("Hostel block", BLOCKS)
        sel_floor = st.selectbox("Floor level", FLOORS)

    info_placeholder = st.empty()
    sidebar_clock_ph = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LAYOUT — static containers, filled in the loop
# ─────────────────────────────────────────────────────────────────────────────

header_ph = st.empty()

# KPI row
kpi_ph = st.empty()

# Alert
alert_ph = st.empty()

st.markdown("---")

# ── Gauges ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Metrics</div>'
            '<div class="section-title">Key indicators</div>',
            unsafe_allow_html=True)
g_cols     = st.columns(3, gap="medium")
gauge_ph   = [c.empty() for c in g_cols]

# ── Chips ───────────────────────────────────────────────────────────────────
chip_ph = st.empty()

st.markdown("---")

# ── Consumption trend ───────────────────────────────────────────────────────
st.markdown('<div class="section-label">Time series</div>'
            '<div class="section-title">Consumption trend</div>',
            unsafe_allow_html=True)
trend_ph = st.empty()

st.markdown("---")

# ── Digital twin (block × floor heatmap) ────────────────────────────────────
st.markdown(
    '<div class="section-label">Facility</div>'
    '<div class="section-title">Digital twin</div>'
    '<div class="section-desc">Block × floor average demand · hover cells for details</div>',
    unsafe_allow_html=True,
)
twin_main_ph = st.empty()

# ── Flow rate trend (chart only — no infographic canvas) ────────────────────
st.markdown('<div class="section-label">Flow</div>'
            '<div class="section-title">Flow rate</div>',
            unsafe_allow_html=True)
flow_ph = st.empty()

st.markdown("---")

# ── Hourly bar + block comparison ────────────────────────────────────────────
col_l, col_r = st.columns([1.1, 1], gap="medium")
with col_l:
    st.markdown('<div class="section-label">Distribution</div>'
                '<div class="section-title">Hourly consumption</div>',
                unsafe_allow_html=True)
    hourly_ph = col_l.empty()

with col_r:
    st.markdown('<div class="section-label">Benchmark</div>'
                '<div class="section-title">Block comparison</div>',
                unsafe_allow_html=True)
    block_ph = col_r.empty()

st.markdown("---")

# ── Scatter + heatmap ───────────────────────────────────────────────────────
col_s, col_h = st.columns(2, gap="medium")
with col_s:
    st.markdown('<div class="section-label">Analysis</div>'
                '<div class="section-title">Flow vs consumption</div>',
                unsafe_allow_html=True)
    scatter_ph = col_s.empty()
with col_h:
    st.markdown('<div class="section-label">Heatmap</div>'
                '<div class="section-title">Demand by block and floor</div>',
                unsafe_allow_html=True)
    heatmap_ph = col_h.empty()

st.markdown("---")

# ── Anomaly timeline ────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Monitoring</div>'
            '<div class="section-title">Anomaly timeline</div>',
            unsafe_allow_html=True)
anom_tl_ph = st.empty()

st.markdown("---")

# ── Forecast + hour deep dive ────────────────────────────────────────────────
col_f, col_d = st.columns(2, gap="medium")
with col_f:
    st.markdown('<div class="section-label">Projection</div>'
                '<div class="section-title">Short-term outlook</div>',
                unsafe_allow_html=True)
    forecast_ph = col_f.empty()

with col_d:
    st.markdown('<div class="section-label">Detail</div>'
                '<div class="section-title">Hour inspector</div>',
                unsafe_allow_html=True)
    selected_hour = st.select_slider("Hour (0–23)", options=list(range(24)), value=8)
    hour_ph = col_d.empty()

# ─────────────────────────────────────────────────────────────────────────────
#  REAL-TIME LOOP  — rerun every REFRESH_INTERVAL_SEC (fixed)
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_CONFIG = dict(displayModeBar=False)

# ── Single-pass render (st.rerun() at the bottom drives the live loop) ──────
st.session_state.loop_counter += 1

if True:

    # ── Live data tick (thread-free) ─────────────────────────────────────────
    # Streamlit session_state is not safe to mutate from background threads.
    # Instead, append a few new rows each rerun based on elapsed time.
    now_ts = time.time()
    elapsed = max(0.0, now_ts - float(st.session_state.last_live_ts))
    # Roughly match the old simulator: a full combo round ~4s (40 combos, 0.1s each).
    # This generates rows deterministically and keeps filters "live".
    steps = int(min(20, max(1, round(elapsed / 0.12))))
    combos = [(b, f) for b in BLOCKS for f in FLOORS]
    for _ in range(steps):
        b, f = combos[st.session_state.live_idx % len(combos)]
        st.session_state.rows.append(_generate_row(datetime.now(), block=b, floor=f))
        st.session_state.live_idx += 1
    if len(st.session_state.rows) > MAX_ROWS:
        st.session_state.rows = st.session_state.rows[-MAX_ROWS:]
    st.session_state.last_live_ts = now_ts

    # ── Pull latest data snapshot ──────────────────────────────────────────
    df = pd.DataFrame(st.session_state.rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"]      = df["timestamp"].dt.hour

    filtered = df[(df["block"] == sel_block) & (df["floor"] == sel_floor)].copy()
    filtered = filtered.sort_values("timestamp").reset_index(drop=True)

    # ── Anomaly detection (time-aware threshold) ──────────────────────────
    # Strategy: compare each reading against 2.2× the expected base consumption
    # for its hour-of-day (from _base_usage). This is robust because:
    #   • Spikes from _generate_row are 3-5× base  → always exceed 2.2× threshold
    #   • Normal readings are 0.85-1.15× base      → never exceed 2.2× threshold
    #   • Works even when the dataset has no historical spikes (no std needed)
    #   • Deterministic: same reading → same label every rerun (no flicker)
    SPIKE_MULT = 2.2
    filtered["_exp_base"] = filtered["hour"].apply(lambda h: _base_usage(int(h)))
    filtered["anomaly"] = (
        filtered["total_liters"] > filtered["_exp_base"] * SPIKE_MULT
    ).astype(int)
    filtered.drop(columns=["_exp_base"], inplace=True)

    # Alert count: last 8 readings only.
    # 8% spike rate in an 8-row window → P(0 spikes) ≈ 0.92⁸ ≈ 51%.
    # So roughly half the time the alert is quiet, half the time it fires.
    # This gives the natural "sometimes normal, sometimes anomaly" rhythm.
    LIVE_WINDOW = 8
    recent_tail = filtered.tail(LIVE_WINDOW)
    anomaly_count = int(recent_tail["anomaly"].sum())

    # Sliding window on latest points (no playback / playhead animation)
    trend_window = 80
    trend_end_idx = None
    timeline_end_idx = None

    # ── Header ─────────────────────────────────────────────────────────────
    header_ph.markdown(f"""
    <div class="header-strip">
        <div class="header-monogram header-monogram--brand">{_aquatrace_brand_logo_html(46)}</div>
        <div>
            <div class="header-title">Operations overview</div>
            <div class="header-sub">
                <span class="live-dot"></span>
                Live data · Block {sel_block} · Floor {sel_floor}
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── KPI cards ──────────────────────────────────────────────────────────
    avg_l = round(filtered["total_liters"].mean(), 2) if len(filtered) else 0
    max_l = round(filtered["total_liters"].max(),  2) if len(filtered) else 0
    min_l = round(filtered["total_liters"].min(),  2) if len(filtered) else 0
    avg_f = round(filtered["flow_rate"].mean(),    3) if len(filtered) else 0

    with kpi_ph.container():
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        kpi_flash = anomaly_count > 0
        accent = lambda hex_base: "#dc2626" if kpi_flash else hex_base
        fmt2 = lambda x: f"{x:.2f}"
        fmt3 = lambda x: f"{x:.3f}"

        c1.markdown(
            kpi_metric_box("kpi-blue", "Avg consumption", "L per reading", fmt2(avg_l), accent("#2563eb")),
            unsafe_allow_html=True,
        )
        c2.markdown(
            kpi_metric_box("kpi-green", "Peak usage", "L maximum", fmt2(max_l), accent("#0d9488")),
            unsafe_allow_html=True,
        )
        c3.markdown(
            kpi_metric_box("kpi-amber", "Minimum usage", "L minimum", fmt2(min_l), accent("#b45309")),
            unsafe_allow_html=True,
        )
        c4.markdown(
            kpi_metric_box("kpi-slate", "Avg flow rate", "L/min", fmt3(avg_f), accent("#64748b")),
            unsafe_allow_html=True,
        )

    # ── Alert banner ────────────────────────────────────────────────────────
    if anomaly_count > 0:
        alert_ph.markdown(
            f'<div class="alert-box"><strong>Anomaly alert.</strong> {anomaly_count} irregular reading'
            f'{"s" if anomaly_count != 1 else ""} flagged in the current selection.</div>',
            unsafe_allow_html=True)
    else:
        # Keep the UI quiet unless there's something to act on.
        alert_ph.empty()

    # ── Gauges ─────────────────────────────────────────────────────────────
    latest_flow  = filtered["flow_rate"].iloc[-1]    if len(filtered) else 0
    latest_use   = filtered["total_liters"].iloc[-1] if len(filtered) else 0
    anom_pct     = round(anomaly_count / max(len(filtered), 1) * 100, 1)

    flow_max  = max(filtered["flow_rate"].max()    * 1.3, 1)   if len(filtered) else 10
    usage_max = max(filtered["total_liters"].max() * 1.3, 1)  if len(filtered) else 10

    # Always-moving gauges (canvas) so this section constantly fluctuates.
    def _status_for(vv, lo, hi):
        if vv <= lo:
            return "NORMAL", "#10b981"
        if vv <= hi:
            return "ELEVATED", "#b45309"
        return "CRITICAL", "#dc2626"

    s0, c0 = _status_for(latest_flow, flow_max * 0.45, flow_max * 0.75)
    s1, c1 = _status_for(latest_use, avg_l, avg_l * 1.6)
    s2, c2 = _status_for(anom_pct, 5, 15)

    with gauge_ph[0]:
        live_gauge_component("g_flow", "Flow Rate", latest_flow, "L/min", flow_max, s0, c0)
    with gauge_ph[1]:
        live_gauge_component("g_use", "Latest Reading", latest_use, "L", usage_max, s1, c1)
    with gauge_ph[2]:
        live_gauge_component("g_anom", "Anomaly Index", anom_pct, "%", 100, s2, c2)

    # ── Status chips ────────────────────────────────────────────────────────
    std_l     = round(filtered["total_liters"].std(), 2) if len(filtered) > 1 else 0
    total_l   = round(filtered["total_liters"].sum(), 1) if len(filtered) else 0
    n_samples = len(filtered)

    with chip_ph.container():
        s1, s2, s3, s4, s5 = st.columns(5, gap="small")
        s1.markdown(chip_html("Avg flow",     round(avg_f, 2),     "L/min",   "#2563eb"), unsafe_allow_html=True)
        s2.markdown(chip_html("Std deviation",std_l,               "L",       "#475569"), unsafe_allow_html=True)
        s3.markdown(chip_html("Cumulative",   total_l,             "L",       "#0d9488"), unsafe_allow_html=True)
        s4.markdown(chip_html("Sample count", n_samples,           "rows",    "#64748b"), unsafe_allow_html=True)
        s5.markdown(chip_html("Anomalies",    anomaly_count,       "flagged",
                              "#dc2626" if anomaly_count > 0 else "#10b981"), unsafe_allow_html=True)

    # ── Trend chart ─────────────────────────────────────────────────────────
    trend_ph.plotly_chart(
        build_trend(filtered),
        use_container_width=True,
        config=PLOTLY_CONFIG,
        key="trend",
    )

    # ── Flow rate chart ─────────────────────────────────────────────────────
    flow_ph.plotly_chart(
        build_flow_rate(filtered),
        use_container_width=True,
        config=PLOTLY_CONFIG,
        key="flow",
    )

    hourly_ph.plotly_chart(build_hourly_bar(filtered), use_container_width=True,
                           config=PLOTLY_CONFIG, key="hourly")
    block_ph.plotly_chart(build_block_bar(df, sel_block), use_container_width=True,
                          config=PLOTLY_CONFIG, key="block")

    scatter_ph.plotly_chart(build_scatter(filtered), use_container_width=True,
                            config=PLOTLY_CONFIG, key="scatter")

    heatmap_ph.plotly_chart(build_floor_heatmap(df), use_container_width=True,
                            config=PLOTLY_CONFIG, key="heatmap")

    with twin_main_ph.container():
        digital_twin_component(df)

    anom_tl_ph.plotly_chart(
        build_anomaly_timeline(filtered),
        use_container_width=True,
        config=PLOTLY_CONFIG,
        key="anom_tl",
    )

    # ── Forecast ────────────────────────────────────────────────────────────
    y_vals = filtered["total_liters"].values
    X_idx  = np.arange(len(y_vals)).reshape(-1, 1)
    if len(y_vals) >= 3:
        lr    = LinearRegression().fit(X_idx, y_vals)
        preds = lr.predict(np.arange(len(y_vals), len(y_vals) + 5).reshape(-1, 1))
        pred_html_str = ""
        for i, p in enumerate(preds, 1):
            if i == 1:
                direction, col = "→", "#94a3b8"
            elif p > preds[i - 2]:
                direction, col = "↑", "#10b981"
            else:
                direction, col = "↓", "#dc2626"
            pred_html_str += pred_html(i, direction, p, col)

        hostel_usage = df.groupby("block")["total_liters"].mean()
        top_block    = hostel_usage.idxmax()
        top_val      = hostel_usage.max()

        forecast_ph.markdown(
            f'<div class="forecast-ticker">{pred_html_str}'
            f'<br><div class="warn-box">Highest average load: <strong>{top_block}</strong>'
            f' at {round(top_val, 2)} L per reading.</div></div>',
            unsafe_allow_html=True)
    else:
        forecast_ph.markdown(
            '<div class="warn-box">Collecting sufficient samples for projection. Please wait.</div>',
            unsafe_allow_html=True)

    # ── Hour deep-dive ──────────────────────────────────────────────────────
    snap_key = f"{sel_block}|{sel_floor}|{int(selected_hour)}"
    src_suffix = ""

    hour_floor = filtered[filtered["hour"] == selected_hour]
    if len(hour_floor) > 0:
        hour_data_src = hour_floor
    else:
        block_hour = df[(df["block"] == sel_block) & (df["hour"] == selected_hour)]
        if len(block_hour) > 0:
            hour_data_src = block_hour
            src_suffix = (
                f'<div class="info-box" style="margin-top:8px;"><strong>Note.</strong> No samples yet for '
                f'<strong>Floor {sel_floor}</strong> at <strong>{selected_hour}:00</strong> — '
                f"showing <strong>block {sel_block}</strong> for that hour.</div>"
            )
        else:
            any_block = df[df["hour"] == selected_hour]
            if len(any_block) > 0:
                hour_data_src = any_block
                src_suffix = (
                    f'<div class="info-box" style="margin-top:8px;"><strong>Note.</strong> No data for '
                    f"<strong>{sel_block}</strong> at <strong>{selected_hour}:00</strong> — "
                    f"showing <strong>all blocks</strong> for that hour.</div>"
                )
            else:
                blk = df[df["block"] == sel_block]
                hour_data_src = pd.DataFrame()
                if len(blk) > 0:
                    hc = blk.groupby("hour").size()
                    if len(hc) > 0:
                        nearest_h = int(
                            min(hc.index, key=lambda h: _hour_circular_dist(int(h), int(selected_hour)))
                        )
                        hour_data_src = blk[blk["hour"] == nearest_h]
                        src_suffix = (
                            f'<div class="info-box" style="margin-top:8px;"><strong>Note.</strong> Nothing at '
                            f"<strong>{selected_hour}:00</strong> for this block — "
                            f"showing nearest hour <strong>{nearest_h}:00</strong> "
                            f"(block <strong>{sel_block}</strong>).</div>"
                        )

    def _render_hour_panel(h_avg, h_max, h_tot, peak_h, peak_v, extra_html: str):
        hour_ph.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:10px;">
            <div class="chip" style="border-top:2px solid #2563eb;">
                <div class="chip-val" style="color:#2563eb;">{h_avg}</div>
                <div class="chip-label">Average L</div>
            </div>
            <div class="chip" style="border-top:2px solid #b45309;">
                <div class="chip-val" style="color:#b45309;">{h_max}</div>
                <div class="chip-label">Max L</div>
            </div>
            <div class="chip" style="border-top:2px solid #0d9488;">
                <div class="chip-val" style="color:#0d9488;">{h_tot}</div>
                <div class="chip-label">Total L</div>
            </div>
        </div>
        <div class="info-box">Peak hour <strong>{peak_h}:00</strong> — average {peak_v} L.</div>
        {extra_html}
        """, unsafe_allow_html=True)

    if len(hour_data_src) > 0:
        h_avg = round(float(hour_data_src["total_liters"].mean()), 1)
        h_max = round(float(hour_data_src["total_liters"].max()), 1)
        h_tot = round(float(hour_data_src["total_liters"].sum()), 1)
        hourly_all = filtered.groupby("hour")["total_liters"].mean().reindex(range(24), fill_value=0)
        if float(hourly_all.sum()) == 0.0:
            hourly_all = df[df["block"] == sel_block].groupby("hour")[
                "total_liters"].mean().reindex(range(24), fill_value=0)
        if float(hourly_all.sum()) == 0.0:
            hourly_all = df.groupby("hour")["total_liters"].mean().reindex(range(24), fill_value=0)
        peak_v = round(float(hourly_all.max()), 2)
        peak_h = int(hourly_all.idxmax()) if peak_v > 0 else int(selected_hour)
        extra = src_suffix
        _render_hour_panel(h_avg, h_max, h_tot, peak_h, peak_v, extra)
        st.session_state.hour_dive_snap[snap_key] = dict(
            h_avg=h_avg, h_max=h_max, h_tot=h_tot, peak_h=peak_h, peak_v=peak_v, extra=extra,
        )
    else:
        cached = st.session_state.hour_dive_snap.get(snap_key)
        if cached:
            hold = (
                '<div class="info-box" style="margin-top:8px;">'
                "Showing the last stable values for this selection while waiting for readings at "
                f"<strong>{selected_hour}:00</strong>.</div>"
            )
            _render_hour_panel(
                cached["h_avg"], cached["h_max"], cached["h_tot"],
                cached["peak_h"], cached["peak_v"],
                cached.get("extra", "") + hold,
            )
        else:
            hour_ph.markdown(
                '<div class="info-box">No data yet for this hour across the network. '
                "Try the current hour or wait a few seconds for new samples.</div>",
                unsafe_allow_html=True,
            )

    # ── Sidebar info ────────────────────────────────────────────────────────
    anom_cls = "sb-accent-alert" if anomaly_count > 0 else "sb-accent-green"
    info_placeholder.markdown(f"""
    <div class="sb-live-card">
      <div class="sb-live-head"><span class="live-dot"></span> Selection summary</div>
      <div class="sb-stat-grid">
        <div class="sb-stat">
          <div class="sb-stat-label">Block</div>
          <div class="sb-stat-val">{sel_block}</div>
        </div>
        <div class="sb-stat">
          <div class="sb-stat-label">Floor</div>
          <div class="sb-stat-val">{sel_floor}</div>
        </div>
        <div class="sb-stat">
          <div class="sb-stat-label">Records</div>
          <div class="sb-stat-val">{len(filtered)}</div>
        </div>
        <div class="sb-stat">
          <div class="sb-stat-label">Network total</div>
          <div class="sb-stat-val">{len(df)}</div>
        </div>
        <div class="sb-stat sb-stat-wide">
          <div class="sb-stat-label">Anomalies flagged</div>
          <div class="sb-stat-val {anom_cls}">{anomaly_count}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)
    # Use IframeMixin._html on the sidebar placeholder (not components.html on st._main).
    sidebar_clock_ph._html(_sidebar_live_clock_html(TOK), height=96, scrolling=False)

    # ── Drive live updates: sleep then trigger a full re-render ────────────
    time.sleep(REFRESH_INTERVAL_SEC)
    st.rerun()