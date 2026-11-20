"""
app.py — Bhada: Indian logistics cost prediction and fleet analytics.
Author: Pawan Singh

Backend calls (src.prediction, src.analytics, src.explainability) are unchanged.
Requires Streamlit >= 1.39 (widget `key=` becomes a CSS class `st-key-<key>`,
which this file uses to style widgets reliably in both Day and Night mode).

NOTE: The data is a synthetic educational dataset (15,000 trips). Costs are
formula-generated, so metrics are not real-world freight performance.
"""

import os
import base64
import html as _html
import json
from string import Template

os.environ.setdefault("KERAS_BACKEND", "torch")

import altair as alt
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.prediction import (
    load_prediction_model,
    predict_single,
    predict_batch,
    validate_prediction_input,
)
from src.analytics import (
    compute_kpis,
    cost_by_truck_type,
    top_routes,
    expense_breakdown,
    load_utilization_distribution,
    monthly_cost_trend,
)
from src.explainability import (
    compute_shap_values,
    explain_single_prediction,
    plot_global_importance,
    plot_single_explanation,
    _load_best_tree_model,
)

# ---------------------------------------------------------------------------
# Project identity
# ---------------------------------------------------------------------------
APP_NAME = "Bhada"
APP_NAME_HINDI = "भाड़ा"          # "freight charge"
AUTHOR = "Pawan Singh"

st.set_page_config(
    page_title=f"{APP_NAME} · Freight Cost Prediction",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Overview"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


def _toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode


def _go_to(page_key):
    st.session_state.nav_page = page_key


IS_DARK = st.session_state.dark_mode

# ---------------------------------------------------------------------------
# Design tokens: every colour in the app comes from this block, so Day and
# Night can never drift apart.
# ---------------------------------------------------------------------------
TEAL = "#0F3B40"
MARIGOLD = "#F2A900"
VERMILION = "#D8392B"
CREAM = "#FFF6DC"
ASPHALT_INK = "#14262A"

if IS_DARK:
    T = dict(
        BG_BASE="#0B1619",
        BG_OVERLAY="linear-gradient(180deg, rgba(9,18,21,0.90) 0%, rgba(12,24,28,0.93) 100%)",
        CARD_BG="rgba(18,34,40,0.94)", INPUT_BG="#14262C",
        INK="#F5F2EA", MUTED="#B4C8C6", LINE="#2E4E56", ACCENT_TEXT="#F2A900",
        MPL_BG="#122228",
        STONE_BODY="#1B3239", STONE_TEXT="#F5F2EA",
        TH_BG="rgba(255,255,255,0.06)", ROW_HOVER="rgba(242,169,0,0.13)",
        # toggle pill
        TOGGLE_BG="#F2A900", TOGGLE_FG="#14262A", TOGGLE_BORDER="#FFF6DC",
        # hover toolbars (tables, charts, images)
        TB_BG="#1F3E46", TB_FG="#F5F2EA", TB_BORDER="#F2A900",
        TB_HOVER_BG="#F2A900", TB_HOVER_FG="#14262A",
        # tooltips + menus
        TIP_BG="#0F2227", TIP_FG="#F5F2EA", TIP_BORDER="#F2A900",
        # milestone nav stones
        NAV_BODY="#16303A", NAV_TEXT="#F5F2EA", NAV_CAP="#2A8F86", NAV_BASE="#0A1517", NAV_BORDER="#3B6770",
        NAV_ON_BODY="#223F49", NAV_ON_TEXT="#FFF6DC",
        UPLOAD_BORDER="#F2A900",
    )
    TRUCK_COLORS = ["#2EC4B6", "#F2A900", "#FF595E", "#52B788"]
    EXPENSE_COLORS = ["#2EC4B6", "#F2A900", "#FF595E", "#52B788", "#4EA8DE", "#ADB5BD"]
    VOLUME_BAR = "#3A595C"
    LINE_COLOR = "#FF595E"
else:
    T = dict(
        BG_BASE="#E7EFEA",
        BG_OVERLAY="linear-gradient(180deg, rgba(237,242,239,0.87) 0%, rgba(231,239,234,0.84) 100%)",
        CARD_BG="rgba(255,253,247,0.95)", INPUT_BG="#FFFFFF",
        INK="#14262A", MUTED="#485C5E", LINE="#C2D2CC", ACCENT_TEXT="#0F3B40",
        MPL_BG="#FFFDF7",
        STONE_BODY="#FFFFFF", STONE_TEXT="#14262A",
        TH_BG="rgba(15,59,64,0.07)", ROW_HOVER="rgba(242,169,0,0.16)",
        TOGGLE_BG="#0F3B40", TOGGLE_FG="#FFF6DC", TOGGLE_BORDER="#F2A900",
        TB_BG="#0F3B40", TB_FG="#FFF6DC", TB_BORDER="#F2A900",
        TB_HOVER_BG="#F2A900", TB_HOVER_FG="#14262A",
        TIP_BG="#FFFDF7", TIP_FG="#14262A", TIP_BORDER="#0F3B40",
        NAV_BODY="#F7F5EE", NAV_TEXT="#14262A", NAV_CAP="#1D5C55", NAV_BASE="#14262A", NAV_BORDER="#14262A",
        NAV_ON_BODY="#FFFDF7", NAV_ON_TEXT="#14262A",
        UPLOAD_BORDER="#0F3B40",
    )
    TRUCK_COLORS = ["#0F3B40", "#E09600", "#D8392B", "#2A8555"]
    EXPENSE_COLORS = ["#0F3B40", "#E09600", "#D8392B", "#2A8555", "#2B6CB0", "#718096"]
    VOLUME_BAR = "#C9D8D3"
    LINE_COLOR = "#D8392B"

INK, MUTED, LINE = T["INK"], T["MUTED"], T["LINE"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_TRUCK_IMG = os.path.join(BASE_DIR, "assets", "truck_bg.jpg")
if os.path.isfile(LOCAL_TRUCK_IMG):
    with open(LOCAL_TRUCK_IMG, "rb") as _f:
        BG_IMAGE_URL = "data:image/jpeg;base64," + base64.b64encode(_f.read()).decode()
else:
    BG_IMAGE_URL = "https://images.unsplash.com/photo-1601584115197-04ecc0da31d7?auto=format&fit=crop&w=1920&q=75"

# ---------------------------------------------------------------------------
# CSS (string.Template: ${TOKEN} placeholders, so no brace escaping needed)
# Widgets are targeted by their `key` class (st-key-...) instead of by
# position, which is what broke the toggle and nav buttons before.
# ---------------------------------------------------------------------------
CSS = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rozha+One&family=Hind:wght@400;500;600;700&display=swap');

html, body, .stApp { font-family: 'Hind', system-ui, sans-serif; font-size: 16px; }
.stApp {
    background-color: ${BG_BASE} !important;
    background-image: ${BG_OVERLAY}, url("${BG_IMAGE_URL}");
    background-size: cover, cover;
    background-position: center, center;
    background-attachment: fixed;
    color: ${INK};
}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
[data-testid="stExpandSidebarButton"], [data-testid="stToolbar"], [data-testid="stDeployButton"],
[data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
/* Streamlit's fixed header is 60px tall with a huge z-index and sat on top of the
   toggle, swallowing its clicks. Remove it and make sure it can never take clicks. */
[data-testid="stHeader"], header[data-testid="stHeader"] {
    display: none !important; height: 0 !important; min-height: 0 !important; pointer-events: none !important; }
.block-container { padding-top: 1rem; max-width: 1240px; }

/* ---- Base text colour (low specificity so custom blocks can override) ---- */
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h4,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h5,
[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
[data-testid="stMain"] [data-testid="stSpinner"] *,
[data-testid="stMain"] [data-testid="stText"] { color: ${INK}; }
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h4 { font-size: 1.08rem; font-weight: 600; margin: 4px 0 6px 0; }
[data-testid="stMain"] [data-testid="stWidgetLabel"] p { font-weight: 600; font-size: .95rem; }
[data-testid="stMain"] [data-testid="stCaptionContainer"] p { color: ${ACCENT_TEXT}; font-weight: 600; }
[data-testid="stMain"] code { color: ${INK}; background: ${TH_BG}; }

/* ---- Page enter animation (key per page => remounts => replays) ---- */
@keyframes pageIn { from { opacity: 0; transform: translate3d(0, 12px, 0); } to { opacity: 1; transform: none; } }
[class*="st-key-page_"] { animation: pageIn .38s cubic-bezier(.22,1,.36,1) backwards; }
[data-testid="stMain"] .st-key-theme_toggle { position: relative; z-index: 1000; }

/* ---- Day / Night toggle ---- */
[data-testid="stMain"] .st-key-theme_toggle button {
    background: ${TOGGLE_BG}; border: 2px solid ${TOGGLE_BORDER}; border-radius: 999px;
    min-height: 40px; padding: 4px 16px; box-shadow: 0 4px 12px rgba(0,0,0,.28);
    transition: transform .2s ease, box-shadow .2s ease;
}
[data-testid="stMain"] .st-key-theme_toggle button:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(242,169,0,.45); border-color: ${TOGGLE_BORDER}; }
[data-testid="stMain"] .st-key-theme_toggle button p { color: ${TOGGLE_FG}; -webkit-text-fill-color: ${TOGGLE_FG}; font-weight: 700; font-size: .9rem; letter-spacing: .4px; margin: 0; }

/* ---- Masthead + road (one unit, so nothing floats loose) ---- */
.masthead { background: linear-gradient(135deg, #0F3B40 0%, #082225 100%); border-radius: 10px; overflow: hidden;
    border: 2px solid #14262A; box-shadow: 0 6px 18px rgba(0,0,0,.25); margin-top: 6px; }
.mh-top { display: flex; justify-content: space-between; align-items: flex-end; padding: 20px 30px 16px 30px; gap: 24px; }
.mh-badge { display: inline-block; background: #F2A900; color: #14262A; font-weight: 700; font-size: .76rem; letter-spacing: 1.2px;
    padding: 2px 10px; border-radius: 3px; margin-bottom: 6px; text-transform: uppercase; }
.mh-title { font-family: 'Rozha One', serif; font-size: 2.9rem; line-height: 1; color: #FFF6DC; }
.mh-sub { color: #DCE8E5; margin-top: 6px; max-width: 56ch; font-size: .98rem; line-height: 1.4; }
.mh-right { text-align: right; }
.mh-hindi { font-family: 'Rozha One', serif; font-size: 3.9rem; line-height: .9; color: #F2A900; }
.mh-horn { font-family: 'Rozha One', serif; font-size: .9rem; color: #FFF6DC; letter-spacing: 2px; margin-top: 5px;
    border: 1px solid rgba(242,169,0,.65); padding: 2px 10px; border-radius: 3px; display: inline-block; background: rgba(216,57,43,.28); }
.zig { height: 16px; background-color: #0F3B40;
    background-image: linear-gradient(135deg, #F2A900 25%, transparent 25%), linear-gradient(225deg, #F2A900 25%, transparent 25%),
                      linear-gradient(315deg, #D8392B 25%, transparent 25%), linear-gradient(45deg, #D8392B 25%, transparent 25%);
    background-size: 16px 16px; background-position: -8px 0, -8px 0, 0 0, 0 0; border-top: 2px solid #14262A; }

/* Road: asphalt strip, edge lines, and a lane line that scrolls seamlessly.
   The dash pattern repeats every 60px and the element moves exactly 60px per
   loop, so the restart is invisible. Only transform is animated (GPU). */
.road { position: relative; height: 40px; background: linear-gradient(180deg, #2A3134 0%, #171C1E 100%); overflow: hidden; border-top: 2px solid #14262A; }
.road::before, .road::after { content: ""; position: absolute; left: 0; right: 0; height: 2px; background: rgba(244,241,232,.5); }
.road::before { top: 6px; }
.road::after { bottom: 6px; }
.lane { position: absolute; top: 50%; left: -60px; width: calc(100% + 60px); height: 4px; margin-top: -2px;
    background: repeating-linear-gradient(90deg, #F4F1E8 0, #F4F1E8 28px, transparent 28px, transparent 60px);
    animation: laneMove 1.3s linear infinite; will-change: transform; }
@keyframes laneMove { from { transform: translate3d(0,0,0); } to { transform: translate3d(60px,0,0); } }
@media (prefers-reduced-motion: reduce) { .lane, [class*="st-key-page_"], [class*="st-key-nav_on_"] button { animation: none !important; } }
@media (max-width: 800px) { .mh-hindi { display: none; } .mh-title { font-size: 2.3rem; } }

/* ---- Milestone navigation stones ---- */
@keyframes stonePulse { 0%,100% { box-shadow: 0 5px 0 rgba(0,0,0,.35), 0 0 0 rgba(242,169,0,0); } 50% { box-shadow: 0 5px 0 rgba(0,0,0,.35), 0 0 16px rgba(242,169,0,.6); } }
[data-testid="stMain"] [class*="st-key-nav_"] button {
    width: 100%; min-height: 88px; padding: 6px 4px;
    background: ${NAV_BODY}; border: 2.5px solid ${NAV_BORDER};
    border-top: 14px solid ${NAV_CAP}; border-bottom: 6px solid ${NAV_BASE};
    border-radius: 30px 30px 8px 8px; box-shadow: 0 5px 0 rgba(0,0,0,.35);
    transition: transform .22s cubic-bezier(.22,1,.36,1), box-shadow .22s ease;
}
[data-testid="stMain"] [class*="st-key-nav_"] button:hover { transform: translate3d(0,-5px,0); border-color: #F2A900; box-shadow: 0 9px 16px rgba(0,0,0,.4); }
[data-testid="stMain"] [class*="st-key-nav_"] button p { color: ${NAV_TEXT}; -webkit-text-fill-color: ${NAV_TEXT}; margin: 0; line-height: 1.25; font-weight: 700; font-size: .88rem; }
[data-testid="stMain"] [class*="st-key-nav_"] button p:not(:first-child) { font-size: .78rem; font-weight: 500; opacity: .85; }
[data-testid="stMain"] [class*="st-key-nav_on_"] button {
    background: ${NAV_ON_BODY}; border-color: #F2A900; border-top-color: #F2A900; border-bottom-color: #D8392B;
    transform: translate3d(0,-4px,0); animation: stonePulse 2.4s ease-in-out infinite;
}
[data-testid="stMain"] [class*="st-key-nav_on_"] button p { color: ${NAV_ON_TEXT}; -webkit-text-fill-color: ${NAV_ON_TEXT}; }

.disclosure { color: ${MUTED}; background: ${CARD_BG}; backdrop-filter: blur(4px); padding: 8px 14px; border-radius: 6px;
    border: 1px solid ${LINE}; font-size: .85rem; margin: 12px 0 4px 0; }

/* ---- Page header plate ---- */
.page-head { background: ${CARD_BG}; backdrop-filter: blur(6px); border: 1px solid ${LINE}; border-radius: 8px; padding: 14px 22px 12px 22px; margin: 16px 0 16px 0; }
.page-title { font-family: 'Rozha One', serif; font-size: 2rem; line-height: 1.15; color: ${INK}; }
.page-title::after { content: ""; display: block; width: 64px; height: 4px; background: #D8392B; margin-top: 8px; }
.page-desc { color: ${MUTED}; margin-top: 10px; max-width: 70ch; font-weight: 500; }

/* ---- Overview lead ---- */
.lead { font-size: 1.45rem; line-height: 1.7; max-width: 34em; margin: 4px 0 22px 0; color: ${INK}; background: ${CARD_BG};
    backdrop-filter: blur(6px); padding: 18px 24px; border-radius: 8px; border: 1px solid ${LINE}; border-left: 5px solid #F2A900;
    box-shadow: 0 3px 12px rgba(0,0,0,.08); }
.lead b { font-family: 'Rozha One', serif; font-weight: 400; color: ${ACCENT_TEXT}; font-size: 1.85rem; font-variant-numeric: tabular-nums;
    background: linear-gradient(transparent 62%, rgba(242,169,0,.38) 62%); padding: 0 3px; }

/* ---- Prediction result ---- */
.result { display: grid; grid-template-columns: 230px 1fr; gap: 36px; margin: 22px 0 10px 0; align-items: start;
    background: ${CARD_BG}; backdrop-filter: blur(6px); padding: 24px; border-radius: 10px; border: 1.5px solid ${LINE}; }
.stone { width: 205px; text-align: center; box-shadow: 0 4px 0 #14262A; border-radius: 100px 100px 6px 6px; overflow: hidden; border: 2px solid #14262A; background: ${STONE_BODY}; }
.stone .cap { background: #F2A900; padding: 28px 10px 12px 10px; border-bottom: 2px solid #14262A; }
.stone .cap .to { font-size: .8rem; color: #14262A; opacity: .85; text-transform: uppercase; font-weight: 700; }
.stone .cap .city { font-family: 'Rozha One', serif; font-size: 1.55rem; color: #14262A; line-height: 1.1; }
.stone .mid { padding: 16px 8px 14px 8px; background: ${STONE_BODY}; }
.stone .mid .km { font-family: 'Rozha One', serif; font-size: 2.6rem; color: ${STONE_TEXT}; line-height: 1; font-variant-numeric: tabular-nums; }
.stone .mid .u { color: ${MUTED}; font-size: .9rem; font-weight: 600; }
.stone .foot { background: #14262A; color: #FFFFFF; padding: 9px 8px; font-size: .92rem; }
.verdict .small { color: ${MUTED}; font-size: .95rem; }
.verdict .amt { font-family: 'Rozha One', serif; font-size: 3.6rem; line-height: 1.1; color: ${INK}; font-variant-numeric: tabular-nums; margin: 2px 0 4px 0; }
.verdict .perkm { color: #D8392B; font-weight: 700; margin-bottom: 18px; font-size: 1.05rem; }
.stack { display: flex; height: 18px; border-radius: 3px; overflow: hidden; margin: 8px 0 12px 0; border: 1px solid ${LINE}; }
.legend { display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px 28px; font-size: .94rem; max-width: 560px; color: ${INK}; }
.legend div { display: flex; justify-content: space-between; border-bottom: 1px solid ${LINE}; padding: 4px 0; }
.legend i { display: inline-block; width: 10px; height: 10px; margin-right: 8px; border-radius: 2px; }
.legend span.n { font-variant-numeric: tabular-nums; font-weight: 600; }
@media (max-width: 800px) { .result { grid-template-columns: 1fr; } .stone { margin: 0 auto; } }

/* ---- Themed HTML tables (replace st.dataframe, whose canvas ignores CSS) ---- */
.tbl-wrap { overflow-x: auto; }
table.tbl { width: 100%; border-collapse: collapse; font-size: .93rem; color: ${INK}; }
table.tbl th { text-align: left; color: ${MUTED}; font-weight: 600; background: ${TH_BG}; padding: 9px 12px; border-bottom: 2px solid ${LINE}; white-space: nowrap; }
table.tbl td { padding: 8px 12px; border-bottom: 1px solid ${LINE}; color: ${INK}; }
table.tbl tr:hover td { background: ${ROW_HOVER}; }
table.tbl .num { text-align: right; font-variant-numeric: tabular-nums; }

/* ---- Hover toolbars (charts, images): one rule set, both modes ---- */
[data-testid="stElementToolbar"] { background: ${TB_BG} !important; border: 1.5px solid ${TB_BORDER} !important; border-radius: 6px !important;
    padding: 2px 4px !important; opacity: 1 !important; box-shadow: 0 4px 12px rgba(0,0,0,.35) !important; }
[data-testid="stElementToolbar"] button, [data-testid="stElementToolbarButton"], [data-testid="StyledFullScreenButton"] {
    background: transparent !important; color: ${TB_FG} !important; border-radius: 4px !important; }
[data-testid="stElementToolbar"] button *, [data-testid="stElementToolbarButton"] *, [data-testid="StyledFullScreenButton"] * { color: ${TB_FG} !important; }
[data-testid="stElementToolbar"] svg, [data-testid="StyledFullScreenButton"] svg { fill: currentColor !important; stroke: none !important; }
[data-testid="stElementToolbar"] button:hover, [data-testid="stElementToolbarButton"]:hover, [data-testid="StyledFullScreenButton"]:hover { background: ${TB_HOVER_BG} !important; }
[data-testid="stElementToolbar"] button:hover *, [data-testid="stElementToolbarButton"]:hover *, [data-testid="StyledFullScreenButton"]:hover * { color: ${TB_HOVER_FG} !important; }
[data-testid="StyledFullScreenButton"] { background: ${TB_BG} !important; border: 1.5px solid ${TB_BORDER} !important; opacity: 1 !important; }

/* Tooltips (Streamlit + Vega) and menus */
[data-baseweb="tooltip"] [role="tooltip"], [data-testid="stTooltipContent"], #vg-tooltip-element {
    background: ${TIP_BG} !important; color: ${TIP_FG} !important; border: 1.5px solid ${TIP_BORDER} !important; border-radius: 6px !important;
    box-shadow: 0 6px 16px rgba(0,0,0,.35) !important; font-family: 'Hind', sans-serif !important; }
[data-baseweb="tooltip"] *, [data-testid="stTooltipContent"] * { color: ${TIP_FG} !important; }
#vg-tooltip-element table td.key { color: ${MUTED} !important; font-weight: 600; }
#vg-tooltip-element table td.value { color: ${TIP_FG} !important; font-weight: 700; }
.vega-embed summary { background: ${TB_BG} !important; border: 1.5px solid ${TB_BORDER} !important; border-radius: 5px !important; opacity: 1 !important; }
.vega-embed summary svg { fill: ${TB_FG} !important; }
.vega-embed .vega-actions { background: ${TIP_BG} !important; border: 1.5px solid ${TIP_BORDER} !important; }
.vega-embed .vega-actions a { color: ${TIP_FG} !important; font-weight: 600; }
.vega-embed .vega-actions a:hover { background: #F2A900 !important; color: #14262A !important; }

/* ---- Panels, metrics, alerts ---- */
div[data-testid="stVerticalBlockBorderWrapper"] { border: 1.5px solid ${LINE} !important; border-radius: 10px !important; background: ${CARD_BG} !important;
    backdrop-filter: blur(8px); box-shadow: 0 4px 14px rgba(0,0,0,.08); }
div[data-testid="stMetric"] { background: ${CARD_BG} !important; border: 1px solid ${LINE}; border-left: 5px solid #F2A900; border-radius: 6px; padding: 12px 16px; }
div[data-testid="stMetricValue"] * { font-family: 'Rozha One', serif; font-weight: 400; color: ${INK} !important; }
div[data-testid="stMetricLabel"] * { color: ${MUTED} !important; font-weight: 600; }
[data-testid="stAlert"], [data-testid="stAlertContainer"] { background: ${CARD_BG} !important; border: 1.5px solid ${LINE} !important; border-left: 5px solid #F2A900 !important; border-radius: 6px !important; }
[data-testid="stAlert"] *, [data-testid="stAlertContainer"] * { color: ${INK} !important; }
.fieldset-title { font-weight: 700; font-size: 1.08rem; color: ${INK}; margin: 0 0 10px 0; padding-left: 10px; border-left: 4px solid #D8392B; }
[data-testid="stImage"] img { border-radius: 6px; border: 1px solid ${LINE}; }

/* ---- Inputs ---- */
[data-testid="stMain"] input { color: ${INK} !important; background: ${INPUT_BG} !important; -webkit-text-fill-color: ${INK} !important; font-weight: 600; }
[data-testid="stMain"] div[data-baseweb="select"] > div, [data-testid="stMain"] div[data-baseweb="input"] > div { background: ${INPUT_BG} !important; border: 1.5px solid ${LINE} !important; }
[data-testid="stMain"] div[data-baseweb="select"] * { color: ${INK} !important; font-weight: 600; }
[data-testid="stMain"] div[data-baseweb="select"] svg { fill: ${INK} !important; }
[data-testid="stMain"] [data-testid="stNumberInputStepDown"], [data-testid="stMain"] [data-testid="stNumberInputStepUp"] { background: ${INPUT_BG} !important; color: ${INK} !important; border-left: 1px solid ${LINE} !important; }
[data-testid="stMain"] [data-testid="stNumberInputStepDown"]:hover, [data-testid="stMain"] [data-testid="stNumberInputStepUp"]:hover { background: #F2A900 !important; color: #14262A !important; }
[data-testid="stMain"] [data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stMain"] [data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stMain"] [data-testid="stSlider"] [data-testid="stThumbValue"] { color: #D8392B !important; font-weight: 700; }
/* Select dropdown list renders in a portal outside the page, so it needs its own rules */
[data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] ul, [data-baseweb="popover"] [data-baseweb="menu"] { background: ${INPUT_BG} !important; }
[data-baseweb="popover"] li, [data-baseweb="popover"] li * { color: ${INK} !important; background: transparent !important; font-weight: 500; }
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] { background: #F2A900 !important; }
[data-baseweb="popover"] li:hover *, [data-baseweb="popover"] li[aria-selected="true"] * { color: #14262A !important; }

/* ---- File uploader ---- */
[data-testid="stFileUploaderDropzone"] { background: ${CARD_BG} !important; backdrop-filter: blur(6px); border: 2px dashed ${UPLOAD_BORDER} !important; border-radius: 8px !important; padding: 20px !important; }
[data-testid="stFileUploaderDropzone"] * { color: ${INK} !important; }
[data-testid="stFileUploaderDropzone"] button { background: #F2A900 !important; border: 1.5px solid #14262A !important; font-weight: 700; }
[data-testid="stFileUploaderDropzone"] button * { color: #14262A !important; }
[data-testid="stFileUploaderDropzone"] button:hover { background: #D8392B !important; }
[data-testid="stFileUploaderDropzone"] button:hover * { color: #FFFFFF !important; }

/* ---- Action + download buttons ---- */
[data-testid="stMain"] [class*="st-key-run_"] button, [data-testid="stMain"] [data-testid="stDownloadButton"] button {
    background: #D8392B; border: 1.5px solid #14262A; border-radius: 6px; padding: .55rem 1.5rem; }
[data-testid="stMain"] [class*="st-key-run_"] button:hover, [data-testid="stMain"] [data-testid="stDownloadButton"] button:hover { background: #B42D21; }
[data-testid="stMain"] [class*="st-key-run_"] button p, [data-testid="stMain"] [data-testid="stDownloadButton"] button p {
    color: #FFFFFF; -webkit-text-fill-color: #FFFFFF; font-weight: 700; margin: 0; }
[data-testid="stMain"] [class*="st-key-run_"] button:disabled { background: ${LINE}; border-color: ${LINE}; }
[data-testid="stMain"] [class*="st-key-run_"] button:disabled p { color: ${MUTED}; -webkit-text-fill-color: ${MUTED}; }
button:focus-visible, input:focus-visible { outline: 3px solid #F2A900 !important; outline-offset: 2px; }

/* ---- Footer ---- */
.footer { background: ${CARD_BG}; backdrop-filter: blur(4px); color: ${MUTED}; font-size: .88rem; margin: 44px 0 10px 0; padding: 12px 18px;
    border-radius: 8px; border: 1px solid ${LINE}; border-top: 3px solid #F2A900; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px; }
.footer b { color: ${ACCENT_TEXT}; }
</style>
""")
st.markdown(CSS.substitute(BG_IMAGE_URL=BG_IMAGE_URL, **T), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "indian_truck_logistics_synthetic_15000.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed_logistics_data.csv")

CITY_STATE_MAP = {
    "Ahmedabad": "Gujarat", "Bengaluru": "Karnataka", "Bhopal": "Madhya Pradesh",
    "Chennai": "Tamil Nadu", "Delhi": "Delhi", "Hyderabad": "Telangana",
    "Indore": "Madhya Pradesh", "Jaipur": "Rajasthan", "Kolkata": "West Bengal",
    "Lucknow": "Uttar Pradesh", "Mumbai": "Maharashtra", "Nagpur": "Maharashtra",
    "Nashik": "Maharashtra", "Pune": "Maharashtra", "Surat": "Gujarat",
    "Vadodara": "Gujarat",
}
TRUCK_TYPES = ["Heavy", "Medium", "LCV", "Trailer"]
TRUCK_CAPACITIES = {"Heavy": 16.0, "Medium": 9.0, "LCV": 3.5, "Trailer": 25.0}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@st.cache_data
def load_dataset_cached():
    path = PROCESSED_DATA_PATH if os.path.exists(PROCESSED_DATA_PATH) else DATA_PATH
    return pd.read_csv(path) if os.path.isfile(path) else None


@st.cache_data
def load_metrics_cached():
    p = os.path.join(MODEL_DIR, "evaluation_metrics.json")
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_resource
def get_prediction_pipeline():
    try:
        return load_prediction_model()
    except Exception:
        return None, None, None, None


dataset = load_dataset_cached()
metrics = load_metrics_cached()
model, preprocessor, schema, model_type = get_prediction_pipeline()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def inr(x, d=0):
    return f"₹{x:,.{d}f}"


def page_header(title, desc=""):
    d = f'<div class="page-desc">{desc}</div>' if desc else ""
    st.markdown(f'<div class="page-head"><div class="page-title">{title}</div>{d}</div>', unsafe_allow_html=True)


def html_table(df, labels=None, fmt=None):
    """Theme-aware table. st.dataframe draws on a canvas and ignores CSS, so it
    could not follow the Day/Night toggle."""
    labels, fmt = labels or {}, fmt or {}
    numeric = {c: pd.api.types.is_numeric_dtype(df[c]) for c in df.columns}
    head = "".join(
        f'<th class="{"num" if numeric[c] else ""}">{_html.escape(str(labels.get(c, c)))}</th>' for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            if v is None or (isinstance(v, float) and pd.isna(v)):
                s = "—"
            elif c in fmt:
                s = fmt[c](v)
            elif isinstance(v, float):
                s = f"{v:,.2f}"
            else:
                s = str(v)
            tds.append(f'<td class="{"num" if numeric[c] else ""}">{_html.escape(s)}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    st.markdown(
        f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>',
        unsafe_allow_html=True)


def show(chart, height=300):
    chart = (
        chart.properties(height=height)
        .configure(background="transparent", font="Hind")
        .configure_view(stroke=None)
        .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=LINE, domainColor=LINE,
                        tickColor=LINE, labelFontSize=12, titleFontSize=12, titleFontWeight=500)
        .configure_legend(labelColor=INK, titleColor=MUTED, orient="top", symbolType="square")
    )
    st.altair_chart(chart, use_container_width=True, theme=None)


def themed_figure(fig):
    """Restyle matplotlib figures coming from src.explainability."""
    fig.patch.set_facecolor(T["MPL_BG"])
    for ax in fig.axes:
        ax.set_facecolor(T["MPL_BG"])
        ax.tick_params(colors=INK)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(INK)
        for txt in ax.texts:
            txt.set_color(INK)
        for s in ax.spines.values():
            s.set_color(LINE)
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_facecolor(T["MPL_BG"])
            leg.get_frame().set_edgecolor(LINE)
            for t in leg.get_texts():
                t.set_color(INK)
    return fig


# ---------------------------------------------------------------------------
# Day / Night toggle
# ---------------------------------------------------------------------------
_, toggle_col = st.columns([6.2, 1.3], vertical_alignment="center")
with toggle_col:
    st.button(
        "☀️ Day mode" if IS_DARK else "🌙 Night mode",
        key="theme_toggle",
        use_container_width=True,
        on_click=_toggle_theme,
    )

# ---------------------------------------------------------------------------
# Masthead with built-in animated road
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="masthead">
  <div class="mh-top">
    <div>
      <div class="mh-badge">शुभ लाभ · All India Permit</div>
      <div class="mh-title">{APP_NAME}</div>
      <div class="mh-sub">Predict what an Indian truck trip will cost before it leaves, and see where the fleet's money goes.</div>
    </div>
    <div class="mh-right">
      <div class="mh-hindi">{APP_NAME_HINDI}</div>
      <div class="mh-horn">HORN OK PLEASE</div>
    </div>
  </div>
  <div class="zig"></div>
  <div class="road"><div class="lane"></div></div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Milestone navigation
# ---------------------------------------------------------------------------
MILESTONES = [
    ("🛣️", "Overview", "सफ़र", "Overview"),
    ("🚛", "Predict trip", "भाड़ा", "Predict a trip"),
    ("📋", "Batch bilty", "बिल्टी", "Predict a batch"),
    ("🗺️", "Routes", "मार्ग", "Routes"),
    ("⛽", "Expenses", "खर्च", "Expenses and loads"),
    ("🧠", "Model & SHAP", "बुद्धि", "Model and SHAP"),
]

st.write("")
nav_cols = st.columns(len(MILESTONES), gap="small")
for idx, (icon, title, hindi, page_key) in enumerate(MILESTONES):
    active = st.session_state.nav_page == page_key
    with nav_cols[idx]:
        # The key prefix (nav_on_ / nav_off_) becomes a CSS class, which is how
        # the active stone is styled without relying on element position.
        st.button(
            f"{icon} {title}\n\n{hindi}",
            key=f"nav_{'on' if active else 'off'}_{idx}",
            use_container_width=True,
            on_click=_go_to,
            args=(page_key,),
        )

st.markdown(
    '<div class="disclosure">The data is synthetic: about 15,000 simulated trips across 16 Indian cities, '
    'with costs generated by a formula. Read every figure as a demonstration, not real freight performance.</div>',
    unsafe_allow_html=True)

navigation = st.session_state.nav_page

# ===========================================================================
# Pages (each inside a keyed container so it fades in on navigation)
# ===========================================================================
with st.container(key=f"page_{navigation.replace(' ', '_')}"):

    # ------------------------------------------------------------------ Overview
    if navigation == "Overview":
        page_header("Overview")

        if dataset is None:
            st.warning("Dataset not found. Place the CSV in the `data/` folder.")
        else:
            k = compute_kpis(dataset)
            st.markdown(f"""
            <div class="lead">
              Across <b>{k["total_trips"]:,}</b> trips, the average cost was <b>{inr(k["avg_cost_inr"])}</b>,
              or <b>{inr(k["avg_cost_per_km"], 2)}</b> per km. Trucks ran <b>{k.get("avg_load_utilization_pct", 0):.1f}%</b> full,
              and <b>{k.get("profitable_trip_pct", 0):.1f}%</b> of trips made a profit.
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2, gap="large")
            with c1:
                with st.container(border=True):
                    st.markdown("#### Average cost by truck type")
                    bt = cost_by_truck_type(dataset)
                    df = pd.DataFrame({"truck": bt.index.astype(str), "avg_cost": bt["avg_cost"].values})
                    base = alt.Chart(df).encode(
                        x=alt.X("truck:N", title=None, sort="-y", axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("avg_cost:Q", title="Average cost (₹)"))
                    bars = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                        color=alt.Color("truck:N", legend=None, scale=alt.Scale(range=TRUCK_COLORS)),
                        tooltip=[alt.Tooltip("truck:N", title="Truck"),
                                 alt.Tooltip("avg_cost:Q", title="Avg cost ₹", format=",.0f")])
                    labels = base.mark_text(dy=-8, color=INK, fontWeight=600).encode(
                        text=alt.Text("avg_cost:Q", format=",.0f"))
                    show(bars + labels)

            with c2:
                with st.container(border=True):
                    st.markdown("#### Cost and trip volume by month")
                    m = monthly_cost_trend(dataset)
                    mdf = pd.DataFrame({"month": MONTHS[: len(m)], "avg_cost": list(m["avg_cost"]),
                                        "trips": list(m["trip_count"])})
                    x = alt.X("month:N", sort=MONTHS, title=None, axis=alt.Axis(labelAngle=0))
                    vol = alt.Chart(mdf).mark_bar(color=VOLUME_BAR, size=20).encode(
                        x=x, y=alt.Y("trips:Q", title="Trips", axis=alt.Axis(orient="right")),
                        tooltip=["month", "trips"])
                    line = alt.Chart(mdf).mark_line(
                        color=LINE_COLOR, strokeWidth=3,
                        point=alt.OverlayMarkDef(color=LINE_COLOR, size=70)).encode(
                        x=x, y=alt.Y("avg_cost:Q", title="Average cost (₹)", scale=alt.Scale(zero=False)),
                        tooltip=["month", alt.Tooltip("avg_cost:Q", format=",.0f", title="Avg cost ₹")])
                    show(alt.layer(vol, line).resolve_scale(y="independent"))

            with st.container(border=True):
                st.markdown("#### Recent trips")
                cols = ["trip_id", "origin_city", "destination_city", "distance_km", "truck_type",
                        "load_weight_tons", "actual_trip_cost_inr", "freight_revenue_inr", "trip_profit_inr"]
                html_table(
                    dataset[cols].head(10),
                    labels={"trip_id": "Trip", "origin_city": "From", "destination_city": "To",
                            "distance_km": "Distance (km)", "truck_type": "Truck", "load_weight_tons": "Load (t)",
                            "actual_trip_cost_inr": "Cost", "freight_revenue_inr": "Revenue",
                            "trip_profit_inr": "Profit"},
                    fmt={"distance_km": lambda v: f"{v:,.0f}", "load_weight_tons": lambda v: f"{v:.1f}",
                         "actual_trip_cost_inr": inr, "freight_revenue_inr": inr, "trip_profit_inr": inr})

    # ------------------------------------------------------------ Predict a trip
    elif navigation == "Predict a trip":
        page_header("Predict a trip",
                    "Enter what you know at dispatch. Anything only known after the trip is left out on purpose.")

        cities = sorted(CITY_STATE_MAP.keys())
        col_a, col_b, col_c = st.columns(3, gap="medium")

        with col_a:
            with st.container(border=True):
                st.markdown('<div class="fieldset-title">Route and timing</div>', unsafe_allow_html=True)
                origin_city = st.selectbox("From", cities, index=cities.index("Mumbai"))
                origin_state = CITY_STATE_MAP[origin_city]
                destination_city = st.selectbox("To", cities, index=cities.index("Delhi"))
                destination_state = CITY_STATE_MAP[destination_city]
                st.caption(f"Corridor: {origin_state} → {destination_state}")
                distance_km = st.number_input("Distance (km)", 50, 3500, 1400, step=50)
                month = st.selectbox("Dispatch month", list(range(1, 13)),
                                     format_func=lambda m: MONTHS[m - 1], index=5)

        with col_b:
            with st.container(border=True):
                st.markdown('<div class="fieldset-title">Truck and load</div>', unsafe_allow_html=True)
                truck_type = st.selectbox("Truck type", TRUCK_TYPES, index=0)
                def_cap = TRUCK_CAPACITIES[truck_type]
                truck_capacity_tons = st.number_input("Capacity (tons)", 1.0, 40.0, def_cap, step=0.5)
                load_weight_tons = st.number_input("Load (tons)", 0.5, 40.0, min(def_cap * 0.75, 12.0), step=0.5)
                fuel_efficiency = st.number_input("Fuel efficiency (km/L)", 1.0, 15.0, 4.2, step=0.1)
                diesel_price = st.number_input("Diesel price (₹/L)", 60.0, 150.0, 94.5, step=0.5)

        with col_c:
            with st.container(border=True):
                st.markdown('<div class="fieldset-title">Conditions and expenses</div>', unsafe_allow_html=True)
                route_factor = st.slider("Route difficulty", 0.8, 1.5, 1.05, 0.01)
                traffic_factor = st.slider("Traffic congestion", 0.8, 1.5, 1.10, 0.01)
                toll_cost = st.number_input("Tolls (₹)", 0.0, 15000.0, 1200.0, step=100.0)
                driver_allowance = st.number_input("Driver allowance (₹)", 0.0, 15000.0, 1800.0, step=100.0)
                maintenance_cost = st.number_input("Maintenance (₹)", 0.0, 20000.0, 2500.0, step=100.0)
                loading_unloading_cost = st.number_input("Loading and unloading (₹)", 0.0, 20000.0, 2000.0, step=100.0)
                other_cost = st.number_input("Other (₹)", 0.0, 10000.0, 800.0, step=50.0)

        input_dict = {
            "origin_city": origin_city, "origin_state": origin_state,
            "destination_city": destination_city, "destination_state": destination_state,
            "distance_km": distance_km, "truck_type": truck_type,
            "truck_capacity_tons": truck_capacity_tons, "load_weight_tons": load_weight_tons,
            "fuel_efficiency_km_per_litre": fuel_efficiency, "diesel_price_inr_per_litre": diesel_price,
            "toll_cost_inr": toll_cost, "driver_allowance_inr": driver_allowance,
            "maintenance_cost_inr": maintenance_cost, "loading_unloading_cost_inr": loading_unloading_cost,
            "other_cost_inr": other_cost, "route_factor": route_factor,
            "traffic_factor": traffic_factor, "month": month,
        }

        errors = validate_prediction_input(input_dict)
        for err in errors:
            st.error(err)

        st.write("")
        go = st.button("Predict trip cost", type="primary", disabled=bool(errors or model is None),
                       key="run_single_pred")

        if go:
            with st.spinner("Estimating cost"):
                pred = predict_single(model, preprocessor, schema, input_dict, model_type)
                fuel_l = distance_km / fuel_efficiency
                fuel_cost = fuel_l * diesel_price
                util = load_weight_tons / truck_capacity_tons * 100

                parts = [
                    ("Fuel", fuel_cost), ("Tolls", toll_cost), ("Driver allowance", driver_allowance),
                    ("Maintenance", maintenance_cost), ("Loading and unloading", loading_unloading_cost),
                    ("Other", other_cost),
                ]
                direct = sum(v for _, v in parts) or 1.0
                stack = "".join(
                    f'<div style="width:{v / direct * 100:.2f}%;background:{EXPENSE_COLORS[i]}"></div>'
                    for i, (_, v) in enumerate(parts))
                legend = "".join(
                    f'<div><span><i style="background:{EXPENSE_COLORS[i]}"></i>{n}</span>'
                    f'<span class="n">{inr(v)}</span></div>' for i, (n, v) in enumerate(parts))

                st.markdown(f"""
                <div class="result">
                  <div class="stone">
                    <div class="cap"><div class="to">to</div><div class="city">{destination_city}</div></div>
                    <div class="mid"><div class="km">{distance_km:,}</div><div class="u">km</div></div>
                    <div class="foot">from {origin_city}</div>
                  </div>
                  <div class="verdict">
                    <div class="small">Predicted operating cost for a {truck_type} truck, {util:.0f}% full</div>
                    <div class="amt">{inr(pred, 2)}</div>
                    <div class="perkm">{inr(pred / distance_km, 2)} per km</div>
                    <div class="small">Expenses you entered add up to {inr(direct)}</div>
                    <div class="stack">{stack}</div>
                    <div class="legend">{legend}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                with st.container(border=True):
                    st.markdown("#### What drove this estimate")
                    try:
                        tree_m, t_name = _load_best_tree_model()
                        if tree_m is not None:
                            row = pd.DataFrame([input_dict])[schema["all_feature_columns"]]
                            proc = preprocessor.transform(row)
                            expl = explain_single_prediction(proc, model=tree_m, model_name=t_name)
                            fig = themed_figure(plot_single_explanation(expl, top_n=8))
                            st.pyplot(fig)
                            plt.close(fig)
                            st.caption(f"Explained with {t_name}. {expl['disclaimer']}")
                    except Exception as e:
                        st.info(f"The SHAP breakdown is not available: {e}")

    # ----------------------------------------------------------- Predict a batch
    elif navigation == "Predict a batch":
        page_header("Predict a batch",
                    "Upload your dispatch bilty (CSV) of planned trips and get a cost estimate for each one.")
        up = st.file_uploader("Upload trip dispatch CSV", type=["csv"])

        if up is not None:
            try:
                batch_df = pd.read_csv(up)
                st.write(f"**{len(batch_df):,}** trips loaded from `{up.name}`.")
                run_batch = st.button("Predict costs", type="primary", disabled=model is None, key="run_batch_pred")
                if run_batch:
                    with st.spinner("Estimating costs"):
                        res = predict_batch(model, preprocessor, schema, batch_df, model_type)
                    st.success(f"Estimated {len(res):,} trips.")
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Total predicted cost", inr(res["predicted_cost_inr"].sum()))
                    s2.metric("Average per trip", inr(res["predicted_cost_inr"].mean()))
                    s3.metric("Most expensive trip", inr(res["predicted_cost_inr"].max()))
                    with st.container(border=True):
                        st.markdown("#### First 20 predictions")
                        html_table(res.head(20), fmt={"predicted_cost_inr": lambda v: inr(v, 2)})
                    st.download_button("Download predictions", data=res.to_csv(index=False).encode("utf-8"),
                                       file_name="fleet_cost_predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Could not process this file: {e}. Check that the columns match the template.")
        else:
            st.info("No file uploaded yet. Use the box above, or download the sample template below first.")
            if dataset is not None and schema is not None:
                tmpl = dataset[schema["all_feature_columns"]].head(5)
                st.download_button("Download CSV template", data=tmpl.to_csv(index=False).encode("utf-8"),
                                   file_name="trip_dispatch_template.csv", mime="text/csv")

    # -------------------------------------------------------------------- Routes
    elif navigation == "Routes":
        page_header("Routes", "Which corridors cost the most, and how cost grows with distance.")

        if dataset is None:
            st.warning("Dataset not found.")
        else:
            left, right = st.columns([1, 1.6], gap="large")
            with left:
                with st.container(border=True):
                    st.markdown("#### Ten most expensive routes")
                    rt = top_routes(dataset, n=10)
                    html_table(
                        rt[["route", "avg_cost", "avg_distance", "trip_count"]],
                        labels={"route": "Route", "avg_cost": "Avg cost", "avg_distance": "Distance (km)",
                                "trip_count": "Trips"},
                        fmt={"avg_cost": inr, "avg_distance": lambda v: f"{v:,.0f}",
                             "trip_count": lambda v: f"{v:,.0f}"})
            with right:
                with st.container(border=True):
                    st.markdown("#### Distance against cost")
                    pts = dataset.sample(n=min(1200, len(dataset)), random_state=42)
                    sc = alt.Chart(pts).mark_circle(size=42, opacity=0.72).encode(
                        x=alt.X("distance_km:Q", title="Distance (km)"),
                        y=alt.Y("actual_trip_cost_inr:Q", title="Trip cost (₹)"),
                        color=alt.Color("truck_type:N", title=None,
                                        scale=alt.Scale(domain=TRUCK_TYPES, range=TRUCK_COLORS)),
                        tooltip=["origin_city", "destination_city", "truck_type",
                                 alt.Tooltip("distance_km:Q", format=",.0f"),
                                 alt.Tooltip("actual_trip_cost_inr:Q", format=",.0f", title="Cost ₹")])
                    show(sc, height=380)

    # ------------------------------------------------------- Expenses and loads
    elif navigation == "Expenses and loads":
        page_header("Expenses and loads", "Where direct expenses go, and how full the trucks run.")

        if dataset is None:
            st.warning("Dataset not found.")
        else:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                with st.container(border=True):
                    st.markdown("#### Share of direct expenses")
                    ex = expense_breakdown(dataset)
                    b = alt.Chart(ex).encode(y=alt.Y("Category:N", sort="-x", title=None),
                                             x=alt.X("Percentage:Q", title="Share of direct expenses (%)"))
                    bars = b.mark_bar(color=TRUCK_COLORS[0], cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
                    txt = b.mark_text(align="left", dx=5, color=INK, fontWeight=600).encode(
                        text=alt.Text("Percentage:Q", format=".1f"))
                    show(bars + txt)

            with c2:
                with st.container(border=True):
                    st.markdown("#### How full trucks are")
                    ut = load_utilization_distribution(dataset)
                    if ut.empty:
                        st.info("No load utilisation data to show.")
                    else:
                        order = list(ut["Band"])
                        b = alt.Chart(ut).encode(
                            x=alt.X("Band:N", sort=order, title="Load as share of capacity",
                                    axis=alt.Axis(labelAngle=0)),
                            y=alt.Y("Percentage:Q", title="Trips (%)"))
                        bars = b.mark_bar(color=MARIGOLD, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                        txt = b.mark_text(dy=-8, color=INK, fontWeight=600).encode(
                            text=alt.Text("Percentage:Q", format=".1f"))
                        show(bars + txt)

    # ------------------------------------------------------------ Model and SHAP
    elif navigation == "Model and SHAP":
        page_header("Model and SHAP", "How each candidate model scored, and which inputs matter most.")

        if not metrics:
            st.warning("No evaluation metrics found. Run `python -m src.train_model` first.")
        else:
            rows = []
            for name in ["dummy_baseline", "linear_regression", "random_forest", "hist_gradient_boosting", "ann"]:
                m = metrics.get(name, {})
                if m:
                    rows.append({"Model": name.replace("_", " ").title(), "Test MAE": m.get("mae", 0),
                                 "Test RMSE": m.get("rmse", 0), "Test R²": m.get("r2", 0),
                                 "CV MAE": m.get("cv_mae_mean")})
            with st.container(border=True):
                st.markdown("#### Candidate models")
                html_table(pd.DataFrame(rows),
                           fmt={"Test MAE": inr, "Test RMSE": inr, "Test R²": lambda v: f"{v:.4f}", "CV MAE": inr})
                best = metrics.get("best_model", {})
                st.markdown(
                    f"**Selected:** {best.get('best_model_name', 'hist_gradient_boosting').replace('_', ' ').title()}, "
                    f"chosen on {best.get('selection_criterion', 'lowest MAE')}.")

            p1, p2 = st.columns(2, gap="large")
            comp = os.path.join(MODEL_DIR, "model_comparison.png")
            avp = os.path.join(MODEL_DIR, "actual_vs_predicted.png")
            if os.path.exists(comp):
                with p1.container(border=True):
                    st.image(comp, caption="Model comparison", use_container_width=True)
            if os.path.exists(avp):
                with p2.container(border=True):
                    st.image(avp, caption="Actual against predicted cost", use_container_width=True)

            with st.container(border=True):
                st.markdown("#### Which inputs move the cost most")
                try:
                    tree_m, t_name = _load_best_tree_model()
                    if tree_m is not None and dataset is not None:
                        sample = dataset[schema["all_feature_columns"]].head(200)
                        proc = preprocessor.transform(sample)
                        shap_res = compute_shap_values(proc, model=tree_m, model_name=t_name)
                        fig = themed_figure(plot_global_importance(shap_res, top_n=10))
                        st.pyplot(fig)
                        plt.close(fig)
                except Exception as e:
                    st.info(f"Global SHAP plot unavailable: {e}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="footer">
  <span>🚛 <b>{APP_NAME} ({APP_NAME_HINDI})</b> · Indian Freight Cost &amp; Fleet Analytics</span>
  <span>Built by <b>{AUTHOR}</b> · Mera Bharat Mahan</span>
</div>
""", unsafe_allow_html=True)
