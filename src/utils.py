"""
Agentic HRMS — UI Utilities, CSS Theme & Helper Components
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─── Color Palette ────────────────────────────────────────────────────────────
COLORS = {
    "bg_primary": "#0E1117",
    "bg_card": "#1B2838",
    "bg_card_alt": "#1E2D3D",
    "bg_hover": "#243447",
    "accent_cyan": "#00D4FF",
    "accent_teal": "#00BFA5",
    "positive": "#00C853",
    "warning": "#FFB300",
    "danger": "#FF1744",
    "text_primary": "#FFFFFF",
    "text_secondary": "#8899AA",
    "text_muted": "#5A6B7C",
    "border": "#2A3A4A",
    "border_light": "#3A4A5A",
    "chart_colors": ["#00D4FF", "#00BFA5", "#7C4DFF", "#FF6D00", "#FFD600",
                     "#FF1744", "#00E676", "#448AFF", "#E040FB", "#FF9100"],
}

# ─── Plotly Chart Template ────────────────────────────────────────────────────
CHART_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "color": "#8899AA", "size": 12},
        "title": {"font": {"color": "#FFFFFF", "size": 16}},
        "xaxis": {"gridcolor": "#1E2D3D", "zerolinecolor": "#2A3A4A", "title_font": {"color": "#8899AA"}},
        "yaxis": {"gridcolor": "#1E2D3D", "zerolinecolor": "#2A3A4A", "title_font": {"color": "#8899AA"}},
        "legend": {"font": {"color": "#8899AA"}},
        "margin": {"l": 40, "r": 20, "t": 50, "b": 40},
    }
}


def apply_chart_style(fig, height=400):
    """Apply consistent dark theme styling to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#8899AA", size=12),
        title_font=dict(color="#FFFFFF", size=16),
        xaxis=dict(gridcolor="#1E2D3D", zerolinecolor="#2A3A4A"),
        yaxis=dict(gridcolor="#1E2D3D", zerolinecolor="#2A3A4A"),
        legend=dict(font=dict(color="#8899AA"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=height,
    )
    return fig


def inject_custom_css():
    """Inject the global custom CSS theme for the entire application."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Reset ───────────────────────────────── */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #0E1117;
    }

    /* ── Sidebar ────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B0F19 0%, #131B2E 100%);
        border-right: 1px solid #1E2D3D;
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        color: #00D4FF !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        color: #8899AA !important;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        color: #CFD8DC !important;
        padding: 6px 12px;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        background: rgba(0, 212, 255, 0.08);
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"],
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {
        background: rgba(0, 212, 255, 0.12);
        color: #00D4FF !important;
    }

    /* ── Main Content ───────────────────────────────── */
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }

    /* ── Headers ────────────────────────────────────── */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
    }
    h1 { font-weight: 700 !important; }
    h2 { font-weight: 600 !important; color: #E0E8F0 !important; }
    h3 { font-weight: 500 !important; color: #B0BEC5 !important; }

    /* ── KPI Card ───────────────────────────────────── */
    .kpi-card {
        background: linear-gradient(135deg, #1B2838 0%, #1E2D3D 100%);
        border: 1px solid #2A3A4A;
        border-radius: 12px;
        padding: 20px 18px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent, #00D4FF), transparent);
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        border-color: #3A4A5A;
    }
    .kpi-icon { font-size: 1.6rem; margin-bottom: 6px; }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #6B7D8E;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
        font-weight: 500;
    }
    .kpi-delta {
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .kpi-delta.positive { color: #00C853; }
    .kpi-delta.negative { color: #FF1744; }
    .kpi-delta.neutral  { color: #FFB300; }

    /* ── Info Card ──────────────────────────────────── */
    .info-card {
        background: linear-gradient(135deg, #1B2838 0%, #1E2D3D 100%);
        border: 1px solid #2A3A4A;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .info-card h4 {
        color: #00D4FF !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }

    /* ── Status Pill ────────────────────────────────── */
    .status-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 50px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .status-pill.active   { background: rgba(0,200,83,0.15); color: #00C853; border: 1px solid rgba(0,200,83,0.3); }
    .status-pill.warning  { background: rgba(255,179,0,0.15); color: #FFB300; border: 1px solid rgba(255,179,0,0.3); }
    .status-pill.danger   { background: rgba(255,23,68,0.15); color: #FF1744; border: 1px solid rgba(255,23,68,0.3); }
    .status-pill.info     { background: rgba(0,212,255,0.15); color: #00D4FF; border: 1px solid rgba(0,212,255,0.3); }

    /* ── Skill Badge ───────────────────────────────── */
    .skill-match  { color: #00C853; margin: 3px 0; font-size: 0.92rem; }
    .skill-gap    { color: #FF1744; margin: 3px 0; font-size: 0.92rem; }

    /* ── Profile Card ──────────────────────────────── */
    .profile-header {
        background: linear-gradient(135deg, #0B1929 0%, #132238 50%, #1B2838 100%);
        border: 1px solid #2A3A4A;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .profile-header::after {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(0,212,255,0.04) 0%, transparent 60%);
    }
    .profile-avatar {
        width: 72px; height: 72px;
        border-radius: 50%;
        background: linear-gradient(135deg, #00D4FF, #00BFA5);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 12px;
        font-size: 1.8rem;
        font-weight: 700;
        color: #0E1117;
    }

    /* ── Metric Overrides ──────────────────────────── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1B2838, #1E2D3D);
        border: 1px solid #2A3A4A;
        border-radius: 10px;
        padding: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
    }
    div[data-testid="stMetric"] label {
        color: #6B7D8E !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* ── Tabs ──────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid #2A3A4A;
    }
    .stTabs [data-baseweb="tab"] {
        color: #6B7D8E;
        font-weight: 500;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #00D4FF !important;
        border-bottom: 2px solid #00D4FF;
        background: rgba(0,212,255,0.06);
    }

    /* ── Expander ──────────────────────────────────── */
    .streamlit-expanderHeader {
        background: #1B2838 !important;
        border: 1px solid #2A3A4A !important;
        border-radius: 8px !important;
        color: #8899AA !important;
    }

    /* ── Selectbox / Input ─────────────────────────── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div > input {
        background-color: #1B2838 !important;
        color: #FFFFFF !important;
        border-color: #2A3A4A !important;
    }

    /* ── Footer ────────────────────────────────────── */
    .app-footer {
        text-align: center;
        padding: 20px 0;
        border-top: 1px solid #1E2D3D;
        margin-top: 40px;
        color: #5A6B7C;
        font-size: 0.78rem;
    }

    /* ── Chat ──────────────────────────────────────── */
    .stChatMessage {
        background: #1B2838 !important;
        border: 1px solid #2A3A4A !important;
        border-radius: 12px !important;
    }

    /* ── Scrollbar ─────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0E1117; }
    ::-webkit-scrollbar-thumb { background: #2A3A4A; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3A4A5A; }

    /* ── Progress bar styling ─────────────────────── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00D4FF, #00BFA5) !important;
    }

    </style>
    """, unsafe_allow_html=True)


def render_kpi_card(icon, value, label, delta=None, delta_type="positive", accent_color="#00D4FF"):
    """Render a styled KPI metric card."""
    delta_html = ""
    if delta is not None:
        arrow = "↑" if delta_type == "positive" else ("↓" if delta_type == "negative" else "→")
        delta_html = f'<div class="kpi-delta {delta_type}">{arrow} {delta}</div>'

    st.markdown(f"""
    <div class="kpi-card" style="--accent: {accent_color}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_status_pill(text, status="info"):
    """Render a status pill badge. status: active|warning|danger|info"""
    return f'<span class="status-pill {status}">{text}</span>'


def render_section_header(title, subtitle=None, icon=None):
    """Render a styled section header."""
    icon_html = f"{icon} " if icon else ""
    sub_html = f'<p style="color:#6B7D8E; font-size:0.88rem; margin-top:4px;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="margin-bottom: 16px;">
        <h2 style="margin-bottom:2px;">{icon_html}{title}</h2>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    """Render the application footer."""
    st.markdown("""
    <div class="app-footer">
        <strong>Agentic HRMS</strong> — AI/ML Capstone Project &nbsp;|&nbsp; Workforce Intelligence Prototype<br>
        <span style="font-size:0.7rem; color:#3A4A5A;">Built with Streamlit, scikit-learn, Plotly & Sentence Transformers</span>
    </div>
    """, unsafe_allow_html=True)


def get_risk_color(risk_level):
    """Return appropriate color for a risk level."""
    if risk_level == "HIGH":
        return COLORS["danger"]
    elif risk_level == "MEDIUM":
        return COLORS["warning"]
    return COLORS["positive"]


def create_gauge_chart(value, title="", max_val=100, color=None):
    """Create a gauge/donut chart for readiness/scores."""
    if color is None:
        if value >= 75:
            color = COLORS["positive"]
        elif value >= 50:
            color = COLORS["warning"]
        else:
            color = COLORS["danger"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 36, "color": "#FFFFFF", "family": "Inter"}},
        title={"text": title, "font": {"size": 14, "color": "#8899AA"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#3A4A5A", "tickfont": {"color": "#5A6B7C"}},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "#1E2D3D",
            "borderwidth": 0,
            "steps": [
                {"range": [0, max_val * 0.33], "color": "rgba(255,23,68,0.08)"},
                {"range": [max_val * 0.33, max_val * 0.66], "color": "rgba(255,179,0,0.08)"},
                {"range": [max_val * 0.66, max_val], "color": "rgba(0,200,83,0.08)"},
            ],
        }
    ))
    apply_chart_style(fig, height=250)
    fig.update_layout(margin=dict(l=30, r=30, t=40, b=10))
    return fig


def render_sidebar_status(model_ready=True, data_ready=True, rag_ready=True, live_mode=False):
    """Render sidebar status indicators."""
    statuses = [
        ("● Data Connected", "active" if data_ready else "danger"),
        ("● Model Ready", "active" if model_ready else "warning"),
        ("● RAG Ready", "active" if rag_ready else "warning"),
        ("● Live Mode", "active" if live_mode else "info"),
    ]
    pills = " &nbsp;".join([render_status_pill(t, s) for t, s in statuses])
    st.markdown(f'<div style="margin: 8px 0 16px 0;">{pills}</div>', unsafe_allow_html=True)
