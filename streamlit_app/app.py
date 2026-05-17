"""
Biodiversity-Aware EU Equity Portfolio — Streamlit Dashboard

Editorial aesthetic dashboard presenting the AI portfolio research pipeline
for the Prince Albert II of Monaco Foundation mandate.

Usage:
    streamlit run streamlit_app/app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Make the rest of the project importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from streamlit_app.styles import inject_editorial_css
from streamlit_app.data_loader import (
    load_portfolio_data,
    load_extractions,
    load_decision_log,
    load_master,
)
from streamlit_app.tabs import (
    render_overview_tab,
    render_holdings_tab,
    render_sustainability_tab,
    render_performance_tab,
    render_audit_trail_tab,
)


# === Page config ===

st.set_page_config(
    page_title="Biodiversity-Aware EU Portfolio",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject editorial CSS theme
inject_editorial_css()


# === Header ===

st.markdown(
    """
    <div class="editorial-header">
        <div class="masthead">
            <div class="masthead-left">
                <span class="masthead-title">ESADE · Sustainable Finance 2026</span>
                <span class="masthead-sub">Issue 04 · 17 May</span>
            </div>
            <div class="masthead-right">
                <span class="status-live">Live</span>
            </div>
        </div>
        <div class="editorial-title">
            A biodiversity-aware<br/>
            <em>European equity portfolio</em>
        </div>
        <div class="editorial-subtitle">
            An AI research mandate for the Prince Albert II of Monaco Foundation,
            balancing 24 holdings across 7 sectors against an EBA-referenced
            carbon intensity ceiling.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# === Load all data once (cached) ===

@st.cache_data
def load_all_data():
    """Load every data source the dashboard needs. Cached for the session."""
    return {
        "portfolio": load_portfolio_data(ROOT),
        "extractions": load_extractions(ROOT),
        "decision_log": load_decision_log(ROOT),
        "master": load_master(ROOT),
    }


try:
    data = load_all_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()


# === Tabs ===

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Holdings",
    "Sustainability",
    "Performance",
    "Audit trail",
])

with tab1:
    render_overview_tab(data)

with tab2:
    render_holdings_tab(data)

with tab3:
    render_sustainability_tab(data)

with tab4:
    render_performance_tab(data)

with tab5:
    render_audit_trail_tab(data)


# === Footer ===

st.markdown(
    """
    <div class="editorial-footer">
        <span>ESADE MSc Business Analytics · Sustainable Finance Group Project</span>
        <span>Pipeline: 10 agents · 130 tests passing · Build 0.9.4</span>
    </div>
    """,
    unsafe_allow_html=True,
)
