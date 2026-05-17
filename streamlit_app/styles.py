"""Editorial CSS theme injection for the dashboard."""

import streamlit as st


EDITORIAL_CSS = """
<style>
/* === Editorial palette ===
   Forest green primary, cream background, deep navy headers, warm dividers
*/

:root {
    --editorial-cream: #FAF6EE;
    --editorial-cream-deep: #F4ECD6;
    --editorial-paper: #FCFAF4;
    --editorial-divider: #E5DCC9;
    --editorial-divider-strong: #C4B594;
    --editorial-text-primary: #1F3D2E;
    --editorial-text-secondary: #5C5A52;
    --editorial-accent-green: #2D5F3F;
    --editorial-accent-amber: #D89B3C;
    --editorial-accent-warning: #B8521F;
    --editorial-accent-danger: #993C1D;
    --editorial-accent-success: #1F6B3A;
}

/* === Base body styling === */

html, body, [class*="css"] {
    font-family: 'Georgia', 'Times New Roman', serif;
    background-color: var(--editorial-cream) !important;
    color: var(--editorial-text-primary) !important;
}

.stApp {
    background-color: var(--editorial-cream) !important;
}

.main .block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* === Editorial header === */

.editorial-header {
    background: var(--editorial-cream);
    padding: 0 0 1rem 0;
    margin-bottom: 1rem;
}

.masthead {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--editorial-text-primary);
    margin-bottom: 1.25rem;
}

.masthead-left {
    display: flex;
    gap: 24px;
    align-items: center;
}

.masthead-title {
    font-family: 'Georgia', serif;
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--editorial-text-primary);
    font-weight: 500;
}

.masthead-sub {
    font-size: 11px;
    color: var(--editorial-text-secondary);
}

.status-live {
    font-size: 10px;
    padding: 3px 10px;
    background: var(--editorial-accent-green);
    color: var(--editorial-cream);
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.editorial-title {
    font-family: 'Georgia', serif;
    font-size: 28px;
    font-weight: 400;
    line-height: 1.2;
    color: var(--editorial-text-primary);
    margin-bottom: 8px;
}

.editorial-title em {
    color: var(--editorial-accent-green);
    font-style: italic;
}

.editorial-subtitle {
    font-size: 13px;
    color: var(--editorial-text-secondary);
    max-width: 600px;
    line-height: 1.5;
}

/* === Section dividers === */

.section-label {
    font-family: 'Georgia', serif;
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--editorial-text-primary);
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--editorial-text-primary);
}

.section-label.green {
    color: var(--editorial-accent-green);
    border-bottom: 1px solid var(--editorial-accent-green);
}

/* === Editorial cards === */

.editorial-card {
    background: var(--editorial-paper);
    border: 1px solid var(--editorial-divider);
    padding: 1rem 1.25rem;
    border-radius: 2px;
    margin-bottom: 0.75rem;
}

.editorial-card-title {
    font-family: 'Georgia', serif;
    font-size: 13px;
    color: var(--editorial-text-primary);
    margin-bottom: 8px;
    letter-spacing: 0.03em;
}

/* === Editorial metric cards === */

.metric-card {
    border-left: 2px solid var(--editorial-accent-green);
    padding-left: 12px;
    padding-top: 4px;
    padding-bottom: 4px;
}

.metric-card-value {
    font-family: 'Georgia', serif;
    font-size: 30px;
    color: var(--editorial-text-primary);
    line-height: 1;
    margin-bottom: 4px;
}

.metric-card-value.green {
    color: var(--editorial-accent-green);
}

.metric-card-label {
    font-size: 11px;
    color: var(--editorial-text-secondary);
    letter-spacing: 0.05em;
}

/* === Methodology trail steps === */

.method-step {
    display: flex;
    gap: 16px;
    align-items: baseline;
    margin-bottom: 14px;
}

.method-step-number {
    font-family: 'Georgia', serif;
    font-size: 14px;
    color: var(--editorial-accent-green);
    min-width: 28px;
}

.method-step-title {
    font-size: 13px;
    color: var(--editorial-text-primary);
    font-weight: 500;
    margin-bottom: 2px;
}

.method-step-text {
    font-size: 12px;
    color: var(--editorial-text-secondary);
    line-height: 1.5;
}

/* === Streamlit tab overrides === */

.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid var(--editorial-text-primary);
    padding: 0;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    padding: 8px 16px;
    font-family: 'Georgia', serif;
    font-size: 12px;
    color: var(--editorial-text-secondary);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--editorial-text-primary);
    border-bottom: 2px solid var(--editorial-accent-green);
    background: transparent;
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
}

/* === Streamlit dataframe / table styling === */

[data-testid="stDataFrame"] {
    background: var(--editorial-paper);
    border: 1px solid var(--editorial-divider);
    border-radius: 2px;
}

/* === Streamlit metric widget overrides === */

[data-testid="stMetric"] {
    background: transparent;
    border-left: 2px solid var(--editorial-accent-green);
    padding: 6px 12px;
}

[data-testid="stMetricLabel"] {
    color: var(--editorial-text-secondary) !important;
    font-size: 11px !important;
    letter-spacing: 0.05em !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Georgia', serif !important;
    font-size: 28px !important;
    color: var(--editorial-text-primary) !important;
}

[data-testid="stMetricDelta"] {
    font-size: 11px !important;
}

/* === Streamlit info/warning/success boxes === */

[data-testid="stInfo"] {
    background: var(--editorial-paper);
    border-left: 2px solid var(--editorial-accent-green);
    color: var(--editorial-text-primary);
}

[data-testid="stSuccess"] {
    background: var(--editorial-paper);
    border-left: 2px solid var(--editorial-accent-success);
    color: var(--editorial-text-primary);
}

[data-testid="stWarning"] {
    background: var(--editorial-paper);
    border-left: 2px solid var(--editorial-accent-amber);
    color: var(--editorial-text-primary);
}

[data-testid="stError"] {
    background: var(--editorial-paper);
    border-left: 2px solid var(--editorial-accent-danger);
    color: var(--editorial-text-primary);
}

/* === Buttons === */

.stButton > button {
    background: var(--editorial-cream);
    color: var(--editorial-accent-green);
    border: 1px solid var(--editorial-accent-green);
    border-radius: 2px;
    font-family: 'Georgia', serif;
    font-size: 12px;
    padding: 6px 14px;
    letter-spacing: 0.05em;
}

.stButton > button:hover {
    background: var(--editorial-accent-green);
    color: var(--editorial-cream);
    border-color: var(--editorial-accent-green);
}

/* === Selectbox / multiselect / slider === */

.stSelectbox label, .stMultiSelect label, .stSlider label {
    font-family: 'Georgia', serif !important;
    font-size: 11px !important;
    color: var(--editorial-text-secondary) !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase;
}

/* === Sidebar (if used) === */

[data-testid="stSidebar"] {
    background: var(--editorial-cream-deep);
}

/* === Footer === */

.editorial-footer {
    border-top: 1px solid var(--editorial-divider);
    padding-top: 1rem;
    margin-top: 3rem;
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--editorial-text-secondary);
    letter-spacing: 0.05em;
}

/* === Status badges === */

.badge {
    display: inline-block;
    font-size: 10px;
    padding: 2px 8px;
    letter-spacing: 0.05em;
    border-radius: 2px;
    font-weight: 500;
}

.badge.green {
    background: var(--editorial-accent-green);
    color: var(--editorial-cream);
}

.badge.amber {
    background: #F8E4C0;
    color: #7A5316;
}

.badge.danger {
    background: #FCE9E0;
    color: var(--editorial-accent-danger);
}

.badge.gray {
    background: var(--editorial-divider);
    color: var(--editorial-text-secondary);
}

/* === Hide Streamlit default branding === */

#MainMenu, footer, header {
    visibility: hidden;
}

</style>
"""


def inject_editorial_css():
    """Inject the editorial CSS theme into the Streamlit app."""
    st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)
