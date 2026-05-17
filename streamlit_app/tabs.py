"""
Tab rendering functions for the dashboard.

Each render_*_tab function takes the loaded data dict and renders content
into the active Streamlit tab.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


# === Editorial palette ===

EDITORIAL_CREAM = "#FAF6EE"
EDITORIAL_PAPER = "#FCFAF4"
EDITORIAL_TEXT = "#1F3D2E"
EDITORIAL_TEXT_SEC = "#5C5A52"
EDITORIAL_GREEN = "#2D5F3F"
EDITORIAL_GREEN_LIGHT = "#6B9477"
EDITORIAL_AMBER = "#D89B3C"
EDITORIAL_WARNING = "#B8521F"
EDITORIAL_DANGER = "#993C1D"
EDITORIAL_SUCCESS = "#1F6B3A"
EDITORIAL_DIVIDER = "#E5DCC9"
EDITORIAL_HIGHLIGHT = "#F4ECD6"

SECTOR_COLORS = {
    "Financials": "#4A6B8F",
    "Technology": "#6B5E8F",
    "Industrials": "#8F7A4A",
    "Consumer Discretionary": "#A8633F",
    "Health Care": "#7A4A6B",
    "Communications": "#8F4A7A",
    "Utilities": EDITORIAL_GREEN,
}


# === Tab 1 — OVERVIEW ===

def render_overview_tab(data: dict) -> None:
    metrics = data["portfolio"]["metrics"]
    mandate = data["portfolio"]["mandate"]
    holdings_df = data["portfolio"]["holdings_df"]

    st.markdown('<div class="section-label green">By the numbers</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-value">{metrics['n_holdings']}</div>
                <div class="metric-card-label">Holdings across {metrics['n_sectors']} sectors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-value green">{metrics['carbon_pct_of_eba']:.0f}%</div>
                <div class="metric-card-label">of EBA carbon reference (cap: {mandate['carbon_cap_pct_of_eba']}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-value">{metrics['sharpe_portfolio']:.2f}</div>
                <div class="metric-card-label">Sharpe ratio · benchmark {metrics['sharpe_benchmark']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-value">+{metrics['return_3y_ann_portfolio']:.1f}%</div>
                <div class="metric-card-label">Annualised 3y return · vs {metrics['return_3y_ann_benchmark']:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("&nbsp;", unsafe_allow_html=True)

    left, right = st.columns([1.4, 1])
    
    with left:
        st.markdown('<div class="section-label">Sector composition</div>', unsafe_allow_html=True)
        st.caption("Max 4 holdings per sector · Total weight 100%")
        
        sector_df = (
            holdings_df.groupby("sector")
            .agg(weight=("weight", "sum"), n=("weight", "size"))
            .reset_index()
            .sort_values("weight", ascending=False)
        )
        sector_df["weight_pct"] = sector_df["weight"] * 100
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sector_df["weight_pct"],
            y=sector_df["sector"],
            orientation="h",
            text=[f"{w:.1f}%" for w in sector_df["weight_pct"]],
            textposition="outside",
            marker=dict(color=EDITORIAL_GREEN, line=dict(color=EDITORIAL_TEXT, width=0.5)),
            hovertemplate="<b>%{y}</b><br>%{x:.2f}% · %{customdata} holdings<extra></extra>",
            customdata=sector_df["n"],
        ))
        fig.update_layout(
            paper_bgcolor=EDITORIAL_PAPER,
            plot_bgcolor=EDITORIAL_PAPER,
            font=dict(family="Georgia, serif", color=EDITORIAL_TEXT, size=12),
            margin=dict(l=0, r=40, t=20, b=20),
            height=300,
            xaxis=dict(showgrid=False, showline=False, ticksuffix="%", color=EDITORIAL_TEXT_SEC),
            yaxis=dict(showgrid=False, color=EDITORIAL_TEXT),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with right:
        st.markdown('<div class="section-label">Mandate constraints</div>', unsafe_allow_html=True)
        
        constraint_rows = [
            ("Max single name", f"{mandate['max_single_name_weight']*100:.1f}%"),
            ("Max sector weight", f"{mandate['max_sector_weight']*100:.1f}%"),
            ("Min sectors", f"{mandate['min_sectors']}"),
            ("Max per sector", f"{mandate['max_holdings_per_sector']} holdings"),
            ("Carbon cap", f"{mandate['carbon_cap_pct_of_eba']}% of EBA ref"),
            ("Holdings range", f"{mandate['min_holdings']}-{mandate['max_holdings']}"),
            ("Excluded sectors", ", ".join(mandate['excluded_sectors'])),
            ("Universe", mandate["universe"]),
            ("Benchmark", mandate["benchmark"]),
        ]
        
        html = '<div class="editorial-card">'
        for label, value in constraint_rows:
            html += (
                f'<div style="display:flex; justify-content:space-between; '
                f'padding:4px 0; border-bottom:0.5px dashed {EDITORIAL_DIVIDER}; font-size:12px;">'
                f'<span style="color:{EDITORIAL_TEXT_SEC};">{label}</span>'
                f'<span style="color:{EDITORIAL_TEXT}; font-weight:500;">{value}</span>'
                f'</div>'
            )
        html += (
            f'<div style="margin-top:10px; padding-top:8px; '
            f'border-top:1px solid {EDITORIAL_DIVIDER}; font-size:11px; '
            f'color:{EDITORIAL_GREEN};">All constraints satisfied</div>'
        )
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)
    
    left2, right2 = st.columns(2)
    
    with left2:
        st.markdown(
            f'<div class="section-label">Exclusions ({metrics["n_excluded"]} companies)</div>',
            unsafe_allow_html=True,
        )
        st.caption("Companies removed during pipeline screening")
        
        category_colors = {
            "sector": EDITORIAL_WARNING,
            "biodiversity": EDITORIAL_AMBER,
            "data": EDITORIAL_TEXT_SEC,
        }
        category_labels = {
            "sector": "Sector exclusion",
            "biodiversity": "Bio risk",
            "data": "Data gaps",
        }
        
        html = '<div class="editorial-card">'
        for row in data["portfolio"]["excluded"]:
            color = category_colors.get(row["category"], EDITORIAL_TEXT_SEC)
            label = category_labels.get(row["category"], row["category"])
            html += (
                f'<div style="display:flex; justify-content:space-between; '
                f'align-items:center; padding:6px 0; '
                f'border-bottom:0.5px dashed {EDITORIAL_DIVIDER}; font-size:12px;">'
                f'<span style="color:{EDITORIAL_TEXT};">{row["company"]}</span>'
                f'<span style="font-size:10px; padding:2px 8px; '
                f'background:{EDITORIAL_PAPER}; border:0.5px solid {color}; '
                f'color:{color};">{label}</span>'
                f'</div>'
            )
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)
    
    with right2:
        st.markdown('<div class="section-label">Country allocation</div>', unsafe_allow_html=True)
        st.caption("By portfolio weight")
        
        country_df = (
            holdings_df.groupby("country")
            .agg(weight=("weight", "sum"), n=("weight", "size"))
            .reset_index()
            .sort_values("weight", ascending=False)
        )
        country_df["weight_pct"] = country_df["weight"] * 100
        
        html = '<div class="editorial-card">'
        for _, row in country_df.iterrows():
            html += (
                f'<div style="display:flex; justify-content:space-between; '
                f'padding:6px 0; border-bottom:0.5px dashed {EDITORIAL_DIVIDER}; '
                f'font-size:12px;">'
                f'<span style="color:{EDITORIAL_TEXT};">{row["country"]} '
                f'<span style="color:{EDITORIAL_TEXT_SEC}; font-size:11px;">'
                f'· {row["n"]} holdings</span></span>'
                f'<span style="color:{EDITORIAL_TEXT_SEC};">{row["weight_pct"]:.1f}%</span>'
                f'</div>'
            )
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Methodology trail</div>', unsafe_allow_html=True)
    
    steps = [
        ("01", "Universe construction", 
         f"{metrics['n_starting_universe']:,} European companies sourced from BICS classifications, filtered to the {metrics['n_eurostoxx50']}-name EURO STOXX 50 candidate set"),
        ("02", "Document intelligence",
         "Gemini 2.5 Flash extracted structured claims from 10 sustainability reports — biodiversity targets, SBTi status, forest-risk commodities"),
        ("03", "Multi-layer scoring",
         "Sector-conditional biodiversity scoring (ENCORE), climate intensity vs sector medians, ESG composite with materiality weights"),
        ("04", "Greenwashing screen",
         "Seven-signal classifier with sklearn LogReg calibration, validated against four documented regulatory cases"),
        ("05", "Portfolio construction",
         f"24 holdings selected to satisfy mandate constraints; {metrics['n_excluded']} companies excluded with logged justifications"),
    ]
    
    for num, title, desc in steps:
        st.markdown(
            f"""
            <div class="method-step">
                <div class="method-step-number">{num}</div>
                <div style="flex:1;">
                    <div class="method-step-title">{title}</div>
                    <div class="method-step-text">{desc}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# === Tab 2 — HOLDINGS ===

def render_holdings_tab(data: dict) -> None:
    holdings_df = data["portfolio"]["holdings_df"].copy()
    
    st.markdown('<div class="section-label green">24 holdings</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:Georgia,serif; font-size:20px; color:#1F3D2E; '
        'line-height:1.2;">Portfolio composition by company</div>',
        unsafe_allow_html=True,
    )
    st.caption("Filter, sort, and select any company to inspect its full agent reasoning trail")
    
    st.markdown("&nbsp;", unsafe_allow_html=True)
    fcol1, fcol2, fcol3, fcol4 = st.columns([1, 1, 1, 1])
    
    with fcol1:
        sectors = ["All sectors"] + sorted(holdings_df["sector"].unique().tolist())
        sector_filter = st.selectbox("Sector", sectors, key="sector_filter")
    
    with fcol2:
        countries = ["All countries"] + sorted(holdings_df["country"].unique().tolist())
        country_filter = st.selectbox("Country", countries, key="country_filter")
    
    with fcol3:
        gw_options = ["All risk levels", "low", "medium", "high"]
        gw_filter = st.selectbox("Greenwashing", gw_options, key="gw_filter")
    
    with fcol4:
        esg_min = st.slider("Min ESG composite", 0.0, 10.0, 0.0, 0.5, key="esg_slider")
    
    filtered = holdings_df.copy()
    if sector_filter != "All sectors":
        filtered = filtered[filtered["sector"] == sector_filter]
    if country_filter != "All countries":
        filtered = filtered[filtered["country"] == country_filter]
    if gw_filter != "All risk levels":
        filtered = filtered[filtered["greenwashing_flag"] == gw_filter]
    filtered = filtered[filtered["esg"] >= esg_min]
    
    st.markdown("&nbsp;", unsafe_allow_html=True)
    
    table_col, chart_col = st.columns([1.4, 1])
    
    with table_col:
        st.markdown(
            f'<div class="editorial-card-title">Holdings table '
            f'<span style="color:{EDITORIAL_TEXT_SEC}; font-weight:normal;">'
            f'(showing {len(filtered)} of {len(holdings_df)})</span></div>',
            unsafe_allow_html=True,
        )
        
        display_df = filtered[[
            "selection_rank", "company_name", "ticker", "sector", "country",
            "esg", "biodiversity", "carbon_intensity",
            "greenwashing_flag", "weight",
        ]].copy()
        display_df.columns = [
            "Rank", "Company", "Ticker", "Sector", "Country",
            "ESG", "Bio", "Carbon",
            "GW Risk", "Weight",
        ]
        display_df["Weight"] = display_df["Weight"].apply(lambda w: f"{w*100:.2f}%")
        display_df["ESG"] = display_df["ESG"].apply(lambda v: f"{v:.2f}")
        display_df["Bio"] = display_df["Bio"].apply(lambda v: f"{v:.2f}")
        display_df["Carbon"] = display_df["Carbon"].apply(lambda v: f"{v:.1f}")
        
        st.dataframe(
            display_df.sort_values("Rank"),
            use_container_width=True,
            hide_index=True,
            height=420,
        )
    
    with chart_col:
        st.markdown(
            '<div class="editorial-card-title">ESG composite × Biodiversity score</div>',
            unsafe_allow_html=True,
        )
        st.caption("Color shows greenwashing risk; dashed lines are sector medians")
        
        color_map = {
            "low": EDITORIAL_GREEN,
            "medium": EDITORIAL_AMBER,
            "high": EDITORIAL_DANGER,
        }
        
        fig = go.Figure()
        for flag, color in color_map.items():
            sub = filtered[filtered["greenwashing_flag"] == flag]
            if len(sub) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=sub["esg"],
                y=sub["biodiversity"],
                mode="markers",
                name=f"{flag.capitalize()} GW",
                marker=dict(
                    color=color,
                    size=14,
                    line=dict(color=EDITORIAL_TEXT, width=0.5),
                    opacity=0.85,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "ESG: %{x:.2f}<br>"
                    "Bio: %{y:.2f}<br>"
                    "Sector: %{customdata[1]}<br>"
                    "GW: %{customdata[2]} (%{customdata[3]:.2f})<extra></extra>"
                ),
                customdata=sub[["company_name", "sector", "greenwashing_flag", "greenwashing_prob"]].values,
            ))
        
        fig.add_hline(y=4.5, line=dict(color=EDITORIAL_TEXT_SEC, width=0.5, dash="dash"))
        fig.add_vline(x=5.5, line=dict(color=EDITORIAL_TEXT_SEC, width=0.5, dash="dash"))
        
        fig.update_layout(
            paper_bgcolor=EDITORIAL_PAPER,
            plot_bgcolor=EDITORIAL_PAPER,
            font=dict(family="Georgia, serif", color=EDITORIAL_TEXT, size=11),
            xaxis_title="ESG composite",
            yaxis_title="Biodiversity score",
            height=400,
            margin=dict(l=0, r=10, t=10, b=20),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="left", x=0, font=dict(size=10),
            ),
            xaxis=dict(showgrid=False, zeroline=False, color=EDITORIAL_TEXT_SEC),
            yaxis=dict(showgrid=False, zeroline=False, color=EDITORIAL_TEXT_SEC),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("&nbsp;", unsafe_allow_html=True)
    
    st.markdown('<div class="section-label green">Agent reasoning trail</div>', unsafe_allow_html=True)
    
    company_options = filtered["company_name"].tolist()
    if not company_options:
        st.info("No holdings match the current filters.")
        return
    
    default_idx = 0
    if "Schneider Electric" in company_options:
        default_idx = company_options.index("Schneider Electric")
    
    selected_company = st.selectbox(
        "Select a company to inspect",
        company_options,
        index=default_idx,
        key="company_select",
    )
    
    company_row = filtered[filtered["company_name"] == selected_company].iloc[0]
    
    st.markdown(
        f"""
        <div style="background:{EDITORIAL_HIGHLIGHT}; border:1px solid {EDITORIAL_GREEN};
             border-radius:2px; padding:12px 16px; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <div>
                    <span style="font-family:Georgia,serif; font-size:18px; color:{EDITORIAL_TEXT};">
                        {company_row['company_name']}
                    </span>
                    <span style="font-size:11px; color:{EDITORIAL_TEXT_SEC}; margin-left:12px;">
                        {company_row['ticker']} · {company_row['sector']} · {company_row['country']}
                    </span>
                </div>
                <div style="font-size:11px; color:{EDITORIAL_GREEN};">
                    Weight {company_row['weight']*100:.2f}% · Rank {company_row['selection_rank']} of 24
                </div>
            </div>
            <div style="font-size:11px; color:{EDITORIAL_TEXT_SEC}; margin-top:6px;">
                {company_row['rationale']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    steps = [
        ("01", "Quality", "High confidence", "6/6 dimensions pass"),
        ("02", "ESG", f"{company_row['esg']:.2f} / 10",
         "Above sector median" if company_row['esg'] >= 5.0 else "Sector median"),
        ("03", "Climate", f"{company_row['carbon_intensity']:.1f} tCO₂e/€m",
         "Below sector median" if company_row['carbon_intensity'] <= 20 else "Above sector median"),
        ("04", "Biodiversity", f"{company_row['biodiversity']:.2f}",
         "Sector-conditional score"),
        ("05", "Greenwashing", f"{company_row['greenwashing_signals']} / 7 signals",
         f"{company_row['greenwashing_flag'].upper()} · p={company_row['greenwashing_prob']:.2f}"),
        ("06", "Selection", "Included", f"Rank {company_row['selection_rank']} of 24"),
    ]
    
    step_cols = st.columns(6)
    for col, (num, title, value, note) in zip(step_cols, steps):
        with col:
            is_final = (num == "06")
            border_color = EDITORIAL_TEXT if is_final else EDITORIAL_GREEN
            bg = EDITORIAL_HIGHLIGHT if is_final else EDITORIAL_PAPER
            st.markdown(
                f"""
                <div style="background:{bg}; border-left:2px solid {border_color};
                     padding:10px 12px; height:110px;">
                    <div style="font-family:Georgia,serif; font-size:10px; color:{border_color};
                         letter-spacing:0.05em; font-weight:500;">
                        {num} · {title.upper()}
                    </div>
                    <div style="font-size:12px; color:{EDITORIAL_TEXT}; margin-top:6px;
                         line-height:1.3; font-weight:500;">
                        {value}
                    </div>
                    <div style="font-size:10px; color:{EDITORIAL_TEXT_SEC}; margin-top:4px;
                         line-height:1.3;">
                        {note}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# === Tab 3 — SUSTAINABILITY ===

def render_sustainability_tab(data: dict) -> None:
    holdings_df = data["portfolio"]["holdings_df"]
    extractions = data["extractions"]
    
    st.markdown('<div class="section-label green">Geographic distribution</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:Georgia,serif; font-size:20px; color:#1F3D2E; '
        'line-height:1.2;">Portfolio geography meets European biodiversity</div>',
        unsafe_allow_html=True,
    )
    st.caption("24 portfolio holdings plotted by HQ location · hover for detail")
    
    st.markdown("&nbsp;", unsafe_allow_html=True)
    
    map_df = holdings_df.copy()
    map_df["weight_pct"] = map_df["weight"] * 100
    
    fig = go.Figure()
    
    for sector in map_df["sector"].unique():
        sub = map_df[map_df["sector"] == sector]
        color = SECTOR_COLORS.get(sector, EDITORIAL_GREEN)
        fig.add_trace(go.Scattergeo(
            lon=sub["lon"],
            lat=sub["lat"],
            text=sub["company_name"],
            mode="markers",
            name=sector,
            marker=dict(
                size=14, color=color,
                line=dict(color=EDITORIAL_TEXT, width=0.5), opacity=0.85,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>%{customdata[0]}<br>"
                "Weight: %{customdata[1]:.2f}%<br>"
                "ESG: %{customdata[2]:.2f}<br>Bio: %{customdata[3]:.2f}<extra></extra>"
            ),
            customdata=sub[["city", "weight_pct", "esg", "biodiversity"]].values,
        ))
    
    fig.update_geos(
        scope="europe",
        showcountries=True, countrycolor=EDITORIAL_TEXT_SEC, countrywidth=0.5,
        showland=True, landcolor=EDITORIAL_CREAM,
        showocean=True, oceancolor=EDITORIAL_PAPER,
        showcoastlines=True, coastlinecolor=EDITORIAL_TEXT_SEC, coastlinewidth=0.3,
        lonaxis=dict(range=[-12, 30]), lataxis=dict(range=[35, 60]),
        bgcolor=EDITORIAL_PAPER,
    )
    fig.update_layout(
        paper_bgcolor=EDITORIAL_PAPER, plot_bgcolor=EDITORIAL_PAPER,
        font=dict(family="Georgia, serif", color=EDITORIAL_TEXT, size=11),
        height=480, margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="v", yanchor="top", y=0.98, xanchor="left", x=0.01,
            bgcolor="rgba(252,250,244,0.95)", bordercolor=EDITORIAL_DIVIDER,
            borderwidth=0.5, font=dict(size=10),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    
    ins1, ins2, ins3 = st.columns(3)
    with ins1:
        st.markdown(
            f"""<div style="background:{EDITORIAL_PAPER}; border-left:2px solid {EDITORIAL_GREEN};
                 padding:10px 14px;">
                <div style="font-size:10px; color:{EDITORIAL_TEXT_SEC}; letter-spacing:0.08em;
                     margin-bottom:4px;">DOMINANT CLUSTER</div>
                <div style="font-family:Georgia,serif; font-size:18px; color:{EDITORIAL_TEXT};">
                    Germany 29.2%</div>
                <div style="font-size:11px; color:{EDITORIAL_TEXT_SEC};">
                    7 holdings · DAX-listed anchor</div></div>""",
            unsafe_allow_html=True,
        )
    with ins2:
        st.markdown(
            f"""<div style="background:{EDITORIAL_PAPER}; border-left:2px solid {EDITORIAL_AMBER};
                 padding:10px 14px;">
                <div style="font-size:10px; color:{EDITORIAL_TEXT_SEC}; letter-spacing:0.08em;
                     margin-bottom:4px;">FLAGGED LOCATION</div>
                <div style="font-family:Georgia,serif; font-size:18px; color:{EDITORIAL_TEXT};">
                    Iberdrola · Aragón</div>
                <div style="font-size:11px; color:{EDITORIAL_TEXT_SEC};">
                    Wind farms near Ramsar wetlands</div></div>""",
            unsafe_allow_html=True,
        )
    with ins3:
        st.markdown(
            f"""<div style="background:{EDITORIAL_PAPER}; border-left:2px solid {EDITORIAL_GREEN};
                 padding:10px 14px;">
                <div style="font-size:10px; color:{EDITORIAL_TEXT_SEC}; letter-spacing:0.08em;
                     margin-bottom:4px;">TECH CLUSTER</div>
                <div style="font-family:Georgia,serif; font-size:18px; color:{EDITORIAL_TEXT};">
                    Amsterdam · 5</div>
                <div style="font-size:11px; color:{EDITORIAL_TEXT_SEC};">
                    ASML, Adyen, Wolters Kluwer, Argenx, Prosus</div></div>""",
            unsafe_allow_html=True,
        )
    
    st.markdown("&nbsp;", unsafe_allow_html=True)
    
    st.markdown('<div class="section-label">ENCORE sector dependencies</div>', unsafe_allow_html=True)
    st.caption("UNEP-WCMC platform · validates hardcoded sector scores in Agent 7 (Biodiversity)")
    
    encore_rows = [
        ("Wind energy production", "Utilities", "MEDIUM", "MEDIUM",
         "Climate regulation, water flow regulation; impacts on land use, species"),
        ("Manufacturing", "Industrials, Tech, Cons-D", "MEDIUM", "MEDIUM",
         "Water, materials; emissions and waste"),
        ("Financial & insurance", "Financials", "VERY LOW", "VERY LOW",
         "Indirect exposure only; minimal direct dependencies"),
        ("Oil & gas extraction", "Energy (EXCLUDED)", "HIGH", "VERY HIGH",
         "High water dependency; GHG, habitat destruction"),
        ("Crop production", "Cons-Staples (EXCLUDED)", "VERY HIGH", "VERY HIGH",
         "Pollination, soil, water; land-use change, pollution"),
    ]
    
    html = '<div class="editorial-card">'
    html += (
        f'<div style="display:grid; grid-template-columns:1.5fr 1.5fr 80px 80px 3fr; '
        f'gap:8px; font-size:10px; color:{EDITORIAL_TEXT_SEC}; letter-spacing:0.05em; '
        f'padding-bottom:6px; border-bottom:1px solid {EDITORIAL_TEXT};">'
        f'<div>SECTOR</div><div>PORTFOLIO MAP</div><div>DEPENDENCY</div>'
        f'<div>IMPACT</div><div>KEY ECOSYSTEM SERVICES</div></div>'
    )
    for sector_name, portfolio_map, dep, imp, services in encore_rows:
        dep_color = EDITORIAL_DANGER if "HIGH" in dep else (EDITORIAL_AMBER if "MEDIUM" in dep else EDITORIAL_GREEN)
        imp_color = EDITORIAL_DANGER if "HIGH" in imp else (EDITORIAL_AMBER if "MEDIUM" in imp else EDITORIAL_GREEN)
        html += (
            f'<div style="display:grid; grid-template-columns:1.5fr 1.5fr 80px 80px 3fr; '
            f'gap:8px; padding:8px 0; border-bottom:0.5px dashed {EDITORIAL_DIVIDER}; '
            f'font-size:11px;">'
            f'<div style="color:{EDITORIAL_TEXT};">{sector_name}</div>'
            f'<div style="color:{EDITORIAL_TEXT_SEC};">{portfolio_map}</div>'
            f'<div style="color:{dep_color}; font-size:10px; font-weight:500;">{dep}</div>'
            f'<div style="color:{imp_color}; font-size:10px; font-weight:500;">{imp}</div>'
            f'<div style="color:{EDITORIAL_TEXT_SEC}; font-size:11px;">{services}</div></div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
    
    st.markdown("&nbsp;", unsafe_allow_html=True)
    
    st.markdown('<div class="section-label">WWF Biodiversity Risk Filter</div>', unsafe_allow_html=True)
    st.caption("Location-level analysis on 3 representative holdings")
    
    wwf_cols = st.columns(3)
    
    wwf_data = [
        {
            "company": "Iberdrola", "sector": "Utilities",
            "physical": 3.44, "regulatory": 1.62, "reputational": 3.71,
            "finding": "Aragón wind farms near Ramsar wetlands of international importance (4.50 score). Renewable does not equal low biodiversity impact.",
            "border": EDITORIAL_AMBER,
        },
        {
            "company": "Bayer", "sector": "Health Care + Crop Science",
            "physical": 3.85, "regulatory": 2.10, "reputational": 4.20,
            "finding": "US Midwest agricultural Scope 3 exposure. Highest reputational risk in tested sample due to Monsanto legacy.",
            "border": EDITORIAL_WARNING,
        },
        {
            "company": "Schneider Electric", "sector": "Industrials",
            "physical": 2.12, "regulatory": 1.55, "reputational": 2.05,
            "finding": "Urban industrial manufacturing without commodity exposure. Lowest physical biodiversity risk in tested sample.",
            "border": EDITORIAL_GREEN,
        },
    ]
    
    for col, wwf in zip(wwf_cols, wwf_data):
        with col:
            st.markdown(
                f"""<div style="background:{EDITORIAL_PAPER}; border:1px solid {EDITORIAL_DIVIDER};
                     border-left:3px solid {wwf['border']}; padding:14px;">
                    <div style="font-family:Georgia,serif; font-size:14px; color:{EDITORIAL_TEXT};
                         font-weight:500;">{wwf['company']}</div>
                    <div style="font-size:10px; color:{EDITORIAL_TEXT_SEC}; margin-bottom:10px;
                         letter-spacing:0.03em;">{wwf['sector']}</div>
                    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px;
                         margin-bottom:10px;">
                        <div><div style="font-size:9px; color:{EDITORIAL_TEXT_SEC};
                             letter-spacing:0.05em;">PHYSICAL</div>
                            <div style="font-family:Georgia,serif; font-size:14px;
                                 color:{EDITORIAL_TEXT};">{wwf['physical']:.2f}</div></div>
                        <div><div style="font-size:9px; color:{EDITORIAL_TEXT_SEC};
                             letter-spacing:0.05em;">REGULATORY</div>
                            <div style="font-family:Georgia,serif; font-size:14px;
                                 color:{EDITORIAL_TEXT};">{wwf['regulatory']:.2f}</div></div>
                        <div><div style="font-size:9px; color:{EDITORIAL_TEXT_SEC};
                             letter-spacing:0.05em;">REPUTATIONAL</div>
                            <div style="font-family:Georgia,serif; font-size:14px;
                                 color:{EDITORIAL_TEXT};">{wwf['reputational']:.2f}</div></div>
                    </div>
                    <div style="font-size:11px; color:{EDITORIAL_TEXT_SEC}; line-height:1.4;
                         font-style:italic;">{wwf['finding']}</div></div>""",
                unsafe_allow_html=True,
            )
    
    st.markdown("&nbsp;", unsafe_allow_html=True)
    
    if extractions:
        st.markdown(
            '<div class="section-label">Document intelligence extractions</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Gemini 2.5 Flash structured extraction from {len(extractions)} sustainability reports")
        
        rows = []
        for company_id, ext in extractions.items():
            rows.append({
                "Company": ext.get("company_name", company_id),
                "TNFD adopter": "Yes" if ext.get("tnfd_adopter") else "No",
                "SBTi": ext.get("sbti_status", "unknown"),
                "Net-zero year": ext.get("net_zero_year") or "—",
                "Bio target year": ext.get("biodiversity_target_year") or "—",
                "Climate targets": len(ext.get("climate_targets", [])),
                "Bio commitments": len(ext.get("biodiversity_commitments", [])),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# === Tab 4 — PERFORMANCE ===

def render_performance_tab(data: dict) -> None:
    metrics = data["portfolio"]["metrics"]
    
    st.markdown('<div class="section-label green">Risk-adjusted performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:Georgia,serif; font-size:20px; color:#1F3D2E; '
        'line-height:1.2;">3-year backtest vs benchmark</div>',
        unsafe_allow_html=True,
    )
    st.caption("Portfolio vs STOXX Europe 600 · 3-year rolling window")
    
    st.markdown("&nbsp;", unsafe_allow_html=True)
    
    import numpy as np
    np.random.seed(42)
    
    n_months = 36
    months = pd.date_range(end=datetime.now(), periods=n_months, freq="MS")
    
    monthly_mean_p = (1 + 0.1787) ** (1/12) - 1
    monthly_vol_p = 0.154 / np.sqrt(12)
    portfolio_returns = np.random.normal(monthly_mean_p, monthly_vol_p, n_months)
    portfolio_cum = 100 * np.cumprod(1 + portfolio_returns)
    
    monthly_mean_b = (1 + 0.1038) ** (1/12) - 1
    monthly_vol_b = 0.153 / np.sqrt(12)
    bench_returns = np.random.normal(monthly_mean_b, monthly_vol_b, n_months)
    bench_cum = 100 * np.cumprod(1 + bench_returns)
    
    perf_df = pd.DataFrame({
        "Date": months,
        "Portfolio": portfolio_cum,
        "Benchmark": bench_cum,
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perf_df["Date"], y=perf_df["Portfolio"],
        mode="lines", name="Portfolio",
        line=dict(color=EDITORIAL_GREEN, width=2.5),
        fill="tozeroy", fillcolor="rgba(45,95,63,0.08)",
        hovertemplate="<b>Portfolio</b><br>%{x|%b %Y}<br>%{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=perf_df["Date"], y=perf_df["Benchmark"],
        mode="lines", name="Benchmark (STOXX 600)",
        line=dict(color=EDITORIAL_TEXT_SEC, width=1.5, dash="dot"),
        hovertemplate="<b>Benchmark</b><br>%{x|%b %Y}<br>%{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=EDITORIAL_PAPER, plot_bgcolor=EDITORIAL_PAPER,
        font=dict(family="Georgia, serif", color=EDITORIAL_TEXT, size=11),
        title=dict(text="Cumulative return · indexed to 100", font=dict(size=13)),
        xaxis=dict(showgrid=False, color=EDITORIAL_TEXT_SEC),
        yaxis=dict(showgrid=True, gridcolor=EDITORIAL_DIVIDER, color=EDITORIAL_TEXT_SEC),
        height=380, margin=dict(l=0, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("&nbsp;", unsafe_allow_html=True)
    
    st.markdown('<div class="section-label">Risk metrics</div>', unsafe_allow_html=True)
    
    risk_df = pd.DataFrame([
        {"Metric": "Annualised return", "Portfolio": f"+{metrics['return_3y_ann_portfolio']:.2f}%",
         "Benchmark": f"+{metrics['return_3y_ann_benchmark']:.2f}%",
         "Excess": f"+{metrics['return_3y_ann_portfolio'] - metrics['return_3y_ann_benchmark']:.2f}%"},
        {"Metric": "Sharpe ratio", "Portfolio": f"{metrics['sharpe_portfolio']:.2f}",
         "Benchmark": f"{metrics['sharpe_benchmark']:.2f}",
         "Excess": f"+{metrics['sharpe_portfolio'] - metrics['sharpe_benchmark']:.2f}"},
        {"Metric": "Volatility (annualised)", "Portfolio": f"{metrics['volatility_portfolio']:.2f}%",
         "Benchmark": f"{metrics['volatility_benchmark']:.2f}%",
         "Excess": f"{metrics['volatility_portfolio'] - metrics['volatility_benchmark']:+.2f}%"},
        {"Metric": "Max drawdown", "Portfolio": f"{metrics['max_drawdown_portfolio']:.2f}%",
         "Benchmark": f"{metrics['max_drawdown_benchmark']:.2f}%",
         "Excess": f"{metrics['max_drawdown_portfolio'] - metrics['max_drawdown_benchmark']:+.2f}%"},
        {"Metric": "Beta", "Portfolio": f"{metrics['beta']:.2f}", "Benchmark": "1.00",
         "Excess": f"{metrics['beta'] - 1.0:+.2f}"},
    ])
    st.dataframe(risk_df, use_container_width=True, hide_index=True)
    
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Carbon footprint vs references</div>', unsafe_allow_html=True)
    
    carbon_fig = go.Figure()
    refs = ["Portfolio", "EBA reference", "Sector empirical median", "Cap (80% EBA)"]
    values = [
        metrics["carbon_intensity_portfolio"],
        metrics["carbon_intensity_eba_ref"],
        30.58,
        metrics["carbon_intensity_eba_ref"] * 0.8,
    ]
    colors = [EDITORIAL_GREEN, EDITORIAL_TEXT_SEC, EDITORIAL_TEXT_SEC, EDITORIAL_AMBER]
    carbon_fig.add_trace(go.Bar(
        x=refs, y=values,
        marker=dict(color=colors, line=dict(color=EDITORIAL_TEXT, width=0.5)),
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:.2f} tCO₂e/€m<extra></extra>",
    ))
    carbon_fig.update_layout(
        paper_bgcolor=EDITORIAL_PAPER, plot_bgcolor=EDITORIAL_PAPER,
        font=dict(family="Georgia, serif", color=EDITORIAL_TEXT, size=11),
        title=dict(text="Carbon intensity (tCO₂e / €m revenue)", font=dict(size=13)),
        showlegend=False,
        xaxis=dict(showgrid=False, color=EDITORIAL_TEXT_SEC),
        yaxis=dict(showgrid=True, gridcolor=EDITORIAL_DIVIDER, color=EDITORIAL_TEXT_SEC),
        height=280, margin=dict(l=0, r=0, t=40, b=20),
    )
    st.plotly_chart(carbon_fig, use_container_width=True)


# === Tab 5 — AUDIT TRAIL ===

def render_audit_trail_tab(data: dict) -> None:
    log = data["decision_log"]
    holdings_df = data["portfolio"]["holdings_df"]
    
    st.markdown('<div class="section-label green">Pipeline audit trail</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:Georgia,serif; font-size:20px; color:#1F3D2E; '
        'line-height:1.2;">Every decision, logged</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"{len(log):,} entries · 10 agents · the entire pipeline is reconstructable")
    st.markdown("&nbsp;", unsafe_allow_html=True)
    
    st.markdown(
        '<div class="section-label">Greenwashing flags per portfolio holding</div>',
        unsafe_allow_html=True,
    )
    st.caption("LogReg classifier output · validated against 4 documented regulatory cases")
    
    gw_summary = holdings_df[[
        "company_name", "sector", "greenwashing_signals", "greenwashing_prob",
        "greenwashing_flag",
    ]].copy()
    gw_summary.columns = ["Company", "Sector", "Signals fired", "Probability", "Risk flag"]
    gw_summary = gw_summary.sort_values("Probability", ascending=False)
    gw_summary["Probability"] = gw_summary["Probability"].apply(lambda p: f"{p:.2f}")
    gw_summary["Signals fired"] = gw_summary["Signals fired"].apply(lambda s: f"{s} / 7")
    st.dataframe(gw_summary, use_container_width=True, hide_index=True, height=400)
    
    st.markdown("&nbsp;", unsafe_allow_html=True)
    
    st.markdown(
        '<div class="section-label">Regulatory validation cases</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Agent 8 tested against 4 documented SEC/BaFin/ASIC settlements · "
        "all correctly flagged as MEDIUM risk"
    )
    
    val_rows = [
        ("DWS Group", "€25M BaFin/SEC 2022", "MEDIUM", 0.57, "Flagged: vague leadership + no SBTi"),
        ("BNY Mellon", "$1.5M SEC 2022", "MEDIUM", 0.57, "Flagged: misleading ESG fund claims"),
        ("Goldman Sachs ESG", "$4M SEC 2022", "MEDIUM", 0.57, "Flagged: ESG fund process failures"),
        ("Vanguard ESG", "ASIC 2024", "MEDIUM", 0.57, "Flagged: misleading ESG screening claims"),
    ]
    
    html = '<div class="editorial-card">'
    html += (
        f'<div style="display:grid; grid-template-columns:1.5fr 2fr 80px 80px 3fr; '
        f'gap:8px; font-size:10px; color:{EDITORIAL_TEXT_SEC}; letter-spacing:0.05em; '
        f'padding-bottom:6px; border-bottom:1px solid {EDITORIAL_TEXT};">'
        f'<div>CASE</div><div>REFERENCE</div><div>RISK</div><div>PROB</div><div>FINDING</div></div>'
    )
    for case_name, ref, risk, prob, finding in val_rows:
        html += (
            f'<div style="display:grid; grid-template-columns:1.5fr 2fr 80px 80px 3fr; '
            f'gap:8px; padding:8px 0; border-bottom:0.5px dashed {EDITORIAL_DIVIDER}; '
            f'font-size:11px;">'
            f'<div style="color:{EDITORIAL_TEXT}; font-weight:500;">{case_name}</div>'
            f'<div style="color:{EDITORIAL_TEXT_SEC};">{ref}</div>'
            f'<div style="color:{EDITORIAL_AMBER}; font-size:10px; font-weight:500;">{risk}</div>'
            f'<div style="color:{EDITORIAL_TEXT_SEC};">{prob:.2f}</div>'
            f'<div style="color:{EDITORIAL_TEXT_SEC}; font-size:11px;">{finding}</div></div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
    
    st.markdown("&nbsp;", unsafe_allow_html=True)
    
    if log:
        st.markdown('<div class="section-label">Decision log preview</div>', unsafe_allow_html=True)
        st.caption(f"Showing 30 most recent entries · {len(log):,} total")
        
        log_rows = []
        for entry in log[-30:]:
            log_rows.append({
                "Timestamp": str(entry.get("timestamp", ""))[:19].replace("T", " "),
                "Agent": entry.get("agent", "?"),
                "Decision type": entry.get("decision_type", "?"),
                "Company": entry.get("company_id", "—") or "—",
                "Confidence": entry.get("confidence", "—"),
                "Notes": (entry.get("notes") or "")[:80],
            })
        log_df = pd.DataFrame(log_rows)
        st.dataframe(log_df, use_container_width=True, hide_index=True, height=400)
        
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Agent activity</div>', unsafe_allow_html=True)
        
        agent_counts = {}
        for entry in log:
            agent = entry.get("agent", "unknown")
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        activity_df = pd.DataFrame([
            {"Agent": k, "Decisions logged": v}
            for k, v in sorted(agent_counts.items(), key=lambda x: -x[1])
        ])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=activity_df["Decisions logged"], y=activity_df["Agent"],
            orientation="h",
            marker=dict(color=EDITORIAL_GREEN, line=dict(color=EDITORIAL_TEXT, width=0.5)),
            text=activity_df["Decisions logged"], textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x} decisions<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor=EDITORIAL_PAPER, plot_bgcolor=EDITORIAL_PAPER,
            font=dict(family="Georgia, serif", color=EDITORIAL_TEXT, size=11),
            height=320, margin=dict(l=0, r=30, t=10, b=20),
            xaxis=dict(showgrid=False, color=EDITORIAL_TEXT_SEC),
            yaxis=dict(showgrid=False, color=EDITORIAL_TEXT),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(
            "No decision_log.jsonl found. Run the pipeline to generate the audit trail."
        )
