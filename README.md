# Biodiversity-Aware Portfolio Pipeline

Multi-agent AI research pipeline for sustainable portfolio construction for the elective of
Sustainable Finance 

## Python Skeleton

Contains:

- **Schemas** — Pydantic data contracts so each role's outputs flow cleanly into the master agent
- **Agent stubs** — placeholder modules each role replaces with real implementation
- **Decision logger** — shared audit trail every agent writes to
- **Working example** — Data Quality agent + tests, demonstrating the pattern



## Folder structure

```
biodiversity-portfolio-pipeline/
├── README.md                  
├── requirements.txt           # Python dependencies
├── pipeline.py                # Main orchestrator 
│
├── agents/                    # One module per agent
│   ├── __init__.py
│   ├── base.py                # BaseAgent class everyone inherits from
│   ├── decision_log.py        # Shared audit trail logger
│   ├── data_quality.py        # Working example — copy this pattern
│   └── stubs.py               # Stubs for every agent 
│
├── schemas/                   # Pydantic data contracts 
│   ├── company.py             # Company identifiers + universe
│   ├── financial.py           # Returns, vol, drawdown, Sharpe
│   ├── esg.py                 # E, S, G scores
│   ├── biodiversity.py        # Multi-layer nature risk + climate
│   ├── greenwashing.py        # Greenwashing flags + document evidence
│   ├── portfolio.py           # Final portfolio + holdings + overrides
│   └── confidence.py          # Confidence levels for every data point
│
├── data/
│   ├── raw/                   # Course data pack CSVs (gitignored)
│   ├── processed/             # Cleaned outputs from data quality agent
│   ├── documents/             # CSR reports, TCFD disclosures, etc.
│   └── cached/                # yfinance cache 
│
├── outputs/
│   ├── logs/                  # decision_log.jsonl lives here
│   ├── reports/               # Generated factsheet, methodology doc
│   └── figures/               # Charts and visualisations
│
├── notebooks/                 # Jupyter notebooks — exploration 
│
├── tests/                     # pytest tests 
│
└── docs/                      # Additional documentation
```


## Tools we use

We use a **hybrid AI stack** — picking the right tool for each job:

| Tool | Used by | Purpose |
|---|---|---|
| **Claude (Anthropic API)** | Roles C, D, E | Document intelligence, claim extraction, report drafting |
| **Hugging Face: FinBERT** | Role D | Financial sentiment on news/disclosures |
| **HF: sentence-transformers** | Role D | Text embeddings for semantic search |
| **HF: bart-large-mnli** | Role D | Zero-shot classification fallback |
| **scikit-learn** | Roles A, D | Trained classifiers (greenwashing, imputation) |
| **cvxpy** | Role E | Constrained portfolio optimisation |
| **Streamlit + Folium** | Role E | Demo dashboard with biodiversity heat map |
| **yfinance** | Role A | Historical market data |


## Phase plan

| Phase | Dates | What's happening |
|---|---|---|
| Phase 1 | Now → 8 May | Mandate, data audit, document corpus, architecture proposal |
| Proposal | 8 May | One-page proposal submitted |
| Phase 2 | 9 → 15 May | Build real agents, replace stubs |
| Clinic | 15 May | Prototype clinic with Prof. Bhattacharya |
| Phase 3 | 15 → 22 May | Polish, dashboard, report assembly, rehearsal |
| Final | 22 May | Presentation |

