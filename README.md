# Biodiversity-Aware Portfolio Pipeline

Multi-agent AI research pipeline for sustainable portfolio construction.
Sustainable Finance · Prof. Budha Bhattacharya · MSc Finance + Business Analytics.

## What this is

This is the shared Python skeleton for our group project. It provides:

- **Schemas** — Pydantic data contracts so each role's outputs flow cleanly into the master agent
- **Agent stubs** — placeholder modules each role replaces with real implementation
- **Decision logger** — shared audit trail every agent writes to
- **Working example** — Data Quality agent + tests, demonstrating the pattern

## Quick start

```bash
# 1. Clone the repo
git clone <repo-url>
cd biodiversity-portfolio-pipeline

# 2. Set up environment (one-time)
python -m venv venv
source venv/bin/activate   # on Mac/Linux
# venv\Scripts\activate    # on Windows
pip install -r requirements.txt

# 3. Run the tests to confirm everything works
pytest tests/

# 4. Run the pipeline (currently a stub — you'll see this evolve in Phase 2)
python pipeline.py
```

If `pytest tests/` shows 3 passed, your environment is ready.

## Folder structure

```
biodiversity-portfolio-pipeline/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── pipeline.py                # Main orchestrator — runs the full pipeline
│
├── agents/                    # One module per agent
│   ├── __init__.py
│   ├── base.py                # BaseAgent class everyone inherits from
│   ├── decision_log.py        # Shared audit trail logger — IMPORT THIS
│   ├── data_quality.py        # Working example — copy this pattern
│   └── stubs.py               # Stubs for every agent (replace in Phase 2)
│
├── schemas/                   # Pydantic data contracts (DO NOT EDIT WITHOUT TELLING TEAM)
│   ├── company.py             # Company identifiers + universe
│   ├── financial.py           # Returns, vol, drawdown, Sharpe
│   ├── esg.py                 # E, S, G scores
│   ├── biodiversity.py        # Multi-layer nature risk + climate
│   ├── greenwashing.py        # Greenwashing flags + document evidence
│   ├── portfolio.py           # Final portfolio + holdings + overrides
│   └── confidence.py          # Confidence levels for every data point
│
├── data/
│   ├── raw/                   # Course data pack CSVs go here (gitignored)
│   ├── processed/             # Cleaned outputs from data quality agent
│   ├── documents/             # CSR reports, TCFD disclosures, etc.
│   └── cached/                # yfinance cache — don't refetch
│
├── outputs/
│   ├── logs/                  # decision_log.jsonl lives here
│   ├── reports/               # Generated factsheet, methodology doc
│   └── figures/               # Charts and visualisations
│
├── notebooks/                 # Jupyter notebooks — exploration only, not production
│
├── tests/                     # pytest tests — add 1-2 per agent
│
└── docs/                      # Additional documentation
```

## How to add your agent (per role)

1. **Decide what your agent's input and output schema is.** Reference `schemas/`.
   If you need a new field, propose it in our group chat before adding it.

2. **Copy the pattern from `agents/data_quality.py`**. It shows:
   - How to inherit from `BaseAgent`
   - How to log decisions
   - How to validate against schemas

3. **Implement your agent in `agents/your_agent.py`**.

4. **Write at least one test** in `tests/test_your_agent.py`. Look at
   `tests/test_skeleton.py` for the pattern.

5. **Update `pipeline.py`** to call your agent in the right phase.

6. **Don't forget to log decisions** — every meaningful choice your agent
   makes should call `self.log()`. This is the audit trail that defends
   our portfolio in Q&A.

## Logging decisions — the rule

Every agent must call `self.log()` at every meaningful decision point.
A meaningful decision is anything we'd be asked to defend in Q&A:

- Imputed a missing value? Log it.
- Excluded a company? Log it.
- Calculated a score? Log it.
- Flagged greenwashing? Log it.
- Human overrode an AI output? Log it (with `confidence="judgement_based"`).

The output goes to `outputs/logs/decision_log.jsonl`. This is our audit
trail. The Streamlit dashboard reads from it. The data dictionary
references it. The lecturer's hardest Q&A questions are answered from it.

## Confidence levels — the rule

Every data point in the final portfolio must carry one of these labels
(per the assignment brief, section 15):

| Label | When to use |
|---|---|
| `reported` | Disclosed by the company itself (CSR report, annual report) |
| `observed` | Measured externally (satellite, market price, regulatory filing) |
| `estimated` | Imputed or modelled (KNN, RF imputation, sector estimate) |
| `ai_extracted` | Pulled from documents by an LLM (Claude, HF model) |
| `judgement_based` | Analyst override or team decision |

Wrap your data points using `schemas.confidence.DataPoint`. Yes, it's
verbose. The audit trail is the grade.

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

## Important conventions

- **One LLM provider** (Claude) — not multiple. Cheaper, more consistent.
- **Pydantic schemas** for all inter-agent data — no raw dicts flying around.
- **Cache yfinance results** — don't refetch. Document the download date.
- **Don't commit raw data** to the repo (gitignored). Document where it came from.
- **Don't commit API keys**. Use a `.env` file (gitignored).

## Phase plan

| Phase | Dates | What's happening |
|---|---|---|
| Phase 1 | Now → 8 May | Mandate, data audit, document corpus, architecture proposal |
| Proposal | 8 May | One-page proposal submitted |
| Phase 2 | 9 → 15 May | Build real agents, replace stubs |
| Clinic | 15 May | Prototype clinic with Prof. Bhattacharya |
| Phase 3 | 15 → 22 May | Polish, dashboard, report assembly, rehearsal |
| Final | 22 May | Presentation 18:15-20:15 |

## Help

- Architecture questions → Analytics Advisor (Lara)
- Data questions → Role A
- ESG methodology questions → Role B
- Biodiversity questions → Role C
- Document/greenwashing questions → Role D
- Portfolio construction questions → Role E

If your tests fail, try `pytest tests/ -v` and share the output in the
group chat. Don't struggle alone for hours — that's not what the
"Analytics Advisor" role is for.
