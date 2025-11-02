# Floodzilla Light-AI Toolkit (Nonprofit-friendly)

Low-cost, CPU-only flood analytics with explainable risk scores, lightweight forecasting, and cached LLM summaries powered by `strandsagentsdk` (Gemini 2.0 Flash-Lite).

## Contents
- [Highlights](#highlights)
- [Architecture Overview](#architecture-overview)
- [Project Layout](#project-layout)
- [Data Preparation Workflow](#data-preparation-workflow)
- [Running the Toolkit](#running-the-toolkit)
- [Environment & Configuration](#environment--configuration)
- [Outputs](#outputs)
- [Risk Score](#risk-score)
- [Further Reading](#further-reading)

## Highlights
- CPU-friendly pipeline: pandas, Prophet, and Streamlit run comfortably on commodity hardware.
- Responsible LLM usage: rate-limited Strands Agent client with on-disk cache and offline fallback.
- Partitioned Parquet datasets for fast incremental updates and downstream analytics.
- Streamlit dashboard combines observations, forecasts, heuristic risk, and weekly summaries.

## Architecture Overview
```mermaid
flowchart LR
    raw[Raw Gauge JSON<br/>(gauge_readings_3y/)] --> curate(curate_data.py<br/>Clean + quality flags)
    curate --> curated[dataset_parquet/]
    curated --> features(build_features.py<br/>Feature engineering)
    features --> features_dir[features_parquet/]
    features_dir --> summaries(summaries_strands.py<br/>Weekly LLM summaries)
    summaries --> summaries_dir[summaries/]
    curated --> forecast(forecast_prophet.py<br/>Prophet per gauge)
    forecast --> forecast_dir[forecasts_parquet/]
    curated --> dashboard(dashboard_app.py<br/>Streamlit dashboard)
    features_dir --> dashboard
    forecast_dir --> dashboard
    summaries_dir --> dashboard
```
See `docs/system_overview.md` for a deeper technical walkthrough.

## Project Layout
| Path | Purpose |
| --- | --- |
| `curate_data.py` | Normalize raw JSON gauge readings into evenly sampled Parquet with quality flags. |
| `build_features.py` | Time-series feature engineering for cheap ML and summary metrics. |
| `forecast_prophet.py` | Prophet forecasts per gauge (48 steps) using CmdStanPy backend. |
| `summaries_strands.py` | Weekly LLM summaries via Strands Agent SDK with quota cache + offline fallback. |
| `dashboard_app.py` | Streamlit situational awareness dashboard combining metrics, forecasts, and summaries. |
| `utils/quota_cache.py` | Lightweight disk-backed cache and RPM/RPD quota helper. |
| `requirements.txt` | Python dependencies (CPU-only stack). |
| `docs/system_overview.md` | Extended documentation covering flow, components, and troubleshooting. |

## Data Preparation Workflow
1. **Collect raw readings** from your upstream fetcher/export into `gauge_readings_3y/` as files named `gauge_<ID>_readings_3y.json` (each containing three years of status + `waterHeight` history).
2. **Curate and resample** the JSON series:
   ```bash
   python curate_data.py
   ```
   This trims deleted rows, interpolates small gaps, flags flats/jumps/outliers, and writes partitioned Parquet to `dataset_parquet/`.
3. **Feature engineering** for downstream analytics:
   ```bash
   python build_features.py
   ```
   Produces lagged metrics, rolling stats, and quality indicators in `features_parquet/`.
4. **Forecast preparation** (optional but recommended):
   ```bash
   # one-time setup if CmdStan is not installed
   python -m cmdstanpy.install_cmdstan

   python forecast_prophet.py
   ```
   Stores per-gauge Prophet forecasts in `forecasts_parquet/`.

You can rerun any step incrementally; partitioned outputs avoid rewriting untouched gauges.

## Running the Toolkit
1. Create and activate a virtual environment, then install dependencies:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Generate weekly LLM summaries (optional without API key):
   ```bash
   export STRANDS_API_KEY=YOUR_KEY  # also reused for LiteLLM if present
   python summaries_strands.py
   ```
   The script enforces ~30 RPM / 200 RPD via `QuotaManager` and falls back to an offline template when the key or SDK is missing.
3. Launch the Streamlit dashboard:
   ```bash
   streamlit run dashboard_app.py
   ```
   Use the sidebar gauge selector to explore observations, forecasts, risk scores, and summaries.

## Environment & Configuration
- Python 3.10+ recommended.
- Optional Strands Agent SDK: `pip install "strands-agents[litellm]"`.
- Environment variables:
  - `STRANDS_API_KEY` (and `LITELLM_API_KEY`) for live LLM calls.
- Prophet requires a working CmdStan installation; run `python -m cmdstanpy.install_cmdstan` inside your virtualenv if you have not set it up.

## Outputs
```
dataset_parquet/           # curated series + flags (partitioned by gauge/year)
features_parquet/          # engineered features (partitioned)
forecasts_parquet/         # Prophet outputs (partitioned)
summaries/weekly_YYYY-MM-DD.txt
llm_cache/*.json           # cached LLM responses
```

## Risk Score
A lightweight heuristic displayed in the dashboard combines:
- normalized level relative to the 95th percentile,
- recent rate-of-rise,
- anomaly counts (jump/outlier flags).

The score is explainable, fast, and model-free.

## Further Reading
- `docs/system_overview.md` for a detailed component breakdown.
- Open an issue or PR with ideas and improvements—contributions welcome!
