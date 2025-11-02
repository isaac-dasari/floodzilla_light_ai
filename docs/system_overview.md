# Floodzilla Light AI – System Overview

This document explains the major features, data flow, and technical details of the Floodzilla Light AI workspace so the team can operate and extend it confidently.

---

## 1. Data Curation (`curate_data.py`)

| Aspect | Details |
| --- | --- |
| **Purpose** | Normalize raw JSON gauge readings into clean, evenly sampled Parquet tables with quality flags. |
| **Inputs** | `gauge_readings_3y/gauge_<ID>_readings_3y.json` (three-year history per gauge). |
| **Outputs** | Partitioned Parquet dataset in `dataset_parquet/gauge_id=<ID>/year=<YYYY>/` plus `dataset_parquet/manifest.json`. |
| **Key steps** |<ul><li>Parse timestamps with fractional seconds (`pd.to_datetime(..., format="ISO8601", errors="coerce")`).</li><li>Coerce `waterHeight` to numeric, synthesize missing boolean columns, deduplicate by timestamp.</li><li>Robust outlier clipping with 0.5/99.5 percentiles and padded span.</li><li>Resample to a 15‑minute cadence, fill gaps up to 2 hours via time interpolation (`MAX_GAP_STEPS = 8`).</li><li>Compute quality flags: flatline detection (rolling abs-diff), jump detection via MAD of rate-of-change, and rolling-median outlier flag.</li><li>Trim leading/trailing NaNs before writing to Parquet (Zstd compression via `pyarrow`).</li></ul> |
| **Notes** | Handles empty gauges gracefully; manifest captures row counts, time span, and missing-rate per gauge for quick QA. |

---

## 2. Feature Engineering (`build_features.py`)

| Aspect | Details |
| --- | --- |
| **Purpose** | Create supervised-learning friendly features for downstream models. |
| **Inputs** | `dataset_parquet/` (output from curation). |
| **Outputs** | `features_parquet/gauge_id=<ID>/year=<YYYY>/`. |
| **Features** |<ul><li>Time-based: hour of day, day-of-week, sinusoidal day-of-year encoding.</li><li>Hydrological history: 1- and 2-step lags, first-difference (`roc`), rolling means/std devs over 8 and 32 windows.</li><li>Quality indicators: carries forward `flag_flat`, `flag_jump`, `flag_outlier`.</li></ul> |
| **Implementation** | Uses `pyarrow.dataset` with `exclude_invalid_files=True` to ignore manifest JSON; groups by `gauge_id` to apply rolling computations; drops rows lacking water heights. |

---

## 3. Forecasting (`forecast_prophet.py`)

| Aspect | Details |
| --- | --- |
| **Purpose** | Train a Prophet model per gauge and generate 12-hour forecasts with confidence bands. |
| **Inputs** | `dataset_parquet/`. |
| **Outputs** | `forecasts_parquet/gauge_id=<ID>/year=<YYYY>/` with columns `ds`, `yhat`, `yhat_lower`, `yhat_upper`. |
| **Workflow** |<ul><li>Discover gauges directly from the Hive directory structure (`dataset_parquet/gauge_id=*`).</li><li>Detect CmdStan backend by querying `cmdstanpy.cmdstan_path()` and updating Prophet’s `CmdStanPyBackend` to use the user-installed toolchain (avoids bundled, outdated CmdStan).</li><li>Drop timezone info before training (Prophet requires naive timestamps).</li><li>Predict 48 steps ahead at 15‑minute intervals and append gauge/year partitions when writing.</li><li>Surface actionable errors if no Stan backend is available (suggest `python -m cmdstanpy.install_cmdstan`).</li></ul> |
| **Dependencies** | `cmdstanpy` ≥ 1.2 and a compiled CmdStan release; `plotly` optional for interactive diagnostics. |

---

## 4. Monitoring Dashboard (`dashboard_app.py`)

| Aspect | Details |
| --- | --- |
| **Purpose** | Streamlit application for real-time situational awareness combining observations, forecasts, and LLM summaries. |
| **Data sources** |<ul><li>Historical series from `dataset_parquet/`.</li><li>Forecasts from `forecasts_parquet/`.</li><li>LLM text from `summaries/weekly_*.txt`.</li></ul> |
| **Highlights** |<ul><li>Gauge selector built from Hive partitions; resilient to manifest files.</li><li>Historical chart with dynamic window (`Days to show`) using `@st.cache_data` caching.</li><li>Heuristic risk score mixing percentile level, short-term rate of change, and anomaly counts.</li><li>Forecast overlay showing central estimate plus lower/upper bands when available.</li><li>Fallback messaging when forecasts or summaries are missing.</li></ul> |
| **Ops tips** | Installing `watchdog` improves auto-reload; Streamlit warns if missing. |

---

## 5. Weekly LLM Summaries (`summaries_strands.py`)

| Aspect | Details |
| --- | --- |
| **Purpose** | Generate concise weekly narratives per gauge using the Strands Agent SDK. |
| **Inputs** | `features_parquet/` (for computed metrics). |
| **Outputs** | Text files in `summaries/weekly_<ISO_DATE>.txt`. |
| **Batch flow** |<ul><li>Enumerate gauges from feature partitions (`gauge_id=*`).</li><li>Slice last week of data per gauge and compute metrics (range, mean, max rise, flat %, jump/outlier counts).</li><li>Build prompt blocks of size 8 gauges, and memoize via `utils.quota_cache` (avoids re-prompting identical weeks).</li><li>Integrate with Strands Agent: instantiate `LiteLLMModel` backed by Gemini Flash Lite using env var `STRANDS_API_KEY`, fall back to offline stub only when key missing. Errors in SDK initialization now raise immediately to expose configuration issues.</li><li>Respect rate limits through `QuotaManager` (RPM/RPD guard) and a 2-second inter-batch sleep.</li></ul> |
| **Environment** | Requires `pip install 'strands-agents[litellm]'` and exporting `STRANDS_API_KEY`. |

---

## 6. Supporting Utilities

- **`utils/quota_cache.py`** (inferred): provides `QuotaManager` and simple key-value caching to enforce API quotas and reuse previous responses.
- **`requirements.txt`** (pinned): ensures consistent environments, notably `cmdstanpy==1.2.4`, `plotly==5.23.0`, `prophet==1.1.5`, `pyarrow==17.0.0`, `streamlit==1.38.0`, plus core scientific stack.

---

## 7. End-to-End Workflow

1. **Curate history**  
   ```bash
   python curate_data.py
   ```
2. **Build features**  
   ```bash
   python build_features.py
   ```
3. **Train forecasts**  
   ```bash
   python -m cmdstanpy.install_cmdstan  # one-time setup  
   python forecast_prophet.py
   ```
4. **Run dashboard**  
   ```bash
   streamlit run dashboard_app.py
   ```
5. **Generate weekly summaries**  
   ```bash
   export STRANDS_API_KEY="..."
   python summaries_strands.py
   ```

Each step produces partitioned Parquet or text artifacts consumed by subsequent modules, allowing the pipeline to be run incrementally.

---

## 8. Known Best Practices & Troubleshooting

- **PyArrow datasets**: `exclude_invalid_files=True` is consistently set to skip manifest JSON files. If new sidecar files are added, keep this flag.
- **Timezone handling**: Prophet requires naive timestamps; all scripts strip UTC offsets before training/predicting, while downstream analytics convert to UTC as needed.
- **CmdStan errors**: If Prophet complains about missing makefiles, rerun `python -m cmdstanpy.install_cmdstan --overwrite` and ensure `cmdstanpy` is installed inside the virtualenv.
- **LLM offline fallback**: Without `STRANDS_API_KEY`, summaries explicitly state `(offline summary ...)`. Once the key and package are set, any initialization failure now throws an exception so teams can react quickly.
- **Streamlit performance**: Install `watchdog` for responsive dev workflows. Use caching decorators already present to minimize Parquet scans.

---

## 9. Component Interactions Diagram

```
Raw JSON (gauge_readings_3y)
        │
        ▼
curate_data.py ──▶ dataset_parquet/ ──┐
                                     ├──▶ build_features.py ──▶ features_parquet/
                                     │                           │
                                     │                           ├──▶ summaries_strands.py ──▶ summaries/
                                     │                           │
                                     │                           └──▶ dashboard_app.py (metrics & risk)
                                     │
                                     └──▶ forecast_prophet.py ──▶ forecasts_parquet/ ─┐
                                                                                      └──▶ dashboard_app.py (forecast overlay)
```

---

For questions or suggested enhancements, drop notes in the repo issues or the team channel. This document should serve as the single reference for how data flows through Floodzilla Light AI and where to hook in new capabilities.

