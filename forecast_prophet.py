# forecast_prophet.py
from pathlib import Path
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from datetime import timedelta
import cmdstanpy

from prophet.models import StanBackendEnum, CmdStanPyBackend
from prophet import Prophet

DATA_DIR = Path("dataset_parquet")
OUT_DIR = Path("forecasts_parquet"); OUT_DIR.mkdir(exist_ok=True)

HORIZON_STEPS = 48   # 48 * 15min = 12 hours (change if desired)
FREQ = "15min"
LOG = logging.getLogger(__name__)

def _prepare_cmdstan():
    try:
        path = cmdstanpy.cmdstan_path()
    except Exception:
        return None
    if not path:
        return None
    cmdstanpy.set_cmdstan_path(path)
    # ensure Prophet doesn't try to reuse bundled (possibly stale) cmdstan
    CmdStanPyBackend.CMDSTAN_VERSION = Path(path).name.replace("cmdstan-", "")
    return path

USER_CMDSTAN_PATH = _prepare_cmdstan()

def _detect_stan_backend():
    attempts = {}
    for candidate in StanBackendEnum:
        try:
            backend_cls = StanBackendEnum.get_backend_class(candidate.name)
            backend_cls()  # instantiate to ensure dependencies are available
            return candidate.name
        except Exception as exc:
            attempts[candidate.name] = exc
            LOG.debug("Prophet backend %s unavailable: %s", candidate.name, exc)
            continue
    raise RuntimeError(
        "No functional Prophet Stan backend available. Install cmdstanpy "
        "(run `pip install cmdstanpy` and `python -m cmdstanpy.install_cmdstan`) "
        "or pystan. Attempted backends: "
        + ", ".join(f"{name}: {err}" for name, err in attempts.items())
    )

try:
    DEFAULT_STAN_BACKEND = _detect_stan_backend()
except RuntimeError as backend_err:
    DEFAULT_STAN_BACKEND = None
    BACKEND_ERROR = backend_err
else:
    BACKEND_ERROR = None

def gauges():
    if not DATA_DIR.exists():
        return []
    gauges = set()
    for path in DATA_DIR.glob("gauge_id=*"):
        if path.is_dir():
            parts = path.name.split("=", 1)
            if len(parts) == 2 and parts[1]:
                gauges.add(parts[1])
    return sorted(gauges)

def train_one(gid: str):
    dataset = ds.dataset(
        str(DATA_DIR),
        format="parquet",
        partitioning="hive",
        exclude_invalid_files=True,
    )
    if BACKEND_ERROR is not None:
        raise RuntimeError(
            f"Prophet backend unavailable while training gauge {gid}: {BACKEND_ERROR}"
        ) from BACKEND_ERROR
    df = dataset.to_table(filter=(ds.field("gauge_id")==gid)).to_pandas().sort_values("ds")
    if df.empty: 
        return None
    d = df[["ds","waterHeight"]].rename(columns={"ds":"ds","waterHeight":"y"})
    d["ds"] = pd.to_datetime(d["ds"], utc=True).dt.tz_localize(None)
    m = Prophet(stan_backend=DEFAULT_STAN_BACKEND)
    try:
        m.fit(d)
    except Exception as exc:
        raise RuntimeError(
            "Prophet failed to fit the model. If using cmdstanpy, ensure CmdStan is "
            "installed by running `python -m cmdstanpy.install_cmdstan`."
        ) from exc
    future = pd.date_range(d["ds"].max()+pd.Timedelta(FREQ),
                           periods=HORIZON_STEPS, freq=FREQ, tz="UTC")
    future = pd.DataFrame({"ds": future})
    future["ds"] = future["ds"].dt.tz_localize(None)
    fc = m.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]]
    fc["gauge_id"] = gid
    return fc

def main():
    if BACKEND_ERROR is not None:
        raise RuntimeError(f"Prophet backend unavailable: {BACKEND_ERROR}") from BACKEND_ERROR
    rows = []
    for gid in gauges():
        print(f"[prophet] {gid}")
        fc = train_one(gid)
        if fc is None: 
            continue
        rows.append(fc)
        # write per gauge partition
        fc["year"] = fc["ds"].dt.year
        table = pa.Table.from_pandas(fc)
        pq.write_to_dataset(table, root_path=str(OUT_DIR), partition_cols=["gauge_id","year"], compression="zstd")
    print("✅ wrote", OUT_DIR)

if __name__ == "__main__":
    main()
