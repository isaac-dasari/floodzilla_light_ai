# curate_data.py
import json
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RAW_DIR = Path("gauge_readings_3y")
OUT_DIR = Path("dataset_parquet")
OUT_DIR.mkdir(exist_ok=True)

RESAMPLE = "15min"
MAX_GAP_STEPS = 8  # ~2 hours for 15-min cadence

def load_one(path: Path) -> pd.DataFrame:
    obj = json.loads(path.read_text())
    rows = obj.get("readings", [])
    if not rows:
        return pd.DataFrame(columns=["ds","waterHeight","id","isDeleted","isMissing"]).set_index("ds")
    df = pd.DataFrame(rows)
    df["ds"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601", errors="coerce")
    df["waterHeight"] = pd.to_numeric(df["waterHeight"], errors="coerce")
    for c in ("isDeleted","isMissing"):
        if c not in df: df[c] = False
    df = df[["ds","waterHeight","id","isDeleted","isMissing"]].dropna(subset=["ds"]).sort_values("ds")
    df = df.drop_duplicates(subset=["ds"], keep="last").set_index("ds")
    return df

def curate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.assign(flag_flat=False, flag_jump=False, flag_outlier=False)
    df = df[~df["isDeleted"].fillna(False)]

    # robust clip
    if df["waterHeight"].notna().sum() > 10:
        lo, hi = np.nanpercentile(df["waterHeight"], [0.5, 99.5])
        span = max(1e-3, hi - lo)
        df["waterHeight"] = df["waterHeight"].clip(lo - 0.25*span, hi + 0.25*span)

    # resample
    dfu = df[["waterHeight"]].resample(RESAMPLE).mean()
    w = dfu["waterHeight"]
    w_interp = w.interpolate(method="time", limit=MAX_GAP_STEPS, limit_direction="both")
    dfu["waterHeight"] = w.where(w.notna(), w_interp)

    # flags
    diff = dfu["waterHeight"].diff().abs()
    flag_flat = diff.rolling(8, min_periods=1).sum() < 1e-6
    roc = dfu["waterHeight"].diff()  # per step
    med = dfu["waterHeight"].rolling(25, center=True, min_periods=7).median()
    mad = (dfu["waterHeight"] - med).abs().rolling(25, center=True, min_periods=7).median()
    flag_outlier = (dfu["waterHeight"] - med).abs() > 6*(mad.replace(0, np.nan))
    flag_jump = roc.abs() > (np.nanmedian(np.abs(roc)) + 6*np.nanmedian(np.abs(roc - np.nanmedian(roc))))

    out = pd.DataFrame({
        "waterHeight": dfu["waterHeight"],
        "flag_flat": flag_flat.fillna(False),
        "flag_jump": flag_jump.fillna(False),
        "flag_outlier": flag_outlier.fillna(False),
    })
    # trim leading/trailing NaNs
    start = out["waterHeight"].first_valid_index()
    end = out["waterHeight"].last_valid_index()
    if start and end:
        out = out.loc[start:end]
    return out

def write_parquet(gauge_id: str, df: pd.DataFrame):
    if df.empty: return
    df = df.reset_index().rename(columns={"index":"ds"})
    df["gauge_id"] = gauge_id
    df["year"] = df["ds"].dt.year
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(table, root_path=str(OUT_DIR), partition_cols=["gauge_id","year"], compression="zstd")

def main():
    manifest = []
    files = sorted(RAW_DIR.glob("gauge_*_readings_3y.json"))
    for p in files:
        gauge_id = p.stem.split("_")[1]
        print(f"[curate] {gauge_id}")
        raw = load_one(p)
        cur = curate(raw)
        write_parquet(gauge_id, cur)
        stats = {
            "gauge_id": gauge_id,
            "rows": int(cur.shape[0]),
            "start": cur.index.min().isoformat() if not cur.empty else None,
            "end": cur.index.max().isoformat() if not cur.empty else None,
            "missing_rate": float(np.mean(pd.isna(cur["waterHeight"]))) if not cur.empty else 1.0,
        }
        manifest.append(stats)
    pd.DataFrame(manifest).to_json(OUT_DIR / "manifest.json", orient="records", indent=2)
    print("✅ wrote", OUT_DIR)

if __name__ == "__main__":
    main()
