# build_features.py
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

DATA_DIR = Path("dataset_parquet")
OUT_DIR = Path("features_parquet"); OUT_DIR.mkdir(exist_ok=True)

def load_all():
    dataset = ds.dataset(
        str(DATA_DIR),
        format="parquet",
        partitioning="hive",
        exclude_invalid_files=True,
    )
    return dataset.to_table().to_pandas()

def featureize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["gauge_id","ds"]).reset_index(drop=True)
    dt = pd.to_datetime(df["ds"], utc=True)
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["doy_sin"] = np.sin(2*np.pi*dt.dt.dayofyear/365.25)
    df["doy_cos"] = np.cos(2*np.pi*dt.dt.dayofyear/365.25)

    def fe(g):
        g = g.sort_values("ds")
        x = g["waterHeight"]
        g["lag_1"] = x.shift(1)
        g["lag_2"] = x.shift(2)
        g["roc"] = x.diff()
        g["rmean_8"] = x.rolling(8).mean()
        g["rstd_8"] = x.rolling(8).std()
        g["rmean_32"] = x.rolling(32).mean()
        g["rstd_32"] = x.rolling(32).std()
        return g

    df = df.groupby("gauge_id", group_keys=False).apply(fe)
    keep = ["ds","gauge_id","waterHeight","flag_flat","flag_jump","flag_outlier",
            "hour","dow","doy_sin","doy_cos","lag_1","lag_2","roc","rmean_8","rstd_8","rmean_32","rstd_32"]
    return df[keep].dropna(subset=["waterHeight"])

def main():
    df = load_all()
    feats = featureize(df)
    feats["year"] = pd.to_datetime(feats["ds"]).dt.year
    table = pa.Table.from_pandas(feats)
    pq.write_to_dataset(table, root_path=str(OUT_DIR), partition_cols=["gauge_id","year"], compression="zstd")
    print("✅ wrote", OUT_DIR)

if __name__ == "__main__":
    main()
