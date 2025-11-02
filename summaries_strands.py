# summaries_strands.py
import os, time, json, logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from utils.quota_cache import QuotaManager, cache_key, cache_get, cache_put

try:
    from strands import Agent as StrandsAgent
    from strands.models.litellm import LiteLLMModel
except Exception:  # pragma: no cover - optional dependency
    StrandsAgent = None  # type: ignore
    LiteLLMModel = None  # type: ignore

DATA_DIR = Path("features_parquet")
OUT_DIR = Path("summaries"); OUT_DIR.mkdir(exist_ok=True)

MAX_RPM = 30
MAX_RPD = 200

class StrandsAgentClient:
    """
    Adapter around the strands Agent SDK using the LiteLLM-backed Gemini model.
    Falls back to offline summaries when the SDK or API key is unavailable.
    """

    def __init__(self, api_key: str, model: str = "gemini/gemini-2.0-flash-lite"):
        self.api_key = api_key
        self.model = model
        self._client = None
        self._init_error: Exception | None = None
        log = logging.getLogger(__name__)

        if StrandsAgent is None or LiteLLMModel is None:
            self._init_error = RuntimeError(
                "strands SDK not installed. Install with `pip install strands`."
            )
            log.warning("%s", self._init_error)
            return

        if not api_key:
            log.warning("STRANDS_API_KEY not set; using offline summary fallback.")
            return

        try:
            os.environ.setdefault("STRANDS_API_KEY", api_key)
            os.environ.setdefault("LITELLM_API_KEY", api_key)
            lite_model = LiteLLMModel(
                model_id=model,
                params={
                    "max_tokens": 4096,
                    "temperature": 0.2,
                },
            )
            self._client = StrandsAgent(model=lite_model)
        except Exception as exc:  # pragma: no cover - SDK specific
            self._init_error = exc
            log.warning("Strands client unavailable: %s", exc)

    def generate(self, prompt: str) -> str:
        if self._client is None:
            if self.api_key and self._init_error is not None:
                raise RuntimeError(
                    "Strands client failed to initialize, cannot generate summary."
                    f" Original error: {self._init_error}"
                ) from self._init_error
            # fallback so script still runs offline
            return "(offline summary) " + prompt[:200]

        try:
            response = self._client(prompt)
            print(response)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Strands agent call failed: {exc}") from exc

        text = extract_agent_text(response)
        if not text.strip():
            raise RuntimeError("Strands agent returned empty response.")
        return text

def extract_agent_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    for attr in ("output", "content", "text"):
        if hasattr(response, attr):
            value = getattr(response, attr)
            if isinstance(value, str):
                return value
            if isinstance(value, (list, tuple)):
                joined = "\n".join(str(item) for item in value)
                if joined.strip():
                    return joined
    if isinstance(response, dict):
        if "output" in response and isinstance(response["output"], str):
            return response["output"]
        return json.dumps(response, ensure_ascii=False)
    if isinstance(response, (list, tuple)):
        joined = "\n".join(str(item) for item in response)
        if joined.strip():
            return joined
    return str(response)

client = StrandsAgentClient(api_key=os.getenv("STRANDS_API_KEY", ""), model="gemini/gemini-2.0-flash-lite")
quota = QuotaManager(MAX_RPM, MAX_RPD)

def load_slice(gauge_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    dataset = ds.dataset(
        str(DATA_DIR),
        format="parquet",
        partitioning="hive",
        exclude_invalid_files=True,
    )
    df = dataset.to_table(filter=(ds.field("gauge_id")==gauge_id)).to_pandas()
    t = pd.to_datetime(df["ds"], utc=True)
    return df[(t>=start)&(t<end)].copy()

def metrics(df: pd.DataFrame) -> dict:
    if df.empty: return {"rows":0}
    m = {}
    x = df["waterHeight"]
    m["rows"] = int(df.shape[0])
    m["min"] = float(np.nanmin(x)); m["max"] = float(np.nanmax(x)); m["mean"]=float(np.nanmean(x))
    m["max_rise_step"] = float(np.nanmax(df["roc"])) if "roc" in df else None
    m["flat_pct"] = float(100*df["flag_flat"].mean()) if "flag_flat" in df else 0.0
    m["jump_cnt"] = int(df["flag_jump"].sum()) if "flag_jump" in df else 0
    m["outlier_cnt"] = int(df["flag_outlier"].sum()) if "flag_outlier" in df else 0
    return m

def build_prompt(payload):
    lines = ["Summarize weekly river conditions, two detailed line per gauge; prioritize flood risk signals. Along with weekly metric numbers."]
    for it in payload:
        g = it["gauge_id"]; m = it["metrics"]
        if m["rows"]==0:
            lines.append(f"- {g}: no data.")
        else:
            lines.append(f"- {g}: range {m['min']:.2f}-{m['max']:.2f} (avg {m['mean']:.2f}); "
                         f"max step rise {m.get('max_rise_step',0):.2f}; "
                         f"flat {m.get('flat_pct',0):.1f}% | jumps {m.get('jump_cnt',0)} | outliers {m.get('outlier_cnt',0)}. "
                         "Give one actionable phrase if risk elevated.")
    lines.append("Keep each line < 25 words.")
    return "\n".join(lines)

def main():
    # find gauges
    if not DATA_DIR.exists():
        print("No gauges found; run build_features.py first.")
        return
    gauges = sorted({
        path.name.split("=", 1)[1]
        for path in DATA_DIR.glob("gauge_id=*")
        if path.is_dir() and "=" in path.name
    })
    if not gauges:
        print("No gauges found; run build_features.py first."); return

    now = datetime.now(timezone.utc)
    week_end = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) - timedelta(days=now.weekday())
    week_start = week_end - timedelta(days=7)

    BATCH = 8
    all_text = []
    for i in range(0, len(gauges), BATCH):
        group = gauges[i:i+BATCH]
        payload = []
        for gid in group:
            df = load_slice(gid, week_start, week_end)
            payload.append({"gauge_id": gid, "metrics": metrics(df)})
        prompt = build_prompt(payload)
        key = cache_key({"start": week_start.isoformat(), "group": group, "prompt": prompt})
        cached = cache_get(key)
        print(cached)
        if cached:
            text = cached["text"]
        else:
            ok, why = quota.can_call()
            print(ok)
            print(why)
            if not ok:
                text = f"(quota) {why}: " + prompt[:200]
            else:
                text = client.generate(prompt)
                quota.bump()
                cache_put(key, {"text": text})
        all_text.append(text)
        time.sleep(2)

    out = "\n".join(all_text)
    fn = OUT_DIR / f"weekly_{week_start.date().isoformat()}.txt"
    fn.write_text(out)
    print("📝", fn)

if __name__ == "__main__":
    main()
