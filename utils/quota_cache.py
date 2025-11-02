# utils/quota_cache.py
import json, time, hashlib
from pathlib import Path
from datetime import datetime, timezone

CACHE_DIR = Path("llm_cache"); CACHE_DIR.mkdir(exist_ok=True)
COUNTERS = Path("llm_counters.json")

class QuotaManager:
    def __init__(self, max_rpm=30, max_rpd=200):
        self.max_rpm = max_rpm
        self.max_rpd = max_rpd

    def _load(self):
        if COUNTERS.exists():
            d = json.loads(COUNTERS.read_text())
        else:
            d = {"day": datetime.now(timezone.utc).date().isoformat(), "rpd": 0, "rpm_ts": []}
        # reset if new day
        today = datetime.now(timezone.utc).date().isoformat()
        if d["day"] != today:
            d = {"day": today, "rpd": 0, "rpm_ts": []}
        # trim rpm window
        cutoff = time.time() - 60
        d["rpm_ts"] = [t for t in d["rpm_ts"] if t > cutoff]
        return d

    def can_call(self):
        d = self._load()
        if d["rpd"] >= self.max_rpd:
            return False, "RPD limit reached"
        if len(d["rpm_ts"]) >= self.max_rpm:
            return False, "RPM limit reached"
        return True, ""

    def bump(self):
        d = self._load()
        d["rpd"] += 1
        d["rpm_ts"].append(time.time())
        COUNTERS.write_text(json.dumps(d))

def cache_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def cache_get(key: str):
    p = CACHE_DIR / f"{key}.json"
    return json.loads(p.read_text()) if p.exists() else None

def cache_put(key: str, value: dict):
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(value, indent=2))