"""Stored production estimates with the information needed to reproduce them.

Every estimate is written as JSON with the hash of each input file, the code version (git commit), the time of the
run and the parameters, so that a number used in pricing can be traced to the data and code that produced it.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import scada as S

STORE = S.ROOT / "results" / "estimates"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_version() -> str:
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=S.ROOT, capture_output=True, text=True)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def save_estimate(asset: str, estimate: dict, inputs: list[Path], params: dict | None = None) -> Path:
    STORE.mkdir(parents=True, exist_ok=True)
    rec = {"asset": asset, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "code_version": code_version(), "inputs": {p.name: sha256(p) for p in inputs},
           "params": params or {}, "estimate": estimate}
    out = STORE / f"{asset}.json"
    out.write_text(json.dumps(rec, indent=1))
    return out


def verify(asset: str) -> dict:
    """Recompute input hashes and report whether the stored estimate still matches the data on disk."""
    rec = json.loads((STORE / f"{asset}.json").read_text())
    now = {name: sha256(S.RAW / name) for name in rec["inputs"]}
    return {"asset": asset, "inputs_unchanged": now == rec["inputs"], "code_version_stored": rec["code_version"],
            "code_version_now": code_version()}
