"""Measured production: load SCADA, label stops from the event log, filter by data rules, build power curves.

The aim is to separate what the wind allowed (production potential) from what the turbine delivered: stops,
derating and sensor faults are removed before a power curve is fitted, and the data-driven filter is checked
against the operator's own event log and lost-production estimate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
RATED_KW = 2050.0
ROTOR_D = 92.0
RHO0 = 1.225


def turbines() -> pd.DataFrame:
    t = pd.read_csv(RAW / "kelmarsh_turbines.csv", encoding="utf-8-sig")
    t["turbine"] = t["Title"].str.extract(r"(\d+)$").astype(int)
    return t.set_index("turbine")


def load_scada() -> pd.DataFrame:
    return pd.read_parquet(RAW / "kelmarsh_scada_2017_10min.parquet")


def _parse_duration(s: pd.Series) -> pd.Series:
    parts = s.str.split(":", expand=True).astype(float)
    return pd.to_timedelta(parts[0] * 3600 + parts[1] * 60 + parts[2], unit="s")


def stop_mask(scada: pd.DataFrame) -> pd.Series:
    """True for 10-minute records that overlap a 'Stop' event in the operator's event log (any category except
    low-wind stops, which are part of normal operation)."""
    st = pd.read_csv(RAW / "kelmarsh_status_2017.csv")
    st = st[(st.Status == "Stop") & (st["IEC category"] != "Out of Environmental Specification")].copy()
    st["start"] = pd.to_datetime(st["Timestamp start"]).dt.floor("10min")
    st["end"] = pd.to_datetime(st["Timestamp end"])
    mask = pd.Series(False, index=scada.index)
    for wt, g in st.groupby("turbine"):
        rows = scada.turbine == wt
        tt = scada.loc[rows, "time"]
        hit = np.zeros(len(tt), dtype=bool)
        tv = tt.to_numpy()
        for a, b in zip(g.start.to_numpy(), g.end.to_numpy()):
            hit |= (tv >= a) & (tv < b)
        mask.loc[rows] = hit
    return mask


def density(t_c: pd.Series | np.ndarray, p_pa: float | np.ndarray = 101325.0) -> np.ndarray:
    return np.asarray(p_pa) / (287.05 * (np.asarray(t_c) + 273.15))


def rule_filter(d: pd.DataFrame) -> pd.DataFrame:
    """Data-driven flags (no event log): each rule is one column, `bad` is their union."""
    f = pd.DataFrame(index=d.index)
    f["missing"] = d[["ws", "power_kw"]].isna().any(axis=1)
    f["stopped"] = (d.ws > 4.5) & (d.power_kw < 0.02 * RATED_KW)                 # wind enough, no power
    # derated: well below the turbine's own median curve while the blades are pitched (power limited on purpose)
    ref = d.groupby([d.turbine, (d.ws * 2).round() / 2]).power_kw.transform("median")
    f["derated"] = (d.ws > 6) & (d.power_kw < 0.7 * ref) & (d.pitch > 3)
    # frozen anemometer: identical wind speed over 3 hours
    same = d.groupby("turbine").ws.transform(lambda s: s.diff().abs().rolling(18).max() < 1e-6)
    f["sensor_frozen"] = same.fillna(False)
    f["bad"] = f.any(axis=1)
    return f


def compare_with_log(flags: pd.DataFrame, log_stop: pd.Series, lost_kwh: pd.Series) -> dict:
    """Agreement between the data-driven filter and the operator's stop log, weighted by lost energy."""
    bad, stop = flags.bad, log_stop
    tp = (bad & stop).sum()
    lost = lost_kwh.fillna(0)
    return {"records_flagged_by_rules": int(bad.sum()), "records_in_log_stops": int(stop.sum()),
            "recall_of_log_stops": float(tp / max(stop.sum(), 1)),
            "precision_vs_log": float(tp / max(bad.sum(), 1)),
            "operator_lost_energy_caught_share": float(lost[bad].sum() / max(lost.sum(), 1e-9)),
            "operator_lost_mwh": float(lost.sum() / 1000)}


def power_curve(d: pd.DataFrame, ws_col: str = "ws_norm", bin_w: float = 0.5, min_n: int = 10) -> pd.DataFrame:
    """IEC 61400-12-1 style method of bins on clean data (bin centres, mean power, count, standard error)."""
    b = (d[ws_col] / bin_w).round() * bin_w
    g = d.groupby(b).power_kw
    pc = pd.DataFrame({"power_kw": g.mean(), "n": g.size(), "se": g.std() / np.sqrt(g.size())})
    pc = pc[pc.n >= min_n]
    pc.index.name = "ws"
    return pc


def normalise_ws(ws: pd.Series, rho: np.ndarray) -> pd.Series:
    """IEC density normalisation of wind speed for pitch-regulated turbines."""
    return ws * (rho / RHO0) ** (1 / 3)


def curve_function(pc: pd.DataFrame):
    """Monotone interpolating function of a binned curve, 0 below the first bin, rated above, cut-out at 25 m/s."""
    x = pc.index.to_numpy(float)
    y = np.maximum.accumulate(np.clip(pc.power_kw.to_numpy(float), 0, None))
    x = np.r_[0.0, x, 25.0, 25.01, 40.0]
    y = np.r_[0.0, y, y[-1], 0.0, 0.0]

    def f(ws):
        return np.interp(ws, x, y)

    return f
