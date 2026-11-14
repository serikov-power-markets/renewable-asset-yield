"""PV performance model on reanalysis weather, fitted to metered systems whose size and orientation are unknown.

Chain per hour: ERA5 global horizontal irradiance -> Erbs decomposition -> plane-of-array irradiance (Perez) ->
cell temperature (Faiman, ERA5 2 m temperature and 10 m wind) -> PVWatts DC with -0.4%/K -> fixed inverter and
system losses. The effective DC size, tilt and azimuth are fitted by least squares on hours that pass the data
checks; hours the checks remove (outages, snow, meter errors) are reported separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from . import scada as S

KONSTANZ = (47.66, 9.18)
GAMMA = -0.004
SYSTEM_EFF = 0.96 * 0.97      # inverter and DC/AC wiring losses


def load_pv() -> pd.DataFrame:
    """Hourly energy (kWh) per system from cumulative meter readings, indexed by interval start (UTC).

    The reading stamped t is taken at the end of the hour that starts at t (OPSD interval convention), so the
    energy of that hour is reading(t) - reading(t-1). Checked in the data: with the other assignment every system
    produces 0.3 to 1.1 hours before solar noon in both summer and winter (scripts/wp5_pv.py, timing check)."""
    h = pd.read_csv(S.RAW / "opsd_konstanz_pv_cumulative_kwh.csv", parse_dates=["utc_timestamp"])
    h = h.set_index("utc_timestamp")
    h.index = h.index.tz_localize(None)
    e = h.diff()
    e.columns = [c.replace("DE_KN_", "") for c in e.columns]
    return e


def weather(e5: pd.DataFrame) -> pd.DataFrame:
    """ERA5 hourly values for the hour starting at each time stamp (accumulated radiation ends at valid time)."""
    w = pd.DataFrame(index=e5.index - pd.Timedelta("1h"))
    w["ghi"] = (e5.ssrd.to_numpy() / 3600.0).clip(0)
    w["t_air"] = e5.t2m.to_numpy() - 273.15
    w["wind"] = np.hypot(e5.u10, e5.v10).to_numpy()
    return w


def irradiance(w: pd.DataFrame, lat=KONSTANZ[0], lon=KONSTANZ[1]) -> pd.DataFrame:
    mid = w.index + pd.Timedelta("30min")
    sp = pvlib.solarposition.get_solarposition(mid.tz_localize("UTC"), lat, lon)
    sp.index = w.index
    erbs = pvlib.irradiance.erbs(w.ghi, sp.zenith, w.index.day_of_year)
    out = pd.DataFrame({"ghi": w.ghi, "dni": erbs["dni"], "dhi": erbs["dhi"], "zenith": sp.apparent_zenith,
                        "azimuth": sp.azimuth, "t_air": w.t_air, "wind": w.wind}, index=w.index)
    out["dni_extra"] = pvlib.irradiance.get_extra_radiation(w.index.day_of_year)
    out["airmass"] = pvlib.atmosphere.get_relative_airmass(out.zenith).fillna(40)
    return out


def ac_per_kwp(irr: pd.DataFrame, tilt: float, azimuth: float) -> pd.Series:
    """AC energy per kWp of DC (kWh per hour) for one orientation."""
    poa = pvlib.irradiance.get_total_irradiance(tilt, azimuth, irr.zenith, irr.azimuth, irr.dni.fillna(0),
                                                irr.ghi, irr.dhi.fillna(0), dni_extra=irr.dni_extra,
                                                airmass=irr.airmass, model="perez").poa_global.fillna(0).clip(0)
    tc = pvlib.temperature.faiman(poa, irr.t_air, irr.wind)
    dc = poa / 1000.0 * (1 + GAMMA * (tc - 25))
    return (dc * SYSTEM_EFF).clip(lower=0)


def checks(meas: pd.Series, model_kwp: pd.Series) -> pd.DataFrame:
    """Flags on measured hours: meter spikes, outage days, snow-like winter days, plus a 'clean' column."""
    f = pd.DataFrame(index=meas.index)
    q99 = meas[meas > 0].quantile(0.999)
    f["missing"] = meas.isna()
    f["spike"] = meas > 1.5 * q99
    day_m = meas.resample("D").sum()
    day_p = model_kwp.resample("D").sum()
    ratio = day_m / (day_p * (day_m.sum() / day_p.sum()))
    outage_day = (ratio < 0.05) & (day_p > day_p.quantile(0.2))
    snow_day = (ratio < 0.4) & (ratio >= 0.05) & ratio.index.month.isin([11, 12, 1, 2, 3]) & (day_p > day_p.quantile(0.1))
    f["outage_day"] = outage_day.reindex(meas.index.normalize()).to_numpy()
    f["snow_like_day"] = snow_day.reindex(meas.index.normalize()).to_numpy()
    f["clean"] = ~f[["missing", "spike", "outage_day", "snow_like_day"]].any(axis=1)
    return f


def fit_orientation(meas: pd.Series, irr: pd.DataFrame, clean: pd.Series) -> dict:
    """Grid search over tilt and azimuth; DC size by least squares for each orientation (daylight clean hours)."""
    m = clean & (irr.zenith < 85)
    y = meas[m]
    best = None
    for tilt in range(0, 61, 5):
        for az in range(45, 316, 15):
            x = ac_per_kwp(irr[m], tilt, az)
            kwp = float((x * y).sum() / (x * x).sum())
            sse = float(((y - kwp * x) ** 2).sum())
            if best is None or sse < best["sse"]:
                best = {"tilt": tilt, "azimuth": az, "kwp": kwp, "sse": sse}
    return best


def metrics(meas: pd.Series, pred: pd.Series) -> dict:
    m = meas.notna() & pred.notna()
    a, b = meas[m], pred[m]
    day = a.resample("D").sum(), b.resample("D").sum()
    n = a.resample("MS").size()
    ok = (n >= 200) & (a.resample("MS").sum() > 1.0)          # months with enough clean hours and production
    mon = a.resample("MS").sum()[ok], b.resample("MS").sum()[ok]
    return {"r_hourly": float(a.corr(b)), "nmae_hourly_pct_of_mean_daylight": float((a - b).abs()[a > 0].mean() / a[a > 0].mean() * 100),
            "daily_r": float(day[0].corr(day[1])), "monthly_abs_err_pct": float(((mon[1] - mon[0]).abs() / mon[0]).mean() * 100),
            "bias_pct": float((b.sum() - a.sum()) / a.sum() * 100)}
