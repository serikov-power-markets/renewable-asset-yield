"""Wake losses: measured from SCADA by wind direction, modelled with PyWake on the real layout.

Both use the same definition so they can be compared: farm efficiency = sum of the six turbines' power divided by
six times the power of the least waked turbine (the reference) in that record.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTOR = 30
WS_RANGE = (5.0, 10.0)


def _circmean(deg: pd.DataFrame) -> pd.Series:
    r = np.radians(deg)
    return (np.degrees(np.arctan2(np.sin(r).mean(axis=1), np.cos(r).mean(axis=1))) % 360)


def observed_efficiency(clean: pd.DataFrame, ref_turbine: pd.Series | None = None) -> pd.DataFrame:
    """Farm efficiency by sector. With `ref_turbine` (sector -> turbine) the reference is that fixed upwind
    turbine; without it, the record-wise maximum (biased: the largest of six noisy values exceeds the free stream)."""
    p = clean.pivot_table(index="time", columns="turbine", values="power_kw")
    w = clean.pivot_table(index="time", columns="turbine", values="ws_norm")
    wd = clean.pivot_table(index="time", columns="turbine", values="wd")
    ok = p.notna().all(axis=1) & w.notna().all(axis=1) & wd.notna().all(axis=1)
    p, w, wd = p[ok], w[ok], wd[ok]
    wdir = _circmean(wd)
    sec = ((wdir + SECTOR / 2) // SECTOR * SECTOR) % 360
    if ref_turbine is None:
        ref_p, ref_ws = p.max(axis=1), w.mean(axis=1)
    else:
        cols = sec.map(ref_turbine)
        idx = p.columns.get_indexer(cols)
        ref_p = pd.Series(p.to_numpy()[np.arange(len(p)), idx], index=p.index)
        ref_ws = pd.Series(w.to_numpy()[np.arange(len(w)), idx], index=w.index)
    keep = (ref_ws >= WS_RANGE[0]) & (ref_ws <= WS_RANGE[1])
    g = sec[keep]
    farm = p[keep].sum(axis=1).groupby(g).mean()
    ref = ref_p[keep].groupby(g).mean()
    out = pd.DataFrame({"eff_observed": farm / (6 * ref), "n_records": g.groupby(g).size()})
    out.index.name = "sector"
    return out


def layout_xy(turbines: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lat0, lon0 = turbines.Latitude.mean(), turbines.Longitude.mean()
    x = (turbines.Longitude - lon0) * 111320 * np.cos(np.radians(lat0))
    y = (turbines.Latitude - lat0) * 110540
    return x.to_numpy(), y.to_numpy()


def make_turbine(power_fn, diameter=92.0, hub=78.5, rated_kw=2050.0):
    from py_wake.wind_turbines import WindTurbine
    from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
    from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

    gen = GenericWindTurbine("generic", diameter, hub, rated_kw)
    ws = np.arange(0, 30.01, 0.5)
    ct = gen.ct(ws)                         # generic thrust curve for the size class (no MM92 thrust data public)
    return WindTurbine("MM92-fitted", diameter, hub, PowerCtTabular(ws, power_fn(ws), "kW", ct))


def pywake_efficiency(turbines: pd.DataFrame, power_fn, ti: float = 0.08, alpha: float = 0.2) -> tuple[pd.DataFrame, pd.Series]:
    """PyWake farm efficiency by sector (reference = the turbine with the highest mean power in that sector) and
    the reference turbine per sector."""
    from py_wake import BastankhahGaussian
    from py_wake.site import UniformSite
    from py_wake.site.shear import PowerShear

    x, y = layout_xy(turbines)
    wt = make_turbine(power_fn)
    site = UniformSite(p_wd=[1.0], ti=ti, shear=PowerShear(h_ref=78.5, alpha=alpha))
    model = BastankhahGaussian(site, wt)
    wds = np.arange(0, 360, 2.0)
    wss = np.arange(WS_RANGE[0], WS_RANGE[1] + 0.01, 0.5)
    hub = turbines["Hub Height (m)"].to_numpy(float)                # two turbines stand 10 m lower (68.5 m)
    p = model(x, y, h=hub, wd=wds, ws=wss).Power.values.mean(axis=2)  # (turbine, wd), mean over wind speeds
    sec = ((wds + SECTOR / 2) // SECTOR * SECTOR) % 360
    per = pd.DataFrame(p.T, index=sec, columns=turbines.index).groupby(level=0).mean()
    ref = per.idxmax(axis=1)
    eff = per.sum(axis=1) / (6 * per.max(axis=1))   # includes the lower hubs' shear deficit, as the SCADA ratio does
    out = pd.DataFrame({"eff_pywake": eff, "ref_turbine": ref})
    out.index.name = "sector"
    return out, ref


def wake_summary(clean: pd.DataFrame, sec: pd.DataFrame) -> dict:
    w = sec.n_records / sec.n_records.sum()
    obs = float((sec.eff_observed * w).sum())
    mod = float((sec.eff_pywake * w).sum())
    worst = sec.eff_observed.idxmin()
    return {"farm_efficiency_observed": obs, "farm_efficiency_pywake": mod,
            "wake_loss_observed_pct": 100 * (1 - obs), "wake_loss_pywake_pct": 100 * (1 - mod),
            "worst_sector_deg": float(worst), "worst_sector_eff_observed": float(sec.eff_observed[worst]),
            "worst_sector_eff_pywake": float(sec.eff_pywake[worst]),
            "sector_corr": float(sec[["eff_observed", "eff_pywake"]].corr().iloc[0, 1])}
