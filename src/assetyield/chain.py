"""From reanalysis weather to farm production: a physical chain with an optional statistical correction.

Physical chain per hour: ERA5 100 m wind -> hub height with the hour's own shear (from 10 m and 100 m) ->
density normalisation (2 m temperature, surface pressure) -> one speed-scaling factor k (ERA5 bias at the site,
calibrated on measured months) -> each turbine's fitted power curve at its own hub height -> PyWake wake model for
the hour's wind direction. The statistical layer fits gradient boosting to the remaining hourly error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.ensemble import HistGradientBoostingRegressor

from . import scada as S, wakes as W


def load_era5(site: str) -> pd.DataFrame:
    e = pd.read_csv(S.RAW / f"era5_{site}_1991_2025.csv.gz", parse_dates=["valid_time"]).set_index("valid_time")
    e.index.name = "time"
    w10, w100 = np.hypot(e.u10, e.v10), np.hypot(e.u100, e.v100)
    e["ws100"] = w100
    e["alpha"] = np.clip(np.log(np.maximum(w100, 0.5) / np.maximum(w10, 0.3)) / np.log(10), 0.0, 0.6)
    e["wd100"] = (np.degrees(np.arctan2(-e.u100, -e.v100)) % 360)      # direction the wind comes from
    e["rho"] = e.sp / (287.05 * e.t2m)
    return e


def measured_farm_hourly(d: pd.DataFrame) -> pd.DataFrame:
    """Hourly farm energy: metered, and production potential = metered + operator's lost production to downtime."""
    d = d.copy()
    d["e_kwh"] = d.power_kw.clip(lower=0) / 6
    d["pot_kwh"] = d.e_kwh + d.lost_downtime_kwh.fillna(0)
    h = d.set_index("time").groupby([pd.Grouper(freq="h"), "turbine"])[["e_kwh", "pot_kwh"]].sum(min_count=6)
    farm = h.groupby(level=0).sum(min_count=6)          # all six turbines with six records each
    n = h.groupby(level=0).size()
    farm = farm[n == 6].dropna()
    return farm / 1000.0                                  # MWh per hour = MW mean


class FarmModel:
    def __init__(self, turbines: pd.DataFrame, power_fn, ti: float = 0.08):
        from py_wake import BastankhahGaussian
        from py_wake.site import UniformSite
        from py_wake.site.shear import PowerShear

        self.t = turbines
        self.x, self.y = W.layout_xy(turbines)
        self.hub = turbines["Hub Height (m)"].to_numpy(float)
        self.wt = W.make_turbine(power_fn)
        self.ti = ti
        self._site = lambda a: UniformSite(p_wd=[1.0], ti=ti, shear=PowerShear(h_ref=100.0, alpha=a))
        self._model = BastankhahGaussian
        self.k = 1.0
        self.gbm = None

    def physical(self, e: pd.DataFrame, k: float | None = None, wakes: bool = True) -> pd.Series:
        """Farm power (MW) per hour; shear applied per alpha bin to keep PyWake vectorised.
        wakes=False gives the free-stream production of the same turbines (for the wake loss)."""
        k = self.k if k is None else k
        ws = (e.ws100 * k * (e.rho / S.RHO0) ** (1 / 3)).clip(0, 35).to_numpy()
        wd = e.wd100.to_numpy()
        alpha = e.alpha.to_numpy()
        if not wakes:
            hub_ws = ws[:, None] * (self.hub[None, :] / 100.0) ** alpha[:, None]
            return pd.Series(self.wt.power(hub_ws).sum(axis=1) / 1e6, index=e.index, name="p_free")
        a_bin = np.clip(np.round(alpha * 20) / 20, 0.0, 0.5)
        out = np.zeros(len(e))
        for a in np.unique(a_bin):
            m = a_bin == a
            sim = self._model(self._site(float(a)), self.wt)(self.x, self.y, h=self.hub, wd=wd[m], ws=ws[m], time=True)
            out[m] = sim.Power.values.sum(axis=0) / 1e6
        return pd.Series(out, index=e.index, name="p_phys")

    def calibrate_k(self, e: pd.DataFrame, energy: pd.Series) -> float:
        """Speed factor that makes modelled energy equal measured energy on the given hours."""
        target = energy.sum()
        f = lambda k: self.physical(e, k).sum() - target
        self.k = brentq(f, 0.6, 1.6, xtol=1e-4)
        return self.k

    @staticmethod
    def features(e: pd.DataFrame, p_phys: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({"p_phys": p_phys, "ws100": e.ws100, "alpha": e.alpha, "sin_wd": np.sin(np.radians(e.wd100)),
                             "cos_wd": np.cos(np.radians(e.wd100)), "t2m": e.t2m, "hour": e.index.hour,
                             "doy": e.index.day_of_year}, index=e.index)

    def fit_correction(self, e: pd.DataFrame, energy: pd.Series, p_phys: pd.Series):
        X = self.features(e, p_phys)
        self.gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                 min_samples_leaf=40, l2_regularization=1.0, random_state=0)
        self.gbm.fit(X, energy - p_phys)

    def corrected(self, e: pd.DataFrame, p_phys: pd.Series) -> pd.Series:
        cap = 6 * S.RATED_KW / 1000
        return (p_phys + self.gbm.predict(self.features(e, p_phys))).clip(0, cap)


def metrics(obs: pd.Series, pred: pd.Series, capacity_mw: float) -> dict:
    err = pred - obs
    m_obs, m_pred = obs.resample("MS").sum(), pred.resample("MS").sum()
    return {"nmae_pct_of_capacity": float(err.abs().mean() / capacity_mw * 100),
            "nrmse_pct_of_capacity": float(np.sqrt((err**2).mean()) / capacity_mw * 100),
            "bias_pct": float(err.sum() / obs.sum() * 100), "r_hourly": float(obs.corr(pred)),
            "monthly_abs_err_pct": float(((m_pred - m_obs).abs() / m_obs).mean() * 100)}
