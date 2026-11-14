"""WP5: three metered PV systems in Konstanz (2015-2018) against a pvlib chain on ERA5; long-term P50/P90.

For each system: data checks (meter spikes, outage days, snow-like winter days), fitted DC size, tilt and azimuth,
verification on clean hours (first year fitted, rest held out), loss drivers, and 1991-2025 specific yield.
Outputs: results/wp5_*.csv/json, figures/wp5_pv.png
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from assetyield import chain as C, pv as P, registry, scada as S

RES, FIG = S.ROOT / "results", S.ROOT / "figures"
SYSTEMS = ["residential4_pv", "residential6_pv", "industrial3_pv_roof", "residential3_pv"]
Z90 = 1.2816


def main():
    pv = P.load_pv()
    e5 = C.load_era5("konstanz")
    w = P.weather(e5)
    irr_all = P.irradiance(w)
    rows, lt_rows, curves = [], {}, {}
    timing = []
    for name in SYSTEMS:          # timing check: daily production centroid vs the irradiance centroid (UTC hours)
        m = pv[name].dropna()
        d = pd.DataFrame({"e": m, "g": irr_all.ghi.reindex(m.index)}).dropna()
        h = d.index.hour + 0.5
        timing.append({"system": name, "centroid_measured_h": float((d.e * h).sum() / d.e.sum()),
                       "centroid_ghi_h": float((d.g * h).sum() / d.g.sum())})
    pd.DataFrame(timing).to_csv(RES / "wp5_timing_check.csv", index=False)
    print(pd.DataFrame(timing).round(2).to_string(), flush=True)
    for name in SYSTEMS:
        meas = pv[name].dropna()
        meas = meas[meas.index.isin(irr_all.index)]
        irr = irr_all.loc[meas.index]
        first = meas.index[0] + pd.Timedelta(days=365)
        guess = P.ac_per_kwp(irr, 30, 180)
        flags = P.checks(meas, guess)
        train = flags.clean & (meas.index < first)
        fit = P.fit_orientation(meas, irr, train)
        model = fit["kwp"] * P.ac_per_kwp(irr, fit["tilt"], fit["azimuth"])
        flags = P.checks(meas, model)            # recheck with the fitted model
        test = flags.clean & (meas.index >= first)
        met = P.metrics(meas[test], model[test])
        lost = {k: float((model[flags[k]].sum() - meas[flags[k]].clip(lower=0).sum()) / model.sum() * 100)
                for k in ("outage_day", "snow_like_day")}
        poa = P.ac_per_kwp(irr, fit["tilt"], fit["azimuth"]) / P.SYSTEM_EFF
        temp_loss = float(1 - (P.ac_per_kwp(irr, fit["tilt"], fit["azimuth"]).sum() / (poa * P.SYSTEM_EFF).sum()))
        rows.append({"system": name, **{k: v for k, v in fit.items() if k != "sse"}, **met,
                     "spikes": int(flags.spike.sum()), "outage_days": int(flags.outage_day.sum() / 24),
                     "snow_like_days": int(flags.snow_like_day.sum() / 24),
                     "lost_outage_pct": lost["outage_day"], "lost_snow_pct": lost["snow_like_day"],
                     "held_out_hours": int(test.sum())})
        lt = fit["kwp"] * P.ac_per_kwp(irr_all, fit["tilt"], fit["azimuth"])
        ann = (lt.resample("YS").sum() / fit["kwp"])
        ann.index = ann.index.year
        ann = ann.loc[1991:2025]
        lt_rows[name] = ann
        curves[name] = (meas, model, flags)
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "wp5_systems.csv", index=False)
    lt = pd.DataFrame(lt_rows)
    lt.to_csv(RES / "wp5_specific_yield_kwh_per_kwp.csv")
    ref = lt["residential4_pv"]
    p50 = float(ref.mean())
    iav = float(ref.std() / p50)
    # calibration: fitted on the first year, the reference system misses the held-out period by this much
    model_unc = float(abs(df.set_index("system").loc["residential4_pv", "bias_pct"]) / 100)
    s1 = np.sqrt(iav**2 + model_unc**2 + 0.02**2)       # 2%: ERA5 irradiance bias not removed by a single site
    s10 = np.sqrt((iav / np.sqrt(10)) ** 2 + model_unc**2 + 0.02**2)
    slope = np.polyfit(ref.index, ref.to_numpy(), 1)[0]
    early, late = ref.loc[1991:2000].mean(), ref.loc[2016:2025].mean()
    summ = {"reference_system": "residential4_pv", "p50_kwh_per_kwp": p50, "iav": iav, "model_unc": model_unc,
            "p90_1yr_ratio": float(1 - Z90 * s1), "p90_10yr_ratio": float(1 - Z90 * s10),
            "trend_kwh_per_kwp_per_decade": float(slope * 10), "mean_2016_2025_vs_1991_2000_pct": float((late / early - 1) * 100),
            "p50_last_10_years": float(late)}
    (RES / "wp5_longterm.json").write_text(json.dumps(summ, indent=1))
    fit = df.set_index("system").loc["residential4_pv"]
    registry.save_estimate("konstanz_residential4_pv", {"p50_kwh_per_kwp": p50, "p90_1yr_ratio": summ["p90_1yr_ratio"],
                                                        "p90_10yr_ratio": summ["p90_10yr_ratio"]},
                           [S.RAW / "opsd_konstanz_pv_cumulative_kwh.csv", S.RAW / "era5_konstanz_1991_2025.csv.gz"],
                           {"tilt": float(fit.tilt), "azimuth": float(fit.azimuth), "kwp": float(fit.kwp), "period": "1991-2025"})
    print(json.dumps(summ, indent=1))
    plot(curves, lt)


def plot(curves, lt):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    meas, model, flags = curves["residential4_pv"]
    a, b = "2016-06-01", "2016-06-14"
    ax[0].plot(meas.loc[a:b], "k", lw=1, label="metered")
    ax[0].plot(model.loc[a:b], "tab:orange", lw=0.8, label="pvlib on ERA5, fitted orientation")
    ax[0].set_ylabel("kWh per hour")
    ax[0].set_title("Konstanz residential PV, two weeks of June 2016", fontsize=9)
    ax[0].legend(fontsize=7)
    ax[0].tick_params(axis="x", rotation=30, labelsize=7)
    for c in lt.columns[:3]:
        ax[1].plot(lt.index, lt[c], lw=1, label=c)
    ax[1].set_ylabel("specific yield (kWh per kWp)")
    ax[1].set_title("Modelled yield 1991-2025 on ERA5 (fitted systems)", fontsize=9)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "wp5_pv.png", dpi=140)


if __name__ == "__main__":
    main()
