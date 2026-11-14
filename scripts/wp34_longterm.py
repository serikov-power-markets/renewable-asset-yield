"""WP3 and WP4: ERA5 -> Kelmarsh production, verified on measured 2017 hours; long-term P50/P90 1991-2025.

WP3: leave-one-month-out hindcast: the speed factor k and the statistical correction are fitted on the other
eleven months and verified on the held-out month (physical chain alone and physical + gradient boosting).
WP4: the model fitted on all of 2017 is run on 35 years of ERA5; annual energy gives P50 and the interannual
variability; an uncertainty budget gives P90 for one year and for ten years; monthly P50 and P90 volumes; wake loss.
Outputs: results/wp3_*, results/wp4_*, figures/wp34_hindcast_longterm.png
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from assetyield import chain as C, registry, scada as S

RES, FIG = S.ROOT / "results", S.ROOT / "figures"
CAP = 6 * S.RATED_KW / 1000
Z90 = 1.2816


def fitted_curve():
    pcs = pd.read_csv(RES / "wp1_power_curves.csv", index_col=0)
    return S.curve_function(pcs[["all"]].rename(columns={"all": "power_kw"}).dropna())


def main():
    d = S.load_scada()
    meas = C.measured_farm_hourly(d)
    e = C.load_era5("kelmarsh")
    common = meas.index.intersection(e.index)
    meas, e17 = meas.loc[common], e.loc[common]
    obs = meas.pot_kwh                 # MWh per hour: production potential (metered + operator lost downtime)
    print("hours", len(common), "potential GWh", round(obs.sum() / 1000, 2), "metered", round(meas.e_kwh.sum() / 1000, 2))

    fm = C.FarmModel(S.turbines(), fitted_curve())
    months = obs.index.month
    phys_cv = pd.Series(index=obs.index, dtype=float)
    corr_cv = pd.Series(index=obs.index, dtype=float)
    ks = {}
    for mth in sorted(set(months)):
        tr, te = months != mth, months == mth
        ks[int(mth)] = fm.calibrate_k(e17[tr], obs[tr])
        p_tr = fm.physical(e17[tr])
        fm.fit_correction(e17[tr], obs[tr], p_tr)
        p_te = fm.physical(e17[te])
        phys_cv[te] = p_te.to_numpy()
        corr_cv[te] = fm.corrected(e17[te], p_te).to_numpy()
        print("month", mth, "k", round(ks[int(mth)], 4), flush=True)
    m_phys, m_corr = C.metrics(obs, phys_cv, CAP), C.metrics(obs, corr_cv, CAP)
    wp3 = {"physical": m_phys, "physical_plus_gbm": m_corr, "k_by_month": ks,
           "k_spread": float(np.std(list(ks.values()))), "measured_potential_gwh": float(obs.sum() / 1000),
           "metered_gwh": float(meas.e_kwh.sum() / 1000), "hours": int(len(obs))}
    (RES / "wp3_hindcast.json").write_text(json.dumps(wp3, indent=1))
    pd.DataFrame({"measured": obs, "physical": phys_cv, "corrected": corr_cv}).to_csv(RES / "wp3_hindcast_hourly.csv")
    print("WP3 physical", m_phys, "\nWP3 corrected", m_corr, flush=True)

    fm.calibrate_k(e17, obs)
    p17 = fm.physical(e17)
    fm.fit_correction(e17, obs, p17)
    p_lt = fm.physical(e)
    lt = pd.DataFrame({"physical": p_lt, "corrected": fm.corrected(e, p_lt), "free_stream": fm.physical(e, wakes=False)})
    lt.to_parquet(RES / "wp4_longterm_hourly.parquet")
    ann = lt.resample("YS").sum() / 1000
    ann.index = ann.index.year
    ann.to_csv(RES / "wp4_annual_gwh.csv")
    a = ann.corrected
    p50 = float(a.mean())
    iav = float(a.std() / p50)
    monthly_bias = corr_cv.resample("MS").sum() / obs.resample("MS").sum() - 1
    unc = {"interannual_variability_1yr": iav, "interannual_variability_10yr": iav / np.sqrt(10),
           "model_calibration": float(monthly_bias.std() / np.sqrt(12)), "measurement": 0.01,
           "long_term_period": float(a.rolling(10).mean().dropna().std() / p50)}
    comb = np.sqrt(unc["model_calibration"] ** 2 + unc["measurement"] ** 2 + unc["long_term_period"] ** 2)
    s1 = float(np.sqrt(comb**2 + unc["interannual_variability_1yr"] ** 2))
    s10 = float(np.sqrt(comb**2 + unc["interannual_variability_10yr"] ** 2))
    trend = np.polyfit(a.index, a.to_numpy(), 1)[0] / p50 * 100
    wake = 1 - ann.physical.sum() / ann.free_stream.sum()
    mon = lt.corrected.resample("MS").sum() / 1000
    mprof = mon.groupby(mon.index.month).agg(p50="mean", p90=lambda x: np.quantile(x, 0.10),
                                             p10=lambda x: np.quantile(x, 0.90))
    mprof.to_csv(RES / "wp4_monthly_volume_gwh.csv")
    wp4 = {"p50_gwh": p50, "capacity_factor_p50": p50 * 1000 / (CAP * 8766), "uncertainty": unc,
           "sigma_1yr": s1, "sigma_10yr": s10, "p90_1yr_gwh": p50 * (1 - Z90 * s1), "p90_10yr_gwh": p50 * (1 - Z90 * s10),
           "lowest_year": int(a.idxmin()), "lowest_year_gwh": float(a.min()), "highest_year": int(a.idxmax()),
           "highest_year_gwh": float(a.max()), "trend_pct_per_year": float(trend), "wake_loss_longterm_pct": float(100 * wake),
           "year_2017_rank_of_35": int(a.rank(ascending=False)[2017]), "year_2017_vs_p50": float(a[2017] / p50 - 1),
           "physical_vs_corrected_longterm": float(ann.physical.mean() / p50 - 1)}
    (RES / "wp4_longterm.json").write_text(json.dumps(wp4, indent=1))
    registry.save_estimate("kelmarsh_wind", {"p50_gwh": wp4["p50_gwh"], "p90_1yr_gwh": wp4["p90_1yr_gwh"],
                                             "p90_10yr_gwh": wp4["p90_10yr_gwh"],
                                             "monthly_gwh": mprof.round(3).to_dict(orient="index")},
                           [S.RAW / "kelmarsh_scada_2017_10min.parquet", S.RAW / "kelmarsh_status_2017.csv",
                            S.RAW / "era5_kelmarsh_1991_2025.csv.gz", S.RAW / "kelmarsh_turbines.csv"],
                           {"k": fm.k, "ti": fm.ti, "wake_model": "BastankhahGaussian", "period": "1991-2025"})
    print("WP4", json.dumps(wp4, indent=1))
    plot(obs, phys_cv, corr_cv, ann, p50, wp4)


def plot(obs, phys, corr, ann, p50, wp4):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    a, b = "2017-02-01", "2017-02-21"
    ax[0].plot(obs.loc[a:b], "k", lw=1, label="measured potential")
    ax[0].plot(phys.loc[a:b], "tab:blue", lw=0.8, label="physical chain (held-out month)")
    ax[0].plot(corr.loc[a:b], "tab:red", lw=0.8, label="physical + gradient boosting")
    ax[0].set_ylabel("farm output (MW)")
    ax[0].set_title("Hindcast from ERA5, February 2017 (fitted without February)", fontsize=9)
    ax[0].legend(fontsize=7)
    ax[0].tick_params(axis="x", rotation=30, labelsize=7)
    ax[1].bar(ann.index, ann.corrected, color="tab:grey")
    ax[1].axhline(p50, color="k", label=f"P50 {p50:.1f} GWh")
    ax[1].axhline(wp4["p90_1yr_gwh"], color="tab:red", ls="--", label=f"P90 one year {wp4['p90_1yr_gwh']:.1f}")
    ax[1].axhline(wp4["p90_10yr_gwh"], color="tab:orange", ls=":", label=f"P90 ten years {wp4['p90_10yr_gwh']:.1f}")
    ax[1].set_ylim(ann.corrected.min() * 0.85, ann.corrected.max() * 1.05)
    ax[1].set_ylabel("production potential (GWh/a)")
    ax[1].set_title("Kelmarsh modelled on ERA5 1991-2025", fontsize=9)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "wp34_hindcast_longterm.png", dpi=140)


if __name__ == "__main__":
    main()
