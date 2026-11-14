"""WP0-WP2: validate the measured data, fit power curves, measure and model wake losses at Kelmarsh (2017).

WP0: data-driven filter vs the operator's event log and lost-production estimate.
WP1: density-normalised power curves per turbine (method of bins) and their spread.
WP2: observed farm efficiency by wind direction (clean records, 5-10 m/s, all six turbines running) against
     PyWake (Bastankhah Gaussian deficit, linear sum) with the real layout and the fitted power curve.
Outputs: results/wp0_filter.json, results/wp1_power_curves.csv, results/wp2_wakes_by_sector.csv,
         figures/wp1_power_curves.png, figures/wp2_wakes.png
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from assetyield import scada as S, wakes as W

RES = S.ROOT / "results"
FIG = S.ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


def main():
    d = S.load_scada()
    stop = S.stop_mask(d)
    flags = S.rule_filter(d)
    wp0 = S.compare_with_log(flags, stop, d.lost_downtime_kwh)
    wp0["share_records_flagged"] = float(flags.bad.mean())
    wp0["by_rule"] = {k: int(v) for k, v in flags.drop(columns="bad").sum().items()}
    (RES / "wp0_filter.json").write_text(json.dumps(wp0, indent=1))
    print("WP0", wp0)

    clean = d[~flags.bad & ~stop].copy()
    clean["rho"] = S.density(clean.t_nacelle)          # nacelle temperature, standard pressure (no site barometer)
    clean["ws_norm"] = S.normalise_ws(clean.ws, clean.rho)
    curves = {wt: S.power_curve(g) for wt, g in clean.groupby("turbine")}
    farm = S.power_curve(clean)
    pcs = pd.concat({f"WT{k}": v.power_kw for k, v in curves.items()} | {"all": farm.power_kw}, axis=1)
    pcs.to_csv(RES / "wp1_power_curves.csv")
    spread = (pcs.drop(columns="all").loc[5:11].std(axis=1) / pcs["all"].loc[5:11]).mean()
    print("WP1 turbine-to-turbine spread of power at 5-11 m/s (mean CV):", round(float(spread), 3))

    mod, ref = W.pywake_efficiency(S.turbines(), S.curve_function(farm))
    obs = W.observed_efficiency(clean, ref)
    biased = W.observed_efficiency(clean).rename(columns={"eff_observed": "eff_observed_maxref"})["eff_observed_maxref"]
    sec = obs.join(mod, how="outer").join(biased)
    sec.to_csv(RES / "wp2_wakes_by_sector.csv")
    print(sec.round(3).to_string())
    summary = W.wake_summary(clean, sec)
    (RES / "wp2_summary.json").write_text(json.dumps(summary, indent=1))
    print("WP2", summary)
    plot(pcs, sec)


def plot(pcs, sec):
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in pcs.columns:
        ax.plot(pcs.index, pcs[c], lw=1.6 if c == "all" else 0.8, color="k" if c == "all" else None, label=c)
    ax.set_xlabel("density-normalised wind speed at the nacelle (m/s)")
    ax.set_ylabel("power (kW)")
    ax.set_title("Kelmarsh 2017: power curves from clean 10-minute data", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "wp1_power_curves.png", dpi=140)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sec.index, sec.eff_observed, "o-", label="observed (SCADA)")
    ax.plot(sec.index, sec.eff_pywake, "s--", label="PyWake, Bastankhah Gaussian")
    ax.set_xlabel("wind direction sector (deg)")
    ax.set_ylabel("farm efficiency (farm power / 6 x reference turbine)")
    ax.set_title("Wake losses by direction, 5-10 m/s", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "wp2_wakes.png", dpi=140)


if __name__ == "__main__":
    main()
