"""Turn the raw downloads into small files in data/raw (run once; the repository ships the result).

Sources:
- Kelmarsh wind farm SCADA 2017, 6 x Senvion MM92 (2.05 MW, 92 m rotor), Cubico Sustainable Investments,
  Plumley (2022), Zenodo 10.5281/zenodo.5841834, CC BY 4.0. Ten-minute data, UTC.
- ERA5 hourly point time series 1991-2025 at Kelmarsh and Konstanz (Copernicus Climate Change Service,
  reanalysis-era5-single-levels-timeseries).
- OPSD household data 2015-2017, Konstanz (Open Power System Data, CC BY 4.0): metered PV of small businesses.
"""
from pathlib import Path
import zipfile

import pandas as pd

HOME = Path.home()
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
G5 = HOME / "data_g5" / "raw"

COLS = {"# Date and time": "time", "Wind speed (m/s)": "ws", "Density adjusted wind speed (m/s)": "ws_rho",
        "Wind direction (°)": "wd", "Nacelle position (°)": "nacelle", "Power (kW)": "power_kw",
        "Nacelle ambient temperature (°C)": "t_nacelle", "Lost Production to Downtime (kWh)": "lost_downtime_kwh",
        "Lost Production to Curtailment (Total) (kWh)": "lost_curtail_kwh",
        "Lost Production to Performance (kWh)": "lost_perf_kwh", "Energy Export (kWh)": "energy_export_kwh",
        "Blade angle (pitch position) A (°)": "pitch"}

z = zipfile.ZipFile(G5 / "kelmarsh" / "Kelmarsh_SCADA_2017_3083.zip")
frames = []
for name in sorted(n for n in z.namelist() if n.startswith("Turbine_Data")):
    wt = int(name.split("_")[3])
    head = pd.read_csv(z.open(name), skiprows=9, nrows=0).columns
    use = [c for c in head if c in COLS]
    d = pd.read_csv(z.open(name), skiprows=9, usecols=use).rename(columns=COLS)
    d["time"] = pd.to_datetime(d.time)
    d["turbine"] = wt
    frames.append(d)
    print(name, len(d), sorted(set(COLS.values()) - set(d.columns)))
scada = pd.concat(frames, ignore_index=True)
scada.to_parquet(RAW / "kelmarsh_scada_2017_10min.parquet", index=False)

st = []
for name in sorted(n for n in z.namelist() if n.startswith("Status")):
    s = pd.read_csv(z.open(name), skiprows=9)
    s["turbine"] = int(name.split("_")[2])
    st.append(s)
pd.concat(st, ignore_index=True).to_csv(RAW / "kelmarsh_status_2017.csv", index=False)
(RAW / "kelmarsh_turbines.csv").write_bytes((G5 / "kelmarsh" / "Kelmarsh_WT_static.csv").read_bytes())

for site in ("kelmarsh", "konstanz"):
    zz = zipfile.ZipFile(G5 / "era5_points" / f"era5_{site}_1991_2025.zip")
    csv = [n for n in zz.namelist() if n.endswith(".csv")][0]
    e = pd.read_csv(zz.open(csv))
    e.to_csv(RAW / f"era5_{site}_1991_2025.csv.gz", index=False, compression="gzip")
    print(site, e.columns.tolist(), len(e))

h = pd.read_csv(HOME / "de-site-energy-management" / "data" / "raw" / "household_data_60min_singleindex.csv",
                usecols=lambda c: c in ("utc_timestamp",) or c.endswith("_pv") or "_pv_" in c)
h.to_csv(RAW / "opsd_konstanz_pv_cumulative_kwh.csv", index=False)
print("pv", h.columns.tolist())
