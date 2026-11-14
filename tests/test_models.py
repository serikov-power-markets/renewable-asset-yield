"""Offline tests on synthetic data: filters, power curves, wakes, PV chain and the estimate registry."""
import numpy as np
import pandas as pd
import pytest

from assetyield import pv as P, scada as S, wakes as W


def synthetic_scada(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ws = rng.uniform(2, 16, n)
    curve = np.clip((ws**3 - 27) / (12**3 - 27), 0, 1) * S.RATED_KW
    d = pd.DataFrame({"time": pd.date_range("2017-01-01", periods=n, freq="10min"), "turbine": 1, "ws": ws,
                      "power_kw": curve + rng.normal(0, 20, n), "pitch": np.where(ws > 12, 8.0, 0.5),
                      "t_nacelle": 10.0, "lost_downtime_kwh": 0.0})
    d.loc[100:150, "power_kw"] = 0.0                 # a stop with wind
    d.loc[100:150, "lost_downtime_kwh"] = 50.0
    return d


def test_rule_filter_catches_stop_and_operator_loss():
    d = synthetic_scada()
    f = S.rule_filter(d)
    stopped = d.index.isin(range(100, 151)) & (d.ws > 4.5)
    assert f.loc[stopped, "bad"].all()
    stop_log = pd.Series(d.index.isin(range(100, 151)), index=d.index)
    cmp = S.compare_with_log(f, stop_log, d.lost_downtime_kwh)
    assert cmp["operator_lost_energy_caught_share"] > 0.8


def test_power_curve_is_monotone_and_capped():
    d = synthetic_scada()
    d = d[~S.rule_filter(d).bad].copy()
    d["ws_norm"] = d.ws
    f = S.curve_function(S.power_curve(d))
    x = np.linspace(0, 24, 200)
    y = f(x)
    assert np.all(np.diff(y) >= -1e-9)
    assert y.max() <= S.RATED_KW * 1.05
    assert f(np.array([26.0]))[0] == 0.0


def test_density_normalisation_direction():
    ws = pd.Series([8.0])
    assert S.normalise_ws(ws, np.array([1.30]))[0] > 8.0 > S.normalise_ws(ws, np.array([1.15]))[0]


def test_pywake_efficiency_below_one_and_aligned_row_worst():
    t = pd.DataFrame({"Latitude": [52.0, 52.0, 52.0], "Longitude": [0.0, 0.0047, 0.0094],
                      "Hub Height (m)": [80.0, 80.0, 80.0]}, index=[1, 2, 3])     # a west-east row, about 320 m apart

    def pc(ws):
        return np.clip((np.asarray(ws) ** 3 - 27) / (12**3 - 27), 0, 1) * 2050

    eff, ref = W.pywake_efficiency(t, pc, alpha=0.0)
    assert (eff.eff_pywake <= 1 + 1e-9).all()
    assert eff.eff_pywake.loc[270] < eff.eff_pywake.loc[0]   # wind along the row loses most


def test_pv_fit_recovers_orientation():
    idx = pd.date_range("2016-04-01", "2016-09-30 23:00", freq="h")
    doy = idx.day_of_year.to_numpy()
    hour = idx.hour.to_numpy() + 0.5
    ghi = np.clip(np.sin(np.pi * (hour - 4.5) / 14), 0, None) * 800 * (0.8 + 0.2 * np.cos(doy))
    w = pd.DataFrame({"ghi": ghi, "t_air": 15.0, "wind": 2.0}, index=idx)
    irr = P.irradiance(w)
    truth = 5.0 * P.ac_per_kwp(irr, 30, 210)
    fit = P.fit_orientation(truth, irr, pd.Series(True, index=idx))
    assert fit["tilt"] == 30 and fit["azimuth"] == 210
    assert fit["kwp"] == pytest.approx(5.0, rel=1e-3)


def test_pv_output_zero_at_night():
    idx = pd.date_range("2016-06-01", periods=24, freq="h")
    w = pd.DataFrame({"ghi": 0.0, "t_air": 10.0, "wind": 1.0}, index=idx)
    assert (P.ac_per_kwp(P.irradiance(w), 30, 180) == 0).all()


def test_registry_detects_changed_input(tmp_path, monkeypatch):
    from assetyield import registry
    f = tmp_path / "x.csv"
    f.write_text("a,b\n1,2\n")
    monkeypatch.setattr(registry, "STORE", tmp_path / "store")
    monkeypatch.setattr(S, "RAW", tmp_path)
    registry.save_estimate("demo", {"p50": 1.0}, [f])
    assert registry.verify("demo")["inputs_unchanged"]
    f.write_text("a,b\n1,3\n")
    assert not registry.verify("demo")["inputs_unchanged"]
