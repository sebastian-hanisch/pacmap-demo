"""Jedes Preset zeigt, was sein Name und seine Hilfe behaupten (Bänder mit dem ausgelieferten Code kalibriert, bewusst weit: PaCMAP ist ein Zufalls-/Adam-Verfahren, die CI-Plattform rechnet anders als lokal)."""

import pytest

import pacmap_constants as C
from pacmap_evaluation import Settings, analyse, make_dataset, verdict


def _settings(p):
    return Settings(n_neighbors=p["n_neighbors"], mn_ratio=float(p["mn_ratio"]), fp_ratio=float(p["fp_ratio"]), n_iter=p["n_iter"], init=p["init"])


def _measure(p):
    dataset = make_dataset(p["n_tours"], p["q"], p["curvature"], p["noise"], p["outlier_pct"], p["seed"])
    s = _settings(p)
    a = analyse(dataset, s)
    code, data = verdict(a, dataset, s)[1:]
    return {"verdict": code, **data}


def test_every_preset_has_help_and_bands():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS)
    assert len(C.PRESETS) == 6


def test_preset_settings_are_within_slider_bounds():
    for p in C.PRESETS.values():
        assert C.N_TOURS_MIN <= p["n_tours"] <= C.N_TOURS_MAX and C.Q_MIN <= p["q"] <= C.Q_MAX
        assert C.CURVATURE_MIN <= p["curvature"] <= C.CURVATURE_MAX and C.NOISE_MIN <= p["noise"] <= C.NOISE_MAX and C.OUTLIER_PCT_MIN <= p["outlier_pct"] <= C.OUTLIER_PCT_MAX
        assert C.N_NEIGHBORS_MIN <= p["n_neighbors"] <= C.N_NEIGHBORS_MAX and p["mn_ratio"] in C.MN_RATIO_CHOICES and p["fp_ratio"] in C.FP_RATIO_CHOICES
        assert C.N_ITER_MIN <= p["n_iter"] <= C.N_ITER_MAX and p["n_iter"] % 9 == 0 and p["init"] in C.INITS


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_stays_inside_its_bands(name):
    measured = _measure(C.PRESETS[name])
    for key, expected in C.PRESET_EXPECTED_BANDS[name].items():
        value = measured[key]
        if isinstance(expected, str):
            assert value == expected, f"{key}: {value}"
        else:
            lo, hi = expected
            assert lo <= value <= hi, f"{key}: {value} nicht in [{lo}, {hi}]"
