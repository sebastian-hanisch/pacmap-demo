"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem OR-Demo-Portfolio, siehe km_presets.py in kmeans-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import pacmap_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


SETTING_SPECS = {
    "n_tours_slider": SettingSpec("n", int, C.DEFAULT_N_TOURS, C.N_TOURS_MIN, C.N_TOURS_MAX),
    "q_slider": SettingSpec("q", int, C.DEFAULT_Q, C.Q_MIN, C.Q_MAX),
    "curvature_slider": SettingSpec("curv", float, C.DEFAULT_CURVATURE, C.CURVATURE_MIN, C.CURVATURE_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "outlier_slider": SettingSpec("out", int, C.DEFAULT_OUTLIER_PCT, C.OUTLIER_PCT_MIN, C.OUTLIER_PCT_MAX),
    "n_neighbors_slider": SettingSpec("nn", int, C.DEFAULT_N_NEIGHBORS, C.N_NEIGHBORS_MIN, C.N_NEIGHBORS_MAX),
    "mn_ratio_select": SettingSpec("mn", float, C.DEFAULT_MN_RATIO, min(C.MN_RATIO_CHOICES), max(C.MN_RATIO_CHOICES)),
    "fp_ratio_select": SettingSpec("fp", float, C.DEFAULT_FP_RATIO, min(C.FP_RATIO_CHOICES), max(C.FP_RATIO_CHOICES)),
    "n_iter_slider": SettingSpec("it", int, C.DEFAULT_N_ITER, C.N_ITER_MIN, C.N_ITER_MAX),
    "init_select": SettingSpec("init", _choice(C.INITS), C.DEFAULT_INIT),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
}
PRESET_KEYS = {"n_tours": "n_tours_slider", "q": "q_slider", "curvature": "curvature_slider", "noise": "noise_slider", "outlier_pct": "outlier_slider",
               "n_neighbors": "n_neighbors_slider", "mn_ratio": "mn_ratio_select", "fp_ratio": "fp_ratio_select", "n_iter": "n_iter_slider", "init": "init_select", "seed": "seed_input"}


def snap(choices, value):
    return min(choices, key=lambda choice: abs(choice - value))


def n_neighbors_max(n_tours, mn_ratio, fp_ratio):
    """Obere Grenze von n_neighbors: k · (1 + MN + FP) muss unter der Tourenzahl bleiben (sonst ordnet PaCMAP die Paare um), höchstens 50, mindestens 2."""
    return int(max(2, min(C.N_NEIGHBORS_MAX, (n_tours - 1) // (1.0 + mn_ratio + fp_ratio))))


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["mn_ratio_select"] = snap(C.MN_RATIO_CHOICES, st.session_state.get("mn_ratio_select", C.DEFAULT_MN_RATIO))
    st.session_state["fp_ratio_select"] = snap(C.FP_RATIO_CHOICES, st.session_state.get("fp_ratio_select", C.DEFAULT_FP_RATIO))
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
