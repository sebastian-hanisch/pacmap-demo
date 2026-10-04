"""PaCMAP an Lieferrouten-Kennzahlen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - PaCMAP - und lässt
stattdessen das Beispiel wachsen. Sechstes Stück der Dimensionsreduktion-Linie der "Konzepte"-Reihe und Schluss der Kette t-SNE → UMAP → PaCMAP: PaCMAP soll UMAPs
verbliebene Schwächen beheben (globale Struktur, lokal-global-Balance) - über einen Drei-Paar-Verlust mit Gewichtsschema in Phasen. Was davon stimmt, wird hier gemessen.
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import pacmap_constants as C
from pacmap_evaluation import (
    Settings, analyse, convergence_rows, fp_ratio_sweep, iterations_sweep, make_dataset, mn_ratio_sweep, n_neighbors_sweep, out_of_sample, stability, timing_sweep, verdict,
)
from pacmap_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    n_neighbors_max,
    randomize_seed,
    sync_query_params,
)
from pacmap_visualization import (
    build_distance_fidelity,
    build_embedding,
    build_optimization,
    build_out_of_sample,
    build_pair_losses,
    build_pairs_view,
    build_stability_compare,
    build_sweep,
    build_timing,
    build_weight_schedule,
)

st.set_page_config(page_title="PaCMAP – Sebastian Hanisch", layout="wide")

STEP_LABELS = {
    1: "1 · Drei Paartypen",
    2: "2 · Verlust & Phasen",
    3: "3 · Optimierung",
    4: "4 · Ergebnis",
}


def _ratio_label(value):
    return f"{value:g}"


@st.cache_data(show_spinner=False)
def _dataset(n_tours, q, curvature, noise, outlier_pct, seed):
    return make_dataset(n_tours, q, curvature, noise, outlier_pct, seed)


@st.cache_data(show_spinner=False)
def _analysis(data_params, settings):
    return analyse(make_dataset(*data_params), settings)


@st.cache_data(show_spinner=False)
def _sweeps(q, curvature, noise, outlier_pct):
    return (n_neighbors_sweep(q, curvature, noise, outlier_pct), mn_ratio_sweep(q, curvature, noise, outlier_pct), fp_ratio_sweep(q, curvature, noise, outlier_pct),
            iterations_sweep(q, curvature, noise, outlier_pct))


@st.cache_data(show_spinner=False)
def _stability(q, curvature, noise, outlier_pct, settings):
    return stability(q, curvature, noise, outlier_pct, settings)


@st.cache_data(show_spinner=False)
def _oos(data_params, settings):
    return out_of_sample(make_dataset(*data_params), settings)


st.title("🧭 PaCMAP an Lieferrouten-Kennzahlen")
st.markdown(
    """
Dieselben **12 Kennzahlen je Lieferroute** wie in der PCA-, Isomap-, LLE-, t-SNE- und UMAP-Demo - erzeugt aus wenigen versteckten Faktoren, aber mit **gekrümmter** Struktur, an der die PCA scheiterte.
**PaCMAP** (Pairwise Controlled Manifold Approximation) ist der Schluss der Kette t-SNE → UMAP → PaCMAP. Es verzichtet auf Wahrscheinlichkeiten und Graphen und arbeitet stattdessen mit **drei Arten von Paaren**:
**nahe** Paare ziehen sich an, **mittlere** Paare ziehen sich - zuerst stark - an, **ferne** Paare stoßen sich ab. Ein **Gewichtsschema in drei Phasen** schaltet erst die globale, dann die lokale Struktur in den Vordergrund.
Es soll UMAPs verbliebene Schwäche beheben: die **globale Struktur** und die **Balance** zwischen lokal und global. Was davon **tatsächlich** stimmt, misst die Demo direkt gegen UMAP, t-SNE, Isomap, LLE und PCA - und sagt es,
wenn eine Erwartung nicht aufgeht. (Einordnung: PaCMAP ist neuer und deutlich weniger verbreitet als UMAP; einen etablierten Standard-Status hat es nicht.)
Wie das Verfahren funktioniert, erklärt der aufgeklappte Abschnitt direkt darunter.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - sechstes Stück der "
    "Dimensionsreduktion-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel: PaCMAP ist die Fortsetzung von UMAP und schließt die Kette; "
    "welche seiner Versprechen sich hier messen lassen, steht unten."
)

with st.expander("So funktioniert PaCMAP", expanded=True):
    st.markdown(
        """
PaCMAP (Wang, Huang, Rudin & Shaposhnik, 2021) besteht aus drei Schritten:

1. **Drei Paartypen**: für jede Tour werden feste Partner gewählt - **nahe Paare** (die *n_neighbors* nächsten Nachbarn, Abstand skaliert an der lokalen Dichte), **mittlere Paare** (aus 6 Zufallstouren die *zweitnächste*: nicht
   nah, aber auch nicht beliebig fern) und **ferne Paare** (Zufallstouren, die keine nahen Paare sind). Wie viele es sind, legen zwei Verhältnisse zu *n_neighbors* fest.
2. **Verlust je Paar**: nahe und mittlere Paare werden zusammengezogen, ferne auseinandergedrückt - jeweils mit einer beschränkten Kraft (der Verlust sättigt), damit weder Anziehung noch Abstoßung das Bild dominieren.
3. **Gewichtsschema in drei Phasen**: in **Phase 1** wiegen die mittleren Paare sehr stark (Gewicht 1000, das auf 3 sinkt) - sie ordnen die Touren grob im Ganzen; in **Phase 2** balancieren nahe und mittlere Paare;
   in **Phase 3** verschwinden die mittleren Paare (Gewicht 0) und die nahen Paare verfeinern nur noch lokal. Optimiert wird mit dem Adam-Verfahren; gestartet wird an der PCA.

Was PaCMAP im Vergleich zu UMAP **wirklich** ändert, steht unten in "🆚" - gemessen, nicht behauptet.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_tours = st.slider("Anzahl Touren", *bounds("n_tours_slider"), key="n_tours_slider", step=50)
    q = st.slider(
        "Wahre Anzahl versteckter Faktoren (q)", *bounds("q_slider"), key="q_slider",
        help="So viele echte Einflussgrößen erzeugen die 12 Kennzahlen. Mit mehr Faktoren als Zielraum-Dimensionen (2) wird die Stichprobe dünner und die Einbettung schwächer.",
    )
    curvature = st.slider(
        "Krümmung", *bounds("curvature_slider"), key="curvature_slider", step=0.05,
        help="0 = die Kennzahlen hängen linear von den Faktoren ab (dann hat PaCMAP keinen Vorteil vor der PCA). Größer = die Touren liegen auf einer zunehmend gebogenen Fläche.",
    )
    noise = st.slider("Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05, help="Messrauschen je Kennzahl.")
    outlier_pct = st.slider(
        "Sonderfahrten (%)", *bounds("outlier_slider"), key="outlier_slider",
        help="Anteil der Touren mit extremem Zeitdruck-Faktor (zehnfach vergrößert). PCA und Isomap behalten die Größenordnung - PaCMAP, UMAP und t-SNE stauchen die Extreme.",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**PaCMAP**")
    nn_max = n_neighbors_max(int(n_tours), float(st.session_state["mn_ratio_select"]), float(st.session_state["fp_ratio_select"]))
    if st.session_state["n_neighbors_slider"] > nn_max:
        st.session_state["n_neighbors_slider"] = nn_max
    n_neighbors = st.slider(
        "n_neighbors (nahe Paare je Tour)", C.N_NEIGHBORS_MIN, nn_max, key="n_neighbors_slider",
        help="Zahl der nahen Paare je Tour. Sehr klein: die lokale Verankerung fehlt (im Test R² 0.49 bei 2 statt 0.80 bei 10, Seed 7). Groß: besser - die Grenze folgt Tourenzahl und den Paar-Verhältnissen.",
    )
    mn_ratio = st.select_slider(
        "MN-Verhältnis (mittlere Paare)", options=C.MN_RATIO_CHOICES, key="mn_ratio_select", format_func=_ratio_label,
        help="Mittlere Paare je Tour = Verhältnis · n_neighbors. 0 = keine mittleren Paare: im Test brach die Abstandstreue ferner Paare ein (Seed 7: -0.04 statt 0.39; Mittel über 3 Seeds 0.06 statt 0.33).",
    )
    fp_ratio = st.select_slider(
        "FP-Verhältnis (ferne Paare)", options=C.FP_RATIO_CHOICES, key="fp_ratio_select", format_func=_ratio_label,
        help="Ferne Paare je Tour = Verhältnis · n_neighbors. Im Test (Verhältnis 0.5 bis 4) hatte das keinen verlässlichen Einfluss auf R² und Abstandstreue - die Ergebnisse schwanken zwischen Datensätzen.",
    )
    n_iter = st.slider(
        "Iterationen", *bounds("n_iter_slider"), key="n_iter_slider", step=90,
        help="Gesamtzahl der Adam-Schritte, die drei Phasen wachsen proportional (450 = 100 + 100 + 250 wie im Original). Im Test änderte sich das R² ab etwa 270 Iterationen kaum.",
    )
    init = st.selectbox(
        "Initialisierung", C.INITS, key="init_select", format_func=lambda i: C.INIT_LABELS[i],
        help="Startlayout. PCA ist deterministisch (Standard). Zufällige Starts zeigen, wie stark das Ergebnis vom Start abhängt (siehe Stabilität unten).",
    )

    st.button("🎲 Neue Touren generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für die Touren.")

sync_query_params({
    "n_tours_slider": int(n_tours), "q_slider": int(q), "curvature_slider": curvature, "noise_slider": noise, "outlier_slider": int(outlier_pct), "n_neighbors_slider": int(n_neighbors),
    "mn_ratio_select": mn_ratio, "fp_ratio_select": fp_ratio, "n_iter_slider": int(n_iter), "init_select": init, "seed_input": int(seed),
})

data_params = (int(n_tours), int(q), float(curvature), float(noise), int(outlier_pct), int(seed))
settings = Settings(n_neighbors=int(n_neighbors), mn_ratio=float(mn_ratio), fp_ratio=float(fp_ratio), n_iter=int(n_iter), init=init)
with st.spinner("Wähle die Paare und optimiere die Einbettung..."):
    dataset = _dataset(*data_params)
    analysis = _analysis(data_params, settings)
model = analysis.pacmap
metrics = analysis.metrics
z_color = dataset.z[:, 0]
iso_idx = analysis.iso_indices
data_key = data_params + (settings,)
snap_iters = sorted(model.snapshots)
with st.spinner("Prüfe n_neighbors, MN-/FP-Verhältnis und Iterationen über feste Sweep-Seeds..."):
    nn_rows, mn_rows, fp_rows, it_rows = _sweeps(int(q), float(curvature), float(noise), int(outlier_pct))
level, code, vd = verdict(analysis, dataset, settings)
r2_points = convergence_rows(analysis)

# --- PaCMAP in Aktion ------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 PaCMAP in Aktion")
st.caption(
    "Die 2-D-Ansicht in Schritt 1 zeigt die Touren in den ersten beiden Hauptkomponenten (Farbe = versteckter Faktor 1) - nur als Zeichenfläche; PaCMAP selbst rechnet in allen 12 Dimensionen."
)
if "pacmap_step" not in st.session_state or st.session_state.get("pacmap_step_owner") != data_key:
    st.session_state["pacmap_step"] = 1
    st.session_state["pacmap_snap"] = snap_iters[-1]
    st.session_state["pacmap_step_owner"] = data_key
step_col, play_col = st.columns([5, 1])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="pacmap_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")

view = analysis.pca_2d
centre = view.mean(0)
focus = int(np.argmin(((view - centre) ** 2).sum(1)))
near_idx = model.nb_pairs[model.nb_pairs[:, 0] == focus, 1]
mid_idx = model.mn_pairs[model.mn_pairs[:, 0] == focus, 1] if len(model.mn_pairs) else np.array([], dtype=int)
far_idx = model.fp_pairs[model.fp_pairs[:, 0] == focus, 1]
if st.session_state.get("pacmap_snap") not in model.snapshots:
    st.session_state["pacmap_snap"] = snap_iters[-1]
if step == 3 and not auto_play:
    snap_it = st.select_slider("Iteration", options=snap_iters, key="pacmap_snap", format_func=lambda i: "Start" if i == 0 else f"Iteration {i}")
else:
    snap_it = st.session_state.get("pacmap_snap", snap_iters[-1])
    if snap_it not in model.snapshots:
        snap_it = snap_iters[-1]


def _mean_distance(pairs):
    if len(pairs) == 0:
        return float("nan")
    return float(np.linalg.norm(model.Z[pairs[:, 0]] - model.Z[pairs[:, 1]], axis=1).mean())


view_slot = st.empty()


def _render(current_step, iteration=None):
    if current_step == 1:
        with view_slot.container():
            c1, c2 = st.columns([3, 2])
            c1.plotly_chart(build_pairs_view(view, z_color, focus, near_idx, mid_idx, far_idx), width="stretch", key="pacmap_pairs_view")
            c2.markdown("**Die drei Paartypen (alle Touren)**")
            c2.table({
                "Paartyp": ["nah", "mittel", "fern"],
                "je Tour": [model.n_neighbors, model.n_mn, model.n_fp],
                "Wirkung": ["zieht an", "zieht an (Phase 1-2)", "stößt ab"],
                "mittlerer Abstand im Original": [f"{_mean_distance(model.nb_pairs):.2f}", "–" if model.n_mn == 0 else f"{_mean_distance(model.mn_pairs):.2f}", f"{_mean_distance(model.fp_pairs):.2f}"],
            })
    elif current_step == 2:
        with view_slot.container():
            c1, c2 = st.columns([3, 2])
            c1.markdown("**Verlust je Paar in den drei Phasen**")
            c1.plotly_chart(build_pair_losses([("Anfang Phase 1", 1000.0, 2.0, 1.0), ("Phase 2", 3.0, 3.0, 1.0), ("Phase 3", 0.0, 1.0, 1.0)]), width="stretch", key="pacmap_pair_losses")
            c2.markdown("**Gewichtsschema**")
            c2.plotly_chart(build_weight_schedule(model.weights, model.iters), width="stretch", key="pacmap_weight_schedule")
    elif current_step == 3:
        it = snap_it if iteration is None else iteration
        with view_slot.container():
            c1, c2 = st.columns([2, 3])
            c1.markdown(f"**Einbettung nach Iteration {it}**")
            c1.plotly_chart(build_embedding(model.snapshots[it], z_color, "Koordinate 1", "Koordinate 2"), width="stretch", key=f"pacmap_snapshot_{it}")
            c2.markdown("**Optimierung**")
            c2.plotly_chart(build_optimization(model.losses, dict(r2_points), analysis.snapshot_far, model.iters, marker=it), width="stretch", key=f"pacmap_optimization_{it}")
    else:
        with view_slot.container():
            c1, c2 = st.columns(2)
            c1.markdown("**PaCMAP: Einbettung**")
            c1.plotly_chart(build_embedding(model.embedding, z_color, "PaCMAP-Koordinate 1", "PaCMAP-Koordinate 2"), width="stretch", key="pacmap_embed_step")
            c2.markdown("**Zum Vergleich: PCA**")
            c2.plotly_chart(build_embedding(analysis.pca_2d, z_color, "PC1", "PC2"), width="stretch", key="pca_embed_step")


if auto_play:
    for s in STEP_LABELS:
        if s == 3:
            for it in snap_iters:
                _render(3, it)
                time.sleep(0.35)
        else:
            _render(s)
            time.sleep(1.0)
    step = 4
else:
    _render(step)

if step == 1:
    st.caption(
        f"Die gewählte Tour (Stern, nahe der Mitte) hat {len(near_idx)} nahe (orange), {len(mid_idx)} mittlere (grün) und {len(far_idx)} ferne Partner (rot). Nahe Partner liegen im Original dicht bei der Tour, mittlere in mittlerer Entfernung, "
        "ferne beliebig weit. Die Paare werden **einmal** gewählt und dann festgehalten - PaCMAP kennt keine Wahrscheinlichkeiten und keinen Graphen, nur diese Paarlisten."
    )
elif step == 2:
    st.caption(
        "Der Verlust jedes Paars sättigt: ein nahes Paar kostet weniger als 1 × Gewicht, ein fernes höchstens ½ × Gewicht - deshalb kann kein einzelnes Paar die Einbettung dominieren. Das Gewicht der mittleren Paare (grün, log) startet bei 1000 und "
        "sinkt in Phase 1 auf 3; in Phase 3 ist es 0 - dann zählen nur noch nahe (orange) und ferne (rot) Paare, die Struktur wird lokal verfeinert."
    )
elif step == 3:
    st.caption(
        f"Die drei Phasen ({model.iters[0]} / {model.iters[1]} / {model.iters[2]} Iterationen) sind hinterlegt. Rechts: die Abstandstreue ferner Paare ist meist am Ende von Phase 1 am höchsten, wenn die mittleren Paare stark wirken, und sinkt danach - "
        "im Test (300 Touren, 6 Datensätze) im Mittel von 0.53 nach Phase 1 auf 0.36 am Ende: die lokale Verfeinerung erodiert die globale Ordnung."
    )
else:
    st.caption("Farbe = versteckter Faktor 1. Verläuft sie in der PaCMAP-Einbettung glatt und ohne Überlappung, hat PaCMAP die Fläche entrollt.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was PaCMAP gefunden hat - und die anderen fünf Verfahren auf denselben Daten")
st.caption(
    f"Vergleichsverfahren mit ihren guten Einstellungen (aus den Demos davor): UMAP n_neighbors {C.UMAP_N_NEIGHBORS}, min_dist {C.UMAP_MIN_DIST:g}; t-SNE Perplexity {C.TSNE_PERPLEXITY}, {C.TSNE_N_ITER} Iterationen; "
    f"Isomap k = {C.ISOMAP_K}; LLE k = {C.LLE_K}, Regularisierung {C.LLE_REG:g}; PCA auf z-Werten."
)
if len(iso_idx) < dataset.n:
    st.warning(f"⚠️ Der Isomap-Graph ist nicht zusammenhängend: nur {len(iso_idx)} von {dataset.n} Touren sind in der Isomap-Einbettung enthalten (siehe Isomap-Demo).")
m1, m2, m3, m4 = st.columns(4)
m1.metric("R² der wahren Faktoren", f"{metrics['pacmap']['r2']:.2f}", delta=f"{metrics['pacmap']['r2'] - metrics['umap']['r2']:+.2f} ggü. UMAP", delta_color="normal",
          help="Wie gut lassen sich die versteckten Faktoren aus den zwei Koordinaten zurückgewinnen (quadratische Regression). UMAP mit derselben Messung im Delta.")
m2.metric("Abstandstreue ferner Paare", f"{metrics['pacmap']['far']:.2f}", delta=f"{metrics['pacmap']['far'] - metrics['umap']['far']:+.2f} ggü. UMAP", delta_color="normal",
          help="Korrelation der Paarabstände in der Einbettung mit den Paarabständen der wahren Faktoren, nur für die obere Hälfte der wahren Abstände - der Test für 'globale Struktur'.")
m3.metric("Trustworthiness", f"{metrics['pacmap']['trust']:.2f}", delta=f"{metrics['pacmap']['trust'] - metrics['umap']['trust']:+.2f} ggü. UMAP", delta_color="normal",
          help=f"Nachbarschaft erhalten: Anteil der Nachbarn in der 2-D-Einbettung, die auch im Originalraum Nachbarn sind (k = {C.TRUST_NEIGHBORS}); 1 = perfekt.")
m4.metric("Paare je Tour", f"{model.n_neighbors} / {model.n_mn} / {model.n_fp}", help="Nahe / mittlere / ferne Paare je Tour (nach den Verhältnissen und der Tourenzahl).")

names = (("pacmap", "PaCMAP", "PaCMAP", model.embedding, z_color), ("umap", "UMAP (Vergleich)", "UMAP", analysis.umap.embedding, z_color), ("tsne", "t-SNE (Vergleich)", "t-SNE", analysis.tsne.embedding, z_color),
         ("isomap", "Isomap (Vergleich)", "Isomap", analysis.iso_2d, z_color[iso_idx]), ("lle", "LLE (Vergleich)", "LLE", analysis.lle.embedding[:, :2], z_color), ("pca", "PCA (Vergleich)", "PC", analysis.pca_2d, z_color))
row1 = st.columns(3)
row2 = st.columns(3)
for slot, (key, title, axis, coords, color) in zip(row1 + row2, names):
    with slot:
        st.markdown(f"**{title}**")
        st.plotly_chart(build_embedding(coords, color, f"{axis}-Koordinate 1" if axis != "PC" else "PC1", f"{axis}-Koordinate 2" if axis != "PC" else "PC2"), width="stretch", key=f"{key}_embedding")

order = ("pacmap", "umap", "tsne", "isomap", "lle", "pca")
labels = {"pacmap": "PaCMAP", "umap": "UMAP", "tsne": "t-SNE", "isomap": "Isomap", "lle": "LLE", "pca": "PCA"}
st.table({
    "Verfahren": [labels[k] for k in order],
    "R² der Faktoren": [f"{metrics[k]['r2']:.2f}" for k in order],
    "Abstandstreue (gesamt)": [f"{metrics[k]['fid']:.2f}" for k in order],
    "nahe Paare": [f"{metrics[k]['near']:.2f}" for k in order],
    "ferne Paare": [f"{metrics[k]['far']:.2f}" for k in order],
    "Trustworthiness": [f"{metrics[k]['trust']:.2f}" for k in order],
})

st.markdown("---")

# --- n_neighbors, Paar-Verhältnisse, Iterationen ---------------------------------------------------------------------------

st.subheader("📐 Wie stark hängen die Ergebnisse von n_neighbors, den Paar-Verhältnissen und den Iterationen ab?")
st.markdown(
    """
*n_neighbors* legt die lokale Verankerung fest, das *MN-Verhältnis* die Kraft, die ferne Regionen grob ordnet, das *FP-Verhältnis* wie stark alles auseinandergedrückt wird. Live für Ihr aktuelles Szenario über
**feste Sweep-Seeds** (unabhängig vom Demo-Seed) geprüft, nicht behauptet:
"""
)
if code == "neighbors_small":
    st.warning(
        f"⚠️ **Zu wenige nahe Paare**: mit n_neighbors = {vd['n_neighbors']} liegt das R² der Faktoren bei {vd['r2']:.2f}, ein Referenzlauf mit den Standard-Paaren (10 nahe, 5 mittlere) erreicht {vd['r2_ref']:.2f} "
        f"(UMAP {vd['r2_umap']:.2f}). Ohne genügend lokale Verankerung entrollt sich die Fläche nur unvollständig."
    )
elif code == "mn_missing":
    st.warning(
        f"⚠️ **Keine mittleren Paare**: die Abstandstreue ferner Paare liegt bei {vd['far']:.2f}, ein Referenzlauf mit den Standard-Paaren erreicht {vd['far_ref']:.2f} (R² {vd['r2']:.2f} gegen {vd['r2_ref']:.2f}). "
        "Die mittleren Paare sind die Kraft, die ferne Regionen in Phase 1 grob ordnet - ohne sie bleibt nur die lokale Struktur. (Das belegt den Mechanismus; ob PaCMAP damit globaler als UMAP ist, sagt die Tabelle unten.)"
    )
elif code == "global_structure":
    st.warning(
        f"⚠️ **Keine globale Struktur bei Sonderfahrten**: mit {vd['outlier_pct']} % Sonderfahrten liegt das R² der Faktoren bei {vd['r2']:.2f} (UMAP {vd['r2_umap']:.2f}, t-SNE {vd['r2_tsne']:.2f}, PCA {vd['r2_pca']:.2f}, Isomap {vd['r2_iso']:.2f}), "
        f"die Abstandstreue ferner Paare bei {vd['far']:.2f} (PCA {vd['far_pca']:.2f}). Auch PaCMAPs mittlere Paare erhalten die Größe von Abständen nicht - die Extreme werden an den Rand gestaucht. "
        "(Das R² wird hier von den Sonderfahrten dominiert - genau darum geht es.)"
    )
elif code == "no_advantage":
    st.info(f"ℹ️ **Kein Vorteil vor der PCA**: die Daten sind gerade (Krümmung 0) - R² der Faktoren {vd['r2']:.2f} (PaCMAP) gegen {vd['r2_pca']:.2f} (PCA); ferne Paare {vd['far']:.2f} gegen {vd['far_pca']:.2f}.")
elif code == "pacmap_wins":
    st.success(
        f"✅ **PaCMAP entrollt die Fläche**: R² der Faktoren {vd['r2']:.2f} gegen {vd['r2_pca']:.2f} bei der PCA (UMAP {vd['r2_umap']:.2f}, t-SNE {vd['r2_tsne']:.2f}, Isomap {vd['r2_iso']:.2f}), Trustworthiness {vd['trust']:.2f}. "
        f"Bei fernen Paaren: Abstandstreue {vd['far']:.2f} (UMAP {vd['far_umap']:.2f}, t-SNE {vd['far_tsne']:.2f}, Isomap {vd['far_iso']:.2f}) - hier liegt es nicht vorn."
    )
else:
    st.info(f"PaCMAP erreicht R² {vd['r2']:.2f} (PCA {vd['r2_pca']:.2f}, UMAP {vd['r2_umap']:.2f}) - kein klarer Gewinn und kein klarer Bruch.")

sweep_defs = (("n_neighbors", nn_rows, "n_neighbors", float(n_neighbors), False, "n_neighbors"), ("MN-Verhältnis", mn_rows, "mn_ratio", float(mn_ratio), False, "mn_ratio"),
              ("FP-Verhältnis", fp_rows, "fp_ratio", float(fp_ratio), False, "fp_ratio"), ("Iterationen", it_rows, "n_iter", float(n_iter), False, "iterations"))
for title, rows_, xkey, current, log, key in sweep_defs:
    st.markdown(f"**{title}**")
    st.plotly_chart(build_sweep(rows_, xkey, current, title, log=log), width="stretch", key=f"{key}_sweep")
st.caption(
    f"Gleiche Daten-Einstellungen (q = {dataset.q}, Krümmung {curvature:.2f}, Rauschen {noise:.2f}, Sonderfahrten {int(outlier_pct)} %), jeweils nur der geprüfte Regler wächst, alles Übrige Standard (10 / 0.5 / 2 / 450, PCA-Start); "
    f"Mittel über {len(C.SWEEP_SEEDS)} feste Seeds mit je {C.SWEEP_N_TOURS} Touren. Im Test: zu wenige nahe Paare und fehlende mittlere Paare verschlechtern das Ergebnis; das FP-Verhältnis und mehr als etwa 270 Iterationen ändern das R² kaum, "
    "und die Abstandstreue ferner Paare schwankt stark zwischen den Datensätzen (auf mittlerer Zahl von Datensätzen bleibt der Trend, aber einzelne Kurven sind zackig)."
)

st.markdown("---")

# --- Löst PaCMAP sein Versprechen ein? ------------------------------------------------------------------------------------

st.markdown("## 🆚 Löst PaCMAP sein Versprechen gegenüber UMAP ein? - gemessen")
st.markdown(
    """
| Versprechen / Frage | Ergebnis im Test (300 Touren, 4 feste Datensätze, wenn nicht anders angegeben) |
|---|---|
| **Bessere globale Struktur als UMAP** | ❌ Abstandstreue ferner Paare im Mittel **0.34** (UMAP 0.60, t-SNE 0.58, PCA 0.38, Isomap 0.89) - PaCMAP liegt hier am schlechtesten der drei Verfahren |
| **Extreme (Sonderfahrten) erhalten** | ❌ 5 %: R² **0.30** (UMAP 0.36, t-SNE 0.35, PCA 0.67, Isomap 0.77) - dieselbe Schwäche |
| **Mittlere Paare sind nötig für globale Ordnung** (Mechanismus) | ✅ ohne MN-Paare: ferne Paare **0.06** statt 0.33 (3 Seeds) |
| **Geschwindigkeit** | ✅ am schnellsten: n = 600: **0.5 s** gegen UMAP 1.7 s und t-SNE 5.1 s (auch bei n = 100–200 mindestens gleichauf) |
| **Stabilität** (Start egal) | ⚠️ zwischen t-SNE und UMAP: mittlere paarweise Abweichung zufälliger Starts (200 Touren, 4 Datensätze) bei q = 2: PaCMAP 0.10–0.43, UMAP 0.01–0.22, t-SNE 0.37–0.53; bei q = 3 gemischt |
| **Neue Touren einbetten** | ⚠️ nur als Behelf: R² 0.79–0.91 (wie `PaCMAP.transform`), UMAP `transform` 0.88–0.93; beim Neu-Rechnen verschieben sich die Trainings-Touren um Procrustes 0.06–0.56 |

Die Experimente mit Knopf unten prüfen Stabilität, Out-of-sample und Rechenzeit für Ihre Einstellungen nach.
"""
)

rng = np.random.default_rng(0)
m_iso = len(iso_idx)
pa = rng.integers(0, m_iso, size=min(1500, m_iso * (m_iso - 1) // 2))
pb = rng.integers(0, m_iso, size=len(pa))
keep = pa != pb
pa, pb = pa[keep], pb[keep]
ga, gb = iso_idx[pa], iso_idx[pb]


def _norm(d):
    return d / d.mean()


latent = _norm(np.linalg.norm(dataset.z[ga] - dataset.z[gb], axis=1))
pac_d = _norm(np.linalg.norm(model.embedding[ga] - model.embedding[gb], axis=1))
umap_d = _norm(np.linalg.norm(analysis.umap.embedding[ga] - analysis.umap.embedding[gb], axis=1))
iso_d = _norm(np.linalg.norm(analysis.iso_2d[pa] - analysis.iso_2d[pb], axis=1))
st.markdown("**📏 Globale Struktur: Abstände**")
st.plotly_chart(build_distance_fidelity(latent, [("PaCMAP", pac_d, "#d68a2e"), ("UMAP", umap_d, "#8e5fbf"), ("Isomap", iso_d, "#1f77b4")]), width="stretch", key="distance_fidelity")
st.caption(
    f"Jeder Punkt ein Tourenpaar: Abstand in der 2-D-Einbettung gegen den Abstand der wahren Faktoren; auf der gestrichelten Diagonale wäre die Einbettung abstandstreu. Korrelation für **nahe** Paare: "
    f"PaCMAP {metrics['pacmap']['near']:.2f}, UMAP {metrics['umap']['near']:.2f}, Isomap {metrics['isomap']['near']:.2f} - für **ferne** Paare: PaCMAP {metrics['pacmap']['far']:.2f}, UMAP {metrics['umap']['far']:.2f}, "
    f"Isomap {metrics['isomap']['far']:.2f}. Sonderfahrten (Regler links) machen den Unterschied zu PCA und Isomap drastisch sichtbar."
)

st.markdown("**🆕 Neue Touren einbetten (Out-of-sample)**")
st.caption(
    "PaCMAP hat im Paper keine Erweiterung auf neue Punkte; die Referenzimplementierung bietet `transform` (Pfade zu den Trainings-Touren, nur Anziehung, Trainings-Einbettung fest), das neue Punkte wie einen zusätzlichen Datensatz behandelt. "
    "Hier eine Behelfsfassung im selben Geist (Start am gewichteten Mittel der Nachbarn), gegen UMAPs `transform` und die t-SNE-Näherung. "
    f"Test: die letzten {C.HOLDOUT_FRACTION * 100:.0f} % der Touren zurückhalten; dazu, wie stark sich die bereits eingebetteten Touren beim Neu-Rechnen mit allen verschieben."
)
if st.button("🆕 Neue Touren testen", key="oos_start"):
    st.session_state["oos_on"] = True
if st.session_state.get("oos_on"):
    with st.spinner("Rechne PaCMAP, UMAP und t-SNE ohne und mit den neuen Touren..."):
        oos = _oos(data_params, settings)
    tc, ec = dataset.z[oos["train"], 0], dataset.z[oos["test"], 0]
    st.plotly_chart(build_out_of_sample([("PaCMAP: transform (Behelf)", oos["model"].embedding, oos["y_pacmap"]), ("UMAP: transform", oos["umap_train"], oos["y_umap"]),
                                         ("t-SNE: Näherung", oos["tsne_train"], oos["y_tsne"])], tc, ec), width="stretch", key="oos_plot")
    st.caption(
        f"Sterne = zurückgehaltene Touren. R² der wahren Faktoren für die neuen Touren: **PaCMAP {oos['r2_pacmap']:.2f}**, UMAP {oos['r2_umap']:.2f}, t-SNE-Näherung {oos['r2_tsne']:.2f} (Trainings-Touren PaCMAP: {oos['r2_train']:.2f}). "
        f"Beim Neu-Rechnen mit allen Touren verschieben sich die PaCMAP-Trainings-Touren um einen Procrustes-Abstand von {oos['shift_pacmap']:.2f} (0 = unverändert)."
    )

st.markdown("**🔀 Stabilität: PaCMAP, UMAP und t-SNE auf denselben Datensätzen**")
st.caption(
    "Wie stark hängt das Bild vom zufälligen Start ab? Je Datensatz vier zufällige Starts; gemessen wird der **mittlere paarweise Procrustes-Abstand** (nach bester Drehung/Spiegelung; 0 = gleiches Bild, 1 = unabhängig) - "
    "für alle drei Verfahren auf denselben vier festen Datensätzen mit 200 Touren und Ihren Datenreglern (q, Krümmung, Rauschen, Sonderfahrten); PaCMAP mit Ihren Einstellungen, UMAP und t-SNE mit ihren Standards. Dauert etwa 20 Sekunden."
)
if st.button("🔀 Stabilität vergleichen", key="stability_start"):
    st.session_state["stability_on"] = True
if st.session_state.get("stability_on"):
    with st.spinner("Rechne 48 Läufe..."):
        stab = _stability(int(q), float(curvature), float(noise), int(outlier_pct), settings)
    st.plotly_chart(build_stability_compare(stab), width="stretch", key="stability_plot")
    st.caption(
        f"Mittlere paarweise Abstände (PaCMAP / UMAP / t-SNE) je Datensatz: " + "; ".join(f"{r['pacmap']:.2f} / {r['umap']:.2f} / {r['tsne']:.2f}" for r in stab)
        + f". Über die vier Datensätze im Mittel: PaCMAP {np.mean([r['pacmap'] for r in stab]):.2f}, UMAP {np.mean([r['umap'] for r in stab]):.2f}, t-SNE {np.mean([r['tsne'] for r in stab]):.2f}."
    )

st.markdown("**⏱️ Rechenzeit**")
st.caption(
    "PaCMAP braucht je Iteration nur die festen Paarlisten (etwa n · 3.5 · n_neighbors Paare), keinen Graphen und keine spektrale Einbettung - dafür 450 Iterationen. UMAP baut zusätzlich einen Fuzzy-Graphen und startet "
    "spektral (dichte n×n-Eigenzerlegung); t-SNE berechnet je Iteration alle n² Paare."
)
if "timing_rows" not in st.session_state:
    if st.button("⏱️ Rechenzeit messen (ca. 15 s)", key="timing_start", help=f"Misst PaCMAP, UMAP, t-SNE, Isomap, LLE und PCA für n = {', '.join(str(n) for n in C.TIMING_NS)} auf diesem Rechner."):
        with st.spinner("Messe..."):
            st.session_state["timing_rows"] = timing_sweep()
        st.rerun()
else:
    rows = st.session_state["timing_rows"]
    st.plotly_chart(build_timing(rows), width="stretch", key="timing_chart")
    st.table({
        "Touren n": [r["n"] for r in rows],
        "PaCMAP": [f"{r['pacmap']:.2f} s" for r in rows],
        "UMAP": [f"{r['umap']:.2f} s" for r in rows],
        "t-SNE": [f"{r['tsne']:.2f} s" for r in rows],
        "Isomap": [f"{r['isomap']:.3f} s" for r in rows],
        "LLE": [f"{r['lle']:.3f} s" for r in rows],
        "PCA": [f"{r['pca'] * 1000:.2f} ms" for r in rows],
        "UMAP / PaCMAP": [f"{r['umap'] / max(r['pacmap'], 1e-9):.1f}×" for r in rows],
    })
    st.caption(f"Gemessen auf diesem Rechner (Wandzeit, ein Lauf je n, Standard-Einstellungen aller Verfahren): die Faktoren hängen von Rechner und Zwischenspeichern ab.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Paare.** Für Punkte $x_i \in \mathbb{R}^{12}$ (z-Werte, danach global auf $[0,1]$ skaliert und zentriert wie im Original) sind
- **nahe Paare** $(i, j)$: die $k$ (= `n_neighbors`) Touren $j$ mit dem kleinsten skalierten Abstand $\lVert x_i - x_j \rVert^2 / (\sigma_i \sigma_j)$ unter den $k + 50$ nächsten; $\sigma_i$ = Mittel der Abstände zum 4. bis 6. Nachbarn;
- **mittlere Paare**: aus 6 Zufallstouren die zweitnächste, $\lfloor r_{MN}\, k \rceil$ mal je Tour;
- **ferne Paare**: Zufallstouren, die weder $i$ noch nahe Paare von $i$ sind, $\lfloor r_{FP}\, k \rceil$ je Tour (einmal gezogen).

**Verlust.** Mit $\tilde d_{ij} = 1 + \lVert y_i - y_j \rVert^2$ und phasenabhängigen Gewichten

$$
L = w_{NB} \sum_{(i,j) \in NB} \frac{\tilde d_{ij}}{10 + \tilde d_{ij}} \;+\; w_{MN} \sum_{(i,j) \in MN} \frac{\tilde d_{ij}}{10000 + \tilde d_{ij}} \;+\; w_{FP} \sum_{(i,j) \in FP} \frac{1}{1 + \tilde d_{ij}} .
$$

Die Terme sättigen: der Beitrag eines nahen Paars steigt mit dem Abstand höchstens auf $w_{NB}$, der eines fernen fällt auf 0 - die Kraft je Paar ist beschränkt. Die Gradienten sind
$\frac{20\, w_{NB}}{(10 + \tilde d)^2}(y_i - y_j)$ (nah), $\frac{20000\, w_{MN}}{(10000 + \tilde d)^2}(y_i - y_j)$ (mittel) und $-\frac{2\, w_{FP}}{(1 + \tilde d)^2}(y_i - y_j)$ (fern); im Test per finite Differenzen geprüft.

**Gewichtsschema** (Standard: 100 / 100 / 250 Iterationen): Phase 1: $w_{MN}$ fällt linear von 1000 auf 3, $w_{NB} = 2$, $w_{FP} = 1$; Phase 2: $w_{MN} = 3$, $w_{NB} = 3$, $w_{FP} = 1$; Phase 3: $w_{MN} = 0$, $w_{NB} = 1$, $w_{FP} = 1$.
Optimiert wird mit Adam (Lernrate 1, $\beta = 0.9 / 0.999$), Start $0.01 \cdot$ PCA oder zufällig $\cdot\, 10^{-4}$.

**Neue Punkte.** Ein neuer Punkt bekommt nahe Paare zu seinen $k$ nächsten Trainings-Touren und wird mit dem $w_{NB}$-Schema (2 / 3 / 1) angezogen, während die Trainings-Einbettung fest bleibt.

**Grenzen.** (1) *Keine belastbare globale Struktur*: der Verlust hat nur die Paare als Information; Größe und Reihenfolge ferner Abstände sind nicht erzwungen (Demo: Sonderfahrten, Abstandstreue ferner Paare unter UMAP). (2) *Zufall in den Paaren*: mittlere und ferne Paare sind
Zufallsziehungen - die Einbettung hängt von ihnen und vom Start ab (Demo: Stabilität). (3) *Kein Konvergenzkriterium*: nach der letzten Phase wird abgebrochen; mehr Iterationen verbessern die globale Ordnung nicht - sie ist meist nach Phase 1 am höchsten (Demo: Schritt 3).
(4) *Nicht-parametrisch*: keine Abbildung für neue Punkte (Demo: Behelf). (5) Exaktes kNN, $O(n^2)$ - für große Datensätze gibt es Näherungen, die hier nicht gebaut sind. (6) Achsen und Abstände im Bild haben keine feste Bedeutung.

**Trustworthiness** (Venna & Kaski, 2001): $T = 1 - \frac{2}{nk(2n - 3k - 1)} \sum_i \sum_{j \in U_i} (r(i,j) - k)$ mit $U_i$ = Nachbarn in der Einbettung, die im Originalraum keine sind, und $r(i,j)$ ihrem Originalrang.
**Abstandstreue** = Pearson-Korrelation der Paarabstände der 2-D-Einbettung mit den Paarabständen der wahren Faktoren (nah/fern: untere/obere Hälfte der wahren Abstände). **Procrustes-Abstand**: $1 - (\sum s_i)^2$ mit $s_i$ den
Singulärwerten von $A^\top B$ nach Zentrierung und Normierung beider Einbettungen (wie `scipy.spatial.procrustes`).

Implementiert in `pacmap_algorithm.py` (Paare, Verlust, Gewichtsschema, Adam, Transform), `pacmap_umap.py` / `pacmap_tsne.py` / `pacmap_isomap.py` / `pacmap_lle.py` (Vergleichsverfahren, wortgleich aus umap-demo / tsne-demo / isomap-demo / lle-demo),
`pacmap_scenario.py` (Lieferrouten-Generator, wortgleich aus pca-demo) und `pacmap_evaluation.py` (Kennzahlen, Sweeps, Verdict, Stabilität, Out-of-sample, Zeitmessung).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Dimensionsreduktion: von PCA bis Autoencoder](https://sebastianhanisch.net/konzepte-dimensionsreduktion.html)."
)
