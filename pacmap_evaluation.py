"""Auswertung: findet PaCMAP die wahren Faktoren zurück - und löst es sein Versprechen gegenüber UMAP ein (bessere globale Struktur, ausgewogene lokal-global-Balance, Geschwindigkeit)? Alle Kennzahlen werden
am Datensatz gemessen; die wahren latenten Faktoren z sind bekannt (Lieferrouten-Erzeugung, inkl. Sonderfahrten). PaCMAP, UMAP, t-SNE, Isomap, LLE (Vergleichsverfahren mit ihren guten Einstellungen) und PCA werden
mit denselben Messungen bewertet.

- **R² der wahren Faktoren**: Rekonstruktion von z aus den zwei Koordinaten per quadratischer Regression (monotone Umparametrisierungen werden nicht bestraft).
- **Abstandstreue** (gesamt / nahe Paare / ferne Paare): Pearson-Korrelation der Paarabstände in der Einbettung mit den Paarabständen der wahren Faktoren; "nah" = untere Hälfte der wahren Paarabstände,
  "fern" = obere Hälfte.
- **Trustworthiness** (Venna & Kaski): bleiben Nachbarn Nachbarn?
- **Procrustes-Abstand**: wie stark unterscheiden sich zwei Einbettungen derselben Touren nach bester Drehung/Spiegelung/Skalierung (0 = gleich, 1 = unabhängig)."""

import itertools
import time
from dataclasses import dataclass

import numpy as np

import pacmap_constants as C
from pacmap_algorithm import fit_pacmap, transform
from pacmap_isomap import fit_isomap, pairwise_distances, standardize
from pacmap_lle import fit_lle
from pacmap_scenario import generate_dataset
from pacmap_tsne import embed_new_naive, fit_tsne, procrustes_disparity
from pacmap_umap import fit_umap
from pacmap_umap import transform as umap_transform


def trustworthiness(X_high, X_low, n_neighbors=C.TRUST_NEIGHBORS):
    """Trustworthiness (Venna & Kaski, 2001): Anteil der Nachbarn im Einbettungsraum, die auch im Originalraum echte Nachbarn sind, mit
    Rang-Strafe für eingeschleppte Fremde. 1 = perfekt. Eigene Implementierung, gegen sklearn geprüft (nur im Test)."""
    n = len(X_high)
    k = n_neighbors
    d_high = np.linalg.norm(X_high[:, None, :] - X_high[None, :, :], axis=-1)
    d_low = np.linalg.norm(X_low[:, None, :] - X_low[None, :, :], axis=-1)
    np.fill_diagonal(d_high, np.inf)
    np.fill_diagonal(d_low, np.inf)
    ranks_high = np.argsort(np.argsort(d_high, axis=1), axis=1) + 1            # Rang 1 = nächster Nachbar
    neighbors_low = np.argsort(d_low, axis=1)[:, :k]
    penalty = 0.0
    for i in range(n):
        r = ranks_high[i, neighbors_low[i]]
        penalty += float(np.maximum(r - k, 0).sum())
    return 1.0 - 2.0 / (n * k * (2 * n - 3 * k - 1)) * penalty


def _quad_features(e):
    return np.column_stack([e[:, 0], e[:, 1], e[:, 0] ** 2, e[:, 0] * e[:, 1], e[:, 1] ** 2, np.ones(len(e))])


def r2_quadratic(coords2, z):
    """R² der Rekonstruktion von z aus zwei Koordinaten (quadratische Regression, Mittel über die Faktoren, gewichtet mit ihrer Varianz)."""
    A = _quad_features(coords2)
    beta, *_ = np.linalg.lstsq(A, z, rcond=None)
    return float(1.0 - (z - A @ beta).var(0).sum() / z.var(0).sum())


def pca_project(X, n_components=2):
    Z = standardize(X)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ vt[:n_components].T


def distance_fidelity(coords2, z):
    iu = np.triu_indices(len(z), 1)
    return float(np.corrcoef(pairwise_distances(coords2)[iu], pairwise_distances(z)[iu])[0, 1])


def distance_fidelity_split(coords2, z):
    """(nahe Paare, ferne Paare): Abstandstreue getrennt für die untere und die obere Hälfte der wahren Paarabstände."""
    iu = np.triu_indices(len(z), 1)
    dz = pairwise_distances(z)[iu]
    dc = pairwise_distances(coords2)[iu]
    near = dz <= np.median(dz)
    return float(np.corrcoef(dz[near], dc[near])[0, 1]), float(np.corrcoef(dz[~near], dc[~near])[0, 1])


def make_dataset(n_tours, q, curvature, noise, outlier_pct, seed):
    return generate_dataset(n_tours, q, curvature, noise, outlier_pct, seed)


def _metrics(coords, z, Z):
    near, far = distance_fidelity_split(coords, z)
    return {"r2": r2_quadratic(coords, z), "fid": distance_fidelity(coords, z), "near": near, "far": far, "trust": trustworthiness(Z, coords)}


@dataclass(frozen=True)
class Settings:
    n_neighbors: int = C.DEFAULT_N_NEIGHBORS
    mn_ratio: float = C.DEFAULT_MN_RATIO
    fp_ratio: float = C.DEFAULT_FP_RATIO
    n_iter: int = C.DEFAULT_N_ITER
    init: str = C.DEFAULT_INIT                       # "pca" oder "random:<Start>"


def _with(settings, **changes):
    return Settings(**{**settings.__dict__, **changes})


def run_pacmap(X, s, seed_offset=0):
    kind, _, start = s.init.partition(":")
    return fit_pacmap(X, s.n_neighbors, s.mn_ratio, s.fp_ratio, s.n_iter, kind, (int(start) if start else 0) + seed_offset)


def _is_reference(s):
    return s.n_neighbors >= C.DEFAULT_N_NEIGHBORS and s.mn_ratio >= C.DEFAULT_MN_RATIO


@dataclass(frozen=True)
class Analysis:
    pacmap: object
    ref: object                      # Referenzlauf mit den Standard-Paaren (k = 10, MN 0.5), wenn die Einstellungen weniger Paare nutzen; sonst None
    umap: object
    tsne: object
    isomap: object
    lle: object
    iso_2d: np.ndarray
    pca_2d: np.ndarray
    iso_indices: np.ndarray
    metrics: dict                    # {"pacmap", "umap", "tsne", "isomap", "lle", "pca", "ref"?} -> {"r2","fid","near","far","trust"}
    snapshot_r2: dict                # Iteration -> R² der Einbettung zu diesem Zeitpunkt
    snapshot_far: dict               # Iteration -> Abstandstreue ferner Paare zu diesem Zeitpunkt


def analyse(dataset, settings):
    Z = standardize(dataset.X)
    model = run_pacmap(dataset.X, settings)
    umap = fit_umap(dataset.X, C.UMAP_N_NEIGHBORS, C.UMAP_MIN_DIST, C.UMAP_N_EPOCHS)
    tsne = fit_tsne(dataset.X, C.TSNE_PERPLEXITY, C.TSNE_N_ITER)
    iso = fit_isomap(dataset.X, C.ISOMAP_K, 2)
    lle = fit_lle(dataset.X, C.LLE_K, 2, C.LLE_REG)
    pca2 = pca_project(dataset.X)
    iso2 = iso.embedding[:, :2]
    metrics = {
        "pacmap": _metrics(model.embedding, dataset.z, Z),
        "umap": _metrics(umap.embedding, dataset.z, Z),
        "tsne": _metrics(tsne.embedding, dataset.z, Z),
        "isomap": _metrics(iso2, dataset.z[iso.indices], Z[iso.indices]),
        "lle": _metrics(lle.embedding[:, :2], dataset.z, Z),
        "pca": _metrics(pca2, dataset.z, Z),
    }
    ref = None
    if not _is_reference(settings):
        ref = run_pacmap(dataset.X, _with(settings, n_neighbors=max(settings.n_neighbors, C.DEFAULT_N_NEIGHBORS), mn_ratio=max(settings.mn_ratio, C.DEFAULT_MN_RATIO)))
        metrics["ref"] = _metrics(ref.embedding, dataset.z, Z)
    snap = {it: r2_quadratic(y, dataset.z) for it, y in model.snapshots.items() if it > 0}
    snap_far = {it: distance_fidelity_split(y, dataset.z)[1] for it, y in model.snapshots.items() if it > 0}
    return Analysis(pacmap=model, ref=ref, umap=umap, tsne=tsne, isomap=iso, lle=lle, iso_2d=iso2, pca_2d=pca2, iso_indices=iso.indices, metrics=metrics, snapshot_r2=snap, snapshot_far=snap_far)


def verdict(analysis, dataset, settings):
    """Verdict-Kaskade (Warnungen zuerst) -> (Stufe, Code, Daten). Jede Warnung stützt sich auf einen Referenzlauf mit den Standard-Paaren oder auf die Vergleichsverfahren."""
    m = analysis.metrics
    p = m["pacmap"]
    data = {"r2": p["r2"], "r2_umap": m["umap"]["r2"], "r2_tsne": m["tsne"]["r2"], "r2_iso": m["isomap"]["r2"], "r2_lle": m["lle"]["r2"], "r2_pca": m["pca"]["r2"], "far": p["far"],
            "far_umap": m["umap"]["far"], "far_tsne": m["tsne"]["far"], "far_pca": m["pca"]["far"], "far_iso": m["isomap"]["far"], "trust": p["trust"], "n_neighbors": settings.n_neighbors,
            "n_mn": analysis.pacmap.n_mn, "n_fp": analysis.pacmap.n_fp, "n_iter": settings.n_iter, "outlier_pct": dataset.outlier_pct}
    if "ref" in m:
        data.update({"r2_ref": m["ref"]["r2"], "far_ref": m["ref"]["far"], "trust_ref": m["ref"]["trust"]})
        if settings.n_neighbors < C.DEFAULT_N_NEIGHBORS and m["ref"]["r2"] - p["r2"] >= 0.10:
            return "warning", "neighbors_small", data
        if analysis.pacmap.n_mn == 0 and m["ref"]["far"] - p["far"] >= 0.15:
            return "warning", "mn_missing", data
    if dataset.outlier_pct > 0 and m["pca"]["far"] - p["far"] >= 0.25 and m["pca"]["r2"] - p["r2"] >= 0.15:
        return "warning", "global_structure", data
    if dataset.curvature == 0 and p["r2"] - m["pca"]["r2"] < 0.03:
        return "info", "no_advantage", data
    if p["r2"] - m["pca"]["r2"] >= 0.10:
        return "success", "pacmap_wins", data
    return "info", "neutral", data


def _sweep(values, key, q, curvature, noise, outlier_pct, n_tours, seeds, base):
    rows = []
    for value in values:
        acc = {"r2": [], "far": [], "trust": []}
        for seed in seeds:
            ds = make_dataset(n_tours, q, curvature, noise, outlier_pct, seed)
            model = run_pacmap(ds.X, _with(base, **{key: value}))
            m = _metrics(model.embedding, ds.z, standardize(ds.X))
            acc["r2"].append(m["r2"]), acc["far"].append(m["far"]), acc["trust"].append(m["trust"])
        rows.append({key: value, **{k: float(np.mean(v)) for k, v in acc.items()}})
    return rows


def n_neighbors_sweep(q, curvature, noise, outlier_pct, n_tours=C.SWEEP_N_TOURS, values=C.SWEEP_N_NEIGHBORS, seeds=C.SWEEP_SEEDS):
    """Feste Sweep-Seeds, sonst Standard-Einstellungen: je n_neighbors mittleres R², Abstandstreue (ferne Paare) und Trustworthiness."""
    return _sweep(values, "n_neighbors", q, curvature, noise, outlier_pct, n_tours, seeds, Settings())


def mn_ratio_sweep(q, curvature, noise, outlier_pct, n_tours=C.SWEEP_N_TOURS, values=C.SWEEP_MN_RATIOS, seeds=C.SWEEP_SEEDS):
    """Wie `n_neighbors_sweep`, aber über das Verhältnis der mittleren Paare (MN)."""
    return _sweep(values, "mn_ratio", q, curvature, noise, outlier_pct, n_tours, seeds, Settings())


def fp_ratio_sweep(q, curvature, noise, outlier_pct, n_tours=C.SWEEP_N_TOURS, values=C.SWEEP_FP_RATIOS, seeds=C.SWEEP_SEEDS):
    """Wie `n_neighbors_sweep`, aber über das Verhältnis der fernen Paare (FP)."""
    return _sweep(values, "fp_ratio", q, curvature, noise, outlier_pct, n_tours, seeds, Settings())


def iterations_sweep(q, curvature, noise, outlier_pct, n_tours=C.SWEEP_N_TOURS, values=(90, 180, 270, 450, 900), seeds=C.SWEEP_SEEDS):
    """Wie `n_neighbors_sweep`, aber über die Zahl der Iterationen (die drei Phasen wachsen proportional)."""
    return _sweep(values, "n_iter", q, curvature, noise, outlier_pct, n_tours, seeds, Settings())


def convergence_rows(analysis):
    """(Iteration, R²) an den Schnappschüssen des aktuellen Laufs - ohne Extra-Rechnung."""
    return sorted(analysis.snapshot_r2.items())


def _median_pairwise(embeddings):
    return float(np.median([procrustes_disparity(a, b) for a, b in itertools.combinations(embeddings, 2)]))


def stability(q, curvature, noise, outlier_pct, settings=None, n_tours=200, seeds=C.STABILITY_SEEDS, inits=C.STABILITY_INITS):
    """Fairer Vergleich auf denselben Datensätzen: je Datensatz vier zufällige Starts, Median der paarweisen Procrustes-Abstände für PaCMAP, UMAP und t-SNE. -> Liste je Datensatz."""
    base = settings or Settings()
    out = []
    for seed in seeds:
        ds = make_dataset(n_tours, q, curvature, noise, outlier_pct, seed)
        pac, um, ts = [], [], []
        for i, init in enumerate(inits):
            pac.append(run_pacmap(ds.X, _with(base, init=init)).embedding)
            um.append(fit_umap(ds.X, C.UMAP_N_NEIGHBORS, C.UMAP_MIN_DIST, C.UMAP_N_EPOCHS, 5, "random", i).embedding)
            ts.append(fit_tsne(ds.X, C.TSNE_PERPLEXITY, C.TSNE_N_ITER, init="random", seed=i).embedding)
        out.append({"seed": seed, "pacmap": _median_pairwise(pac), "umap": _median_pairwise(um), "tsne": _median_pairwise(ts), "embeddings": pac})
    return out


def out_of_sample(dataset, settings, fraction=C.HOLDOUT_FRACTION):
    """Die letzten `fraction` der Touren zurückhalten: PaCMAP-Behelf `transform` gegen UMAP-`transform` und die t-SNE-Näherung; dazu die Verschiebung der Trainings-Touren beim Neu-Rechnen."""
    n = dataset.n
    n_test = max(10, int(round(fraction * n)))
    train, test = np.arange(n - n_test), np.arange(n - n_test, n)
    z_test = dataset.z[test]

    def r2_new(train_emb, y):
        beta, *_ = np.linalg.lstsq(_quad_features(train_emb), dataset.z[train], rcond=None)
        resid = z_test - _quad_features(y) @ beta                                                          # nicht zentrieren: ein konstanter Versatz der Vorhersage zählt als Fehler
        return float(1 - (resid ** 2).sum() / ((z_test - z_test.mean(0)) ** 2).sum())
    pac_train = run_pacmap(dataset.X[train], settings)
    y_pac = transform(pac_train, dataset.X[test])
    pac_full = run_pacmap(dataset.X, settings)
    um_train = fit_umap(dataset.X[train], C.UMAP_N_NEIGHBORS, C.UMAP_MIN_DIST, C.UMAP_N_EPOCHS)
    y_um = umap_transform(um_train, dataset.X[test])
    ts_train = fit_tsne(dataset.X[train], C.TSNE_PERPLEXITY, C.TSNE_N_ITER)
    y_ts = embed_new_naive(ts_train, dataset.X[test], 10)
    return {"train": train, "test": test, "model": pac_train, "umap_train": um_train.embedding, "tsne_train": ts_train.embedding, "y_pacmap": y_pac, "y_umap": y_um, "y_tsne": y_ts,
            "r2_pacmap": r2_new(pac_train.embedding, y_pac), "r2_umap": r2_new(um_train.embedding, y_um), "r2_tsne": r2_new(ts_train.embedding, y_ts),
            "r2_train": r2_quadratic(pac_train.embedding, dataset.z[train]), "shift_pacmap": procrustes_disparity(pac_full.embedding[train], pac_train.embedding)}


def timing_sweep(ns=C.TIMING_NS, seed=100_000):
    """Gemessene Rechenzeit (Sekunden) von PaCMAP, UMAP, t-SNE, Isomap, LLE und PCA für wachsende n (eigene Messung, Rechner-abhängig)."""
    rows = []
    for n in ns:
        ds = generate_dataset(n, 2, 0.0, 0.1, 0, seed)
        out = {"n": int(n)}
        for name, fn in (("pacmap", lambda: fit_pacmap(ds.X, C.DEFAULT_N_NEIGHBORS, C.DEFAULT_MN_RATIO, C.DEFAULT_FP_RATIO, C.DEFAULT_N_ITER)),
                         ("umap", lambda: fit_umap(ds.X, C.UMAP_N_NEIGHBORS, C.UMAP_MIN_DIST, C.UMAP_N_EPOCHS)), ("tsne", lambda: fit_tsne(ds.X, C.TSNE_PERPLEXITY, C.TSNE_N_ITER)),
                         ("isomap", lambda: fit_isomap(ds.X, C.ISOMAP_K, 2)), ("lle", lambda: fit_lle(ds.X, C.LLE_K, 2, C.LLE_REG)), ("pca", lambda: pca_project(ds.X))):
            t = time.perf_counter()
            fn()
            out[name] = time.perf_counter() - t
        rows.append(out)
    return rows
