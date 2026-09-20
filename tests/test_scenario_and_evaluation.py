import numpy as np
import pytest
from sklearn.manifold import trustworthiness as sk_trustworthiness

import pacmap_constants as C
from pacmap_evaluation import (
    Settings, analyse, convergence_rows, distance_fidelity, distance_fidelity_split, fp_ratio_sweep, iterations_sweep, make_dataset, mn_ratio_sweep, n_neighbors_sweep, out_of_sample,
    pca_project, r2_quadratic, run_pacmap, stability, timing_sweep, trustworthiness, verdict,
)
from pacmap_umap import fit_umap


def _far(embedding, ds):
    return distance_fidelity_split(embedding, ds.z)[1]


def test_trustworthiness_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 6))
    for embedding in (X[:, :2], rng.standard_normal((80, 2)), pca_project(X)):
        assert abs(trustworthiness(X, embedding, 8) - sk_trustworthiness(X, embedding, n_neighbors=8)) < 1e-9


def test_r2_quadratic_recovers_monotone_reparametrisations_and_rejects_noise():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((200, 2))
    coords = np.column_stack([z[:, 0] + 0.3 * z[:, 0] ** 2, z[:, 1] + 0.2 * z[:, 1] ** 2])
    assert r2_quadratic(coords, z) > 0.9
    assert r2_quadratic(rng.standard_normal((200, 2)), z) < 0.1


def test_distance_fidelity_is_one_for_a_scaled_copy_and_split_reports_near_and_far():
    rng = np.random.default_rng(2)
    z = rng.standard_normal((100, 2))
    assert distance_fidelity(3.0 * z, z) > 0.999999
    near, far = distance_fidelity_split(3.0 * z, z)
    assert near > 0.999999 and far > 0.999999
    squashed = z * (1.0 / (1.0 + np.linalg.norm(z, axis=1, keepdims=True) ** 2))
    near2, far2 = distance_fidelity_split(squashed, z)
    assert near2 > far2


def test_analysis_on_the_default_surface_has_the_expected_structure():
    ds = make_dataset(300, 2, 1.0, 0.25, 0, 7)
    a = analyse(ds, Settings())
    m = a.metrics
    assert 0.65 < m["pacmap"]["r2"] < 0.96 and m["umap"]["r2"] > 0.8 and m["tsne"]["r2"] > 0.88 and m["isomap"]["r2"] > 0.94 and m["pca"]["r2"] < 0.6
    assert a.ref is None and "ref" not in m and len(a.iso_indices) == 300
    assert sorted(a.snapshot_r2) == [i for i in sorted(a.pacmap.snapshots) if i > 0] == sorted(a.snapshot_far)
    assert convergence_rows(a)[-1][0] == 450 and abs(convergence_rows(a)[-1][1] - m["pacmap"]["r2"]) < 1e-12


def test_fewer_pairs_are_compared_against_a_reference_run_with_the_standard_pairs():
    ds = make_dataset(300, 2, 1.0, 0.25, 0, 7)
    a = analyse(ds, Settings(n_neighbors=2))
    assert a.ref is not None and (a.ref.n_neighbors, a.ref.n_mn) == (10, 5) and a.metrics["ref"]["r2"] - a.metrics["pacmap"]["r2"] >= 0.1
    assert analyse(ds, Settings(n_neighbors=20)).ref is None and analyse(ds, Settings(fp_ratio=4.0)).ref is None


def test_pacmap_does_not_beat_umap_on_global_structure_over_four_datasets():
    """Belegt die Tabelle in der App: ferne Paare im Mittel PaCMAP 0.34 gegen UMAP 0.60 (Aussage 'bessere globale Struktur' nicht bestätigt)."""
    far_p, far_u, r2_p = [], [], []
    for seed in range(100_000, 100_004):
        ds = make_dataset(300, 2, 1.0, 0.25, 0, seed)
        p, u = run_pacmap(ds.X, Settings()).embedding, fit_umap(ds.X, C.UMAP_N_NEIGHBORS, C.UMAP_MIN_DIST, C.UMAP_N_EPOCHS).embedding
        far_p.append(_far(p, ds)), far_u.append(_far(u, ds)), r2_p.append(r2_quadratic(p, ds.z))
    assert np.mean(far_p) < np.mean(far_u) - 0.1 and np.mean(r2_p) > 0.75


def test_sonderfahrten_break_pacmap_over_four_datasets():
    r2_p, r2_pca, far_p, far_pca = [], [], [], []
    for seed in range(100_000, 100_004):
        ds = make_dataset(300, 2, 1.0, 0.25, 5, seed)
        p = run_pacmap(ds.X, Settings()).embedding
        pc = pca_project(ds.X)
        r2_p.append(r2_quadratic(p, ds.z)), r2_pca.append(r2_quadratic(pc, ds.z)), far_p.append(_far(p, ds)), far_pca.append(_far(pc, ds))
    assert np.mean(r2_p) < 0.5 < 0.6 < np.mean(r2_pca) and np.mean(far_p) < 0.3 < 0.8 < np.mean(far_pca)


def test_without_mid_near_pairs_the_far_pair_fidelity_collapses():
    """Belegt den Mechanismus (Tabelle in der App): ohne MN-Paare ferne Paare im Mittel 0.06 statt 0.33 (3 Datensätze)."""
    with_mn, without = [], []
    for seed in range(100_000, 100_003):
        ds = make_dataset(300, 2, 1.0, 0.25, 0, seed)
        with_mn.append(_far(run_pacmap(ds.X, Settings()).embedding, ds))
        without.append(_far(run_pacmap(ds.X, Settings(mn_ratio=0.0)).embedding, ds))
    assert np.mean(without) < np.mean(with_mn) - 0.1


def test_far_pair_fidelity_is_highest_after_phase_one_and_erodes_afterwards():
    """Belegt Schritt 3 der App: im Mittel über 6 Datensätze (300 Touren) 0.53 nach Phase 1 (Iteration 100) gegen 0.36 am Ende."""
    at_100, at_end = [], []
    for seed in range(100_000, 100_006):
        ds = make_dataset(300, 2, 1.0, 0.25, 0, seed)
        m = run_pacmap(ds.X, Settings())
        at_100.append(_far(m.snapshots[100], ds)), at_end.append(_far(m.embedding, ds))
    assert np.mean(at_100) > np.mean(at_end) + 0.08


@pytest.mark.parametrize("q,curv,noise,out,settings,code", [
    (2, 1.0, 0.25, 0, Settings(), "pacmap_wins"),
    (2, 1.0, 0.25, 0, Settings(mn_ratio=0.0), "mn_missing"),
    (2, 1.0, 0.25, 5, Settings(), "global_structure"),
    (2, 1.0, 0.25, 0, Settings(n_neighbors=2), "neighbors_small"),
    (2, 1.0, 0.8, 0, Settings(), "pacmap_wins"),
    (2, 0.0, 0.25, 0, Settings(), "no_advantage"),
])
def test_verdict_codes(q, curv, noise, out, settings, code):
    ds = make_dataset(300, q, curv, noise, out, 7)
    assert verdict(analyse(ds, settings), ds, settings)[1] == code


def test_sweeps_are_deterministic_and_show_the_effects():
    kw = dict(n_tours=120, seeds=(100_000, 100_001))
    a = n_neighbors_sweep(2, 1.0, 0.25, 0, values=(2, 10), **kw)
    assert a == n_neighbors_sweep(2, 1.0, 0.25, 0, values=(2, 10), **kw) and a[0]["r2"] < a[1]["r2"]
    mn = mn_ratio_sweep(2, 1.0, 0.25, 0, values=(0.0, 0.5), **kw)
    fp = fp_ratio_sweep(2, 1.0, 0.25, 0, values=(0.5, 2.0), **kw)
    it = iterations_sweep(2, 1.0, 0.25, 0, values=(90, 180), **kw)
    assert [r["mn_ratio"] for r in mn] == [0.0, 0.5] and [r["fp_ratio"] for r in fp] == [0.5, 2.0] and [r["n_iter"] for r in it] == [90, 180]
    assert all(np.isfinite(r["r2"]) and np.isfinite(r["far"]) for r in mn + fp + it)


def test_sweep_and_stability_seeds_are_separate_from_demo_seeds():
    assert min(C.SWEEP_SEEDS) >= 100_000 > C.DEFAULT_SEED and min(C.STABILITY_SEEDS) >= 100_000


def test_stability_is_measured_fairly_on_the_same_datasets_and_umap_and_pacmap_beat_tsne():
    """Belegt die Tabelle in der App (q = 2, 200 Touren, 4 Datensätze): t-SNE weicht bei zufälligen Starts am stärksten ab, PaCMAP und UMAP weniger."""
    rows = stability(2, 1.0, 0.25, 0)
    assert [r["seed"] for r in rows] == list(C.STABILITY_SEEDS) and all(len(r["embeddings"]) == 4 for r in rows)
    pac, um, ts = (np.mean([r[k] for r in rows]) for k in ("pacmap", "umap", "tsne"))
    assert pac < ts - 0.05 and um < ts - 0.05


def test_out_of_sample_behelf_is_close_to_but_not_better_than_umap_transform():
    ds = make_dataset(300, 2, 1.0, 0.25, 0, 100_000)
    oos = out_of_sample(ds, Settings())
    assert len(oos["test"]) == 60 and oos["test"][0] == 240 and oos["y_pacmap"].shape == (60, 2) and oos["y_umap"].shape == (60, 2) and oos["y_tsne"].shape == (60, 2)
    assert oos["r2_pacmap"] > 0.6 and oos["r2_umap"] > 0.8 and 0.0 <= oos["shift_pacmap"] <= 1.0


def test_timing_sweep_has_the_expected_shape_and_pacmap_is_the_fastest_embedding_at_large_n():
    rows = timing_sweep(ns=(100, 600))
    assert [r["n"] for r in rows] == [100, 600] and all(r[key] > 0 for r in rows for key in ("pacmap", "umap", "tsne", "isomap", "lle", "pca"))
    assert rows[1]["pacmap"] < rows[1]["umap"] < rows[1]["tsne"]
