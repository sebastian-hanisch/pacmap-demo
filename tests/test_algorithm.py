import numpy as np
import pacmap as pacmap_lib
import pytest
from pacmap import pacmap as pmm

from pacmap_algorithm import (
    adam_step, decide_num_pairs, find_weight, fit_pacmap, further_pairs, mid_near_pairs, neighbour_pairs, pair_gradient, phase_iterations, preprocess, snapshot_iterations, transform,
)
from pacmap_isomap import fit_isomap, pairwise_distances, standardize
from pacmap_lle import SingularNeighbourhood, fit_lle
from pacmap_scenario import generate_dataset
from pacmap_tsne import fit_tsne, procrustes_disparity
from pacmap_umap import fit_umap


def _data(n=300, q=2, curvature=1.0, noise=0.25, seed=7):
    return generate_dataset(n, q, curvature, noise, 0, seed)


def _pairs_setup(n=150, k=10, mn=0.5, fp=2.0, seed=0):
    Z = standardize(_data(n).X)
    Xp = preprocess(Z)[0]
    dist = pairwise_distances(Xp)
    k, n_mn, n_fp = decide_num_pairs(n, k, mn, fp)
    rng = np.random.default_rng(seed)
    nb = neighbour_pairs(dist, k)
    return Z, Xp, dist, k, n_mn, n_fp, nb, mid_near_pairs(dist, n_mn, rng), further_pairs(n, n_fp, nb, k, rng)


def test_phase_iterations_and_weight_schedule_match_the_reference():
    assert phase_iterations(450) == (100, 100, 250) and phase_iterations(90) == (20, 20, 50) and sum(phase_iterations(900)) == 900
    iters = (100, 100, 250)
    for itr in (0, 1, 50, 99, 100, 150, 199, 200, 449):
        assert np.allclose(find_weight(itr, iters), pmm.find_weight(1000.0, itr, num_iters=iters))
    assert find_weight(0, iters)[0] == 1000.0 and find_weight(200, iters)[0] == 0.0


def test_preprocess_is_global_min_max_then_centred():
    Z = standardize(_data(100).X)
    Xp, xmin, xmax, xmean = preprocess(Z)
    assert abs(Xp.mean(0)).max() < 1e-12 and abs((Xp + xmean).min()) < 1e-12 and abs((Xp + xmean).max() - 1.0) < 1e-12 and abs(xmin - Z.min()) < 1e-12


@pytest.mark.parametrize("n,k,mn,fp", [(300, 10, 0.5, 2.0), (100, 10, 0.5, 2.0), (100, 14, 2.0, 4.0), (100, 40, 0.5, 2.0)])
def test_pair_counts_match_the_reference(n, k, mn, fp):
    ref = pacmap_lib.PaCMAP(n_neighbors=k, MN_ratio=mn, FP_ratio=fp)
    ref.decide_num_pairs(n)
    assert decide_num_pairs(n, k, mn, fp) == (ref.n_neighbors, ref.n_MN, ref.n_FP)


def test_near_pairs_are_the_k_nearest_under_scaled_distance():
    Z, Xp, dist, k, n_mn, n_fp, nb, mn_pairs, fp_pairs = _pairs_setup()
    n = len(Z)
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    order = np.argsort(d, axis=1)[:, :60]
    kd = np.take_along_axis(dist, order, axis=1)
    sig = np.maximum(kd[:, 3:6].mean(1), 1e-10)
    scaled = kd ** 2 / sig[:, None] / sig[order]
    expected = np.take_along_axis(order, np.argsort(scaled, axis=1)[:, :k], axis=1)
    assert nb.shape == (n * k, 2) and (nb[:, 0] == np.repeat(np.arange(n), k)).all()
    assert all(set(nb[nb[:, 0] == i, 1]) == set(expected[i]) for i in range(n))


def test_near_pairs_agree_with_the_reference_implementation():
    n = 200
    Z = standardize(_data(n).X)
    ref = pacmap_lib.PaCMAP(n_neighbors=10, random_state=0, save_tree=False).fit(Z.astype(np.float32), init="pca")
    Xp = preprocess(Z)[0]
    mine = neighbour_pairs(pairwise_distances(Xp), 10)
    ours = set(map(tuple, mine.tolist()))
    theirs = set(map(tuple, ref.pair_neighbors.tolist()))
    assert len(ours & theirs) / len(theirs) > 0.9                   # die Referenz sucht näherungsweise, wir exakt


def test_mid_near_and_further_pairs_have_the_documented_structure():
    Z, Xp, dist, k, n_mn, n_fp, nb, mn, fp = _pairs_setup()
    n = len(Z)
    assert mn.shape == (n * n_mn, 2) and fp.shape == (n * n_fp, 2)
    for pairs, count in ((mn, n_mn), (fp, n_fp)):
        assert (pairs[:, 0] != pairs[:, 1]).all()
        assert all(len(set(pairs[pairs[:, 0] == i, 1])) == count for i in range(n))                  # je Tour verschiedene Partner
    nb_sets = [set(nb[nb[:, 0] == i, 1]) for i in range(n)]
    assert not any(j in nb_sets[i] for i, j in fp)                                                     # ferne Paare sind keine nahen Paare
    d = lambda p: dist[p[:, 0], p[:, 1]].mean()
    assert d(nb) < d(mn) < d(fp)                                                                       # nah < mittel < fern (im Original)


def test_pair_sampling_is_deterministic_for_a_fixed_seed_and_differs_between_seeds():
    a, b, c = _pairs_setup(seed=0), _pairs_setup(seed=0), _pairs_setup(seed=1)
    assert np.array_equal(a[-2], b[-2]) and np.array_equal(a[-1], b[-1]) and not np.array_equal(a[-1], c[-1])


def test_pair_gradient_matches_the_reference_gradient_and_finite_differences():
    Z, Xp, dist, k, n_mn, n_fp, nb, mn, fp = _pairs_setup(n=80)
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(80, 2))
    w = (3.0, 500.0, 1.0)                                                                             # (w_nb, w_mn, w_fp)
    grad, losses = pair_gradient(Y, nb, mn, fp, *w)
    ref = pmm.pacmap_grad(Y.astype(np.float32), nb.astype(np.int32), mn.astype(np.int32), fp.astype(np.int32), np.float32(w[0]), np.float32(w[1]), np.float32(w[2]))
    assert np.allclose(grad, ref[:-1], rtol=1e-3, atol=1e-3) and abs(losses.sum() - ref[-1, 0]) < 1e-2 * abs(ref[-1, 0])

    def total(Yt):
        return pair_gradient(Yt, nb, mn, fp, *w)[1].sum()
    for i, c in ((0, 0), (10, 1), (55, 0)):
        e = np.zeros_like(Y)
        e[i, c] = 1e-6
        assert abs((total(Y + e) - total(Y - e)) / 2e-6 - grad[i, c]) < 1e-4 * max(1.0, abs(grad[i, c]))


def test_adam_step_matches_the_reference_update():
    rng = np.random.default_rng(4)
    Y, grad = rng.normal(size=(20, 2)).astype(np.float32), rng.normal(size=(20, 2)).astype(np.float32)
    m, v = rng.normal(size=(20, 2)).astype(np.float32) * 0.1, np.abs(rng.normal(size=(20, 2))).astype(np.float32) * 0.1
    ref_Y, ref_m, ref_v = Y.copy(), m.copy(), v.copy()
    pmm.update_embedding_adam(ref_Y, grad, ref_m, ref_v, np.float32(0.9), np.float32(0.999), np.float32(1.0), 7)
    mine_Y, mine_m, mine_v = Y.astype(float), m.astype(float), v.astype(float)
    adam_step(mine_Y, grad.astype(float), mine_m, mine_v, 7)
    assert np.allclose(mine_Y, ref_Y, atol=1e-5) and np.allclose(mine_m, ref_m, atol=1e-6) and np.allclose(mine_v, ref_v, atol=1e-6)


def test_fit_is_deterministic_records_snapshots_losses_and_weights():
    ds = _data(150)
    m1, m2 = fit_pacmap(ds.X, 10, 0.5, 2.0, 180), fit_pacmap(ds.X, 10, 0.5, 2.0, 180)
    assert np.array_equal(m1.embedding, m2.embedding) and not np.array_equal(m1.embedding, fit_pacmap(ds.X, 10, 0.5, 2.0, 180, seed=3).embedding)
    assert m1.iters == (40, 40, 100) and m1.n_iter == 180 and m1.losses.shape == (180, 3) and m1.weights.shape == (180, 3)
    assert set(m1.snapshots) == set(snapshot_iterations(m1.iters)) and np.array_equal(m1.snapshots[180], m1.embedding)
    assert m1.weights[0, 0] == 1000.0 and m1.weights[-1, 0] == 0.0 and m1.losses[-1, 1] == 0.0        # Phase 3: keine mittleren Paare
    assert (m1.n_neighbors, m1.n_mn, m1.n_fp) == (10, 5, 20)


def test_random_start_depends_on_its_own_seed_and_pca_start_is_small():
    ds = _data(120)
    a, b, c = (fit_pacmap(ds.X, 10, 0.5, 2.0, 90, "random", s) for s in (0, 0, 1))
    assert np.array_equal(a.embedding, b.embedding) and not np.array_equal(a.embedding, c.embedding)
    assert np.abs(fit_pacmap(ds.X, 10, 0.5, 2.0, 90).snapshots[0]).max() < 0.05


def test_invalid_options_are_rejected():
    X = _data(60).X
    with pytest.raises(ValueError):
        fit_pacmap(X, 10, 0.5, 2.0, 90, init="umap")
    with pytest.raises(ValueError):
        fit_pacmap(X, 10, 0.5, 2.0, 5)


def test_zero_mid_near_pairs_is_supported():
    m = fit_pacmap(_data(100).X, 10, 0.0, 2.0, 90)
    assert m.n_mn == 0 and len(m.mn_pairs) == 0 and np.isfinite(m.embedding).all()


def test_quality_is_comparable_to_the_reference_implementation_over_several_datasets():
    """Chaotischer Optimierer: über drei Datensätze gemittelt statt Einzelwert (Seed 7 einzeln: 0.80 gegen 0.67 der Referenz)."""
    from pacmap_evaluation import r2_quadratic, trustworthiness
    ours, ref = [], []
    for seed in (100_000, 100_001, 100_002):
        ds = _data(seed=seed)
        Z = standardize(ds.X)
        m = fit_pacmap(ds.X, 10, 0.5, 2.0, 450)
        r = pacmap_lib.PaCMAP(n_neighbors=10, random_state=0, save_tree=False).fit_transform(Z.astype(np.float32), init="pca")
        ours.append((r2_quadratic(m.embedding, ds.z), trustworthiness(Z, m.embedding)))
        ref.append((r2_quadratic(r, ds.z), trustworthiness(Z, r)))
    ours, ref = np.mean(ours, axis=0), np.mean(ref, axis=0)
    assert ours[0] > ref[0] - 0.15 and ours[1] > ref[1] - 0.03 and ours[0] > 0.6


def test_transform_of_training_points_lands_near_their_own_coordinates():
    ds = _data()
    m = fit_pacmap(ds.X, 10, 0.5, 2.0, 450)
    y = transform(m, ds.X[:30])
    assert y.shape == (30, 2) and np.abs(y - m.embedding[:30]).max() < 0.15 * np.abs(m.embedding).max()


def test_transform_is_deterministic_and_lands_inside_the_training_embedding():
    ds = _data(200)
    m = fit_pacmap(ds.X[:160], 10, 0.5, 2.0, 180)
    a = transform(m, ds.X[160:])
    assert np.array_equal(a, transform(m, ds.X[160:])) and (np.abs(a) <= np.abs(m.embedding).max() * 1.3).all()


def test_copied_isomap_lle_tsne_umap_and_scenario_match_their_reference_values():
    ds = _data()
    r = fit_isomap(ds.X, 8, 2)
    assert abs(float(np.abs(r.embedding).sum()) - 1786.6726227517283) < 1e-6 and abs(float(r.geodesic.sum()) - 621595.593373937) < 1e-3
    lle = fit_lle(ds.X, 14, 2, 1e-2)
    assert np.allclose(lle.embedding.mean(0), 0, atol=1e-9) and abs(lle.eigenvalues[0]) < 1e-10
    with pytest.raises(SingularNeighbourhood):
        fit_lle(ds.X, 20, 2, 0.0)
    assert abs(fit_tsne(ds.X, 30, 750).kl - 0.3098433168465976) < 2e-3            # chaotisch: auf CI andere BLAS, Abweichung 5e-5
    um = fit_umap(ds.X, 15, 0.1, 100)
    assert um.connected and abs(float(um.graph.sum()) - 1809.4579300820042) < 1e-3 and abs(um.a - 1.5769) < 1e-3     # deterministische Größen des Graphen und der Kurve
    assert procrustes_disparity(np.eye(2), np.eye(2) * 3) < 1e-12
    assert abs(float(ds.X.sum()) - 14553337.310875032) < 1e-6 and abs(float(generate_dataset(200, 3, 0.4, 0.3, 5, 42).X.sum()) - 10763498.969287368) < 1e-6
