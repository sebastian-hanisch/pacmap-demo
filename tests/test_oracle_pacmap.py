"""Orakel-Tests (unabhängiger Rechenweg): Verlust und Gradient von Hand per Schleife und finiter Differenz (ohne Bibliothek), nahe Paare gegen die Funktionen der Referenz `pacmap` auf EXAKTEM kNN,
Gewichtsschema, Paarzahlen und Adam gegen die Referenz (fehlt `pacmap` im Test-venv: übersprungen), Trustworthiness gegen sklearn, Procrustes gegen scipy, R²/Abstandstreue von Hand,
Out-of-sample-R² mit unzentrierten Residuen, Optimalitätsbedingung von `transform`."""

import numpy as np
import pytest
from scipy.spatial import procrustes as sp_procrustes
from scipy.spatial.distance import cdist

import pacmap_algorithm as A
import pacmap_evaluation as E
import pacmap_scenario as S
from pacmap_isomap import standardize
from pacmap_tsne import procrustes_disparity

manifold = pytest.importorskip("sklearn.manifold")
neighbors = pytest.importorskip("sklearn.neighbors")


def _setup(rng):
    n = int(rng.integers(40, 110))
    ds = S.generate_dataset(n, int(rng.integers(1, 4)), float(rng.uniform(0, 1)), float(rng.uniform(0, 1)), 0, int(rng.integers(10 ** 6)))
    Xp = A.preprocess(standardize(ds.X))[0]
    return ds, Xp, cdist(Xp, Xp)


def test_loss_and_gradient_equal_the_formulas_by_loop_and_finite_differences():
    rng = np.random.default_rng(1)
    for t in range(14):
        ds, Xp, dist = _setup(rng)
        n = len(Xp)
        k, n_mn, n_fp = A.decide_num_pairs(n, int(rng.integers(3, 14)), float(rng.choice([0.5, 1.0, 2.0])), float(rng.choice([1.0, 2.0])))
        nb = A.neighbour_pairs(dist, k)
        r = np.random.default_rng(t)
        mn, fp = A.mid_near_pairs(dist, n_mn, r), A.further_pairs(n, n_fp, nb, k, r)
        Y = r.normal(size=(n, 2)) * r.uniform(0.1, 5)
        w_nb, w_mn, w_fp = float(r.uniform(0.5, 4)), float(r.choice([3.0, 500.0])), float(r.uniform(0.2, 2))

        def loss(Yt):
            d = lambda i, j: 1 + ((Yt[i] - Yt[j]) ** 2).sum()
            return np.array([sum(w_nb * d(i, j) / (10 + d(i, j)) for i, j in nb), sum(w_mn * d(i, j) / (10000 + d(i, j)) for i, j in mn),
                             sum(w_fp / (1 + d(i, j)) for i, j in fp)])
        grad, losses = A.pair_gradient(Y, nb, mn, fp, w_nb, w_mn, w_fp)
        assert np.allclose(losses, loss(Y), rtol=1e-10)
        for _ in range(3):
            i, c = int(r.integers(n)), int(r.integers(2))
            e = np.zeros_like(Y)
            e[i, c] = 1e-6
            assert (loss(Y + e).sum() - loss(Y - e).sum()) / 2e-6 == pytest.approx(grad[i, c], rel=1e-5, abs=1e-5)


def test_pair_lists_have_the_documented_structure_and_near_pairs_equal_the_reference_on_exact_knn():
    rng = np.random.default_rng(2)
    pmm = None
    try:
        from pacmap import pacmap as pmm
    except ImportError:
        pass
    for t in range(10):
        ds, Xp, dist = _setup(rng)
        n = len(Xp)
        k, n_mn, n_fp = A.decide_num_pairs(n, int(rng.integers(3, 14)), float(rng.choice([0.5, 1.0, 2.0])), float(rng.choice([1.0, 2.0])))
        nb = A.neighbour_pairs(dist, k)
        r = np.random.default_rng(t)
        mn, fp = A.mid_near_pairs(dist, n_mn, r), A.further_pairs(n, n_fp, nb, k, r)
        nbs = [set(nb[nb[:, 0] == i, 1].tolist()) for i in range(n)]
        assert all(len(s) == k and i not in s for i, s in enumerate(nbs))
        assert not any(j in nbs[i] or i == j for i, j in fp) and not any(i == j for i, j in mn)
        for pairs, count in ((mn, n_mn), (fp, n_fp)):
            assert all(len(set(pairs[pairs[:, 0] == i, 1].tolist())) == count for i in range(n))
        if pmm is None:
            continue
        extra = min(k + 50, n - 1)
        kd, ki = neighbors.NearestNeighbors(n_neighbors=extra + 1).fit(Xp).kneighbors(Xp)
        kd, ki = kd[:, 1:].astype(np.float32), ki[:, 1:].astype(np.int32)
        sig = np.maximum(kd[:, 3:6].mean(1), 1e-10).astype(np.float32)
        ref = pmm.sample_neighbors_pair(Xp.astype(np.float32), (kd ** 2 / sig[:, None] / sig[ki]).astype(np.float32), ki, k)
        differing = sum(set(nb[nb[:, 0] == i, 1].tolist()) != set(ref[ref[:, 0] == i, 1].tolist()) for i in range(n))
        assert differing <= 0.02 * n                                                                    # Gleichstände / float32 dürfen einzelne Zeilen kippen


def test_schedule_pair_counts_preprocessing_and_adam_equal_the_reference():
    lib = pytest.importorskip("pacmap")
    from pacmap import pacmap as pmm
    for n_iter in (90, 180, 450, 901):
        iters = A.phase_iterations(n_iter)
        assert sum(iters) == n_iter
        for itr in range(0, n_iter, max(1, n_iter // 30)):
            assert np.allclose(A.find_weight(itr, iters), pmm.find_weight(1000.0, itr, num_iters=iters))
    rng = np.random.default_rng(3)
    for _ in range(25):
        n, k, mnr, fpr = int(rng.integers(40, 300)), int(rng.integers(3, 40)), float(rng.choice([0.0, 0.5, 1.0, 2.0])), float(rng.choice([1.0, 2.0, 4.0]))
        ref = lib.PaCMAP(n_neighbors=k, MN_ratio=mnr, FP_ratio=fpr)
        ref.decide_num_pairs(n)
        assert A.decide_num_pairs(n, k, mnr, fpr) == (ref.n_neighbors, ref.n_MN, ref.n_FP)
    for t in range(30):
        r = np.random.default_rng(t)
        Y, g = r.normal(size=(15, 2)).astype(np.float32), r.normal(size=(15, 2)).astype(np.float32)
        m, v = (r.normal(size=(15, 2)) * 0.1).astype(np.float32), (np.abs(r.normal(size=(15, 2))) * 0.1).astype(np.float32)
        itr = int(r.integers(0, 450))
        rY, rm, rv = Y.copy(), m.copy(), v.copy()
        pmm.update_embedding_adam(rY, g, rm, rv, np.float32(0.9), np.float32(0.999), np.float32(1.0), itr)
        mY, mm, mv = Y.astype(float), m.astype(float), v.astype(float)
        A.adam_step(mY, g.astype(float), mm, mv, itr)
        assert np.allclose(mY, rY, atol=1e-4) and np.allclose(mm, rm, atol=1e-5) and np.allclose(mv, rv, atol=1e-5)


def test_metrics_equal_sklearn_scipy_and_by_hand_computations():
    rng = np.random.default_rng(4)
    for _ in range(30):
        ds = S.generate_dataset(int(rng.integers(40, 100)), int(rng.integers(1, 4)), float(rng.uniform(0, 1)), float(rng.uniform(0, 1)), 0, int(rng.integers(10 ** 6)))
        n = len(ds.z)
        e2 = rng.normal(size=(n, 2)) + ds.z[:, :1] * rng.uniform(0, 2)
        Z, kk = standardize(ds.X), int(rng.integers(3, 9))
        assert E.trustworthiness(Z, e2, kk) == pytest.approx(manifold.trustworthiness(Z, e2, n_neighbors=kk), abs=1e-9)
        dz, de = cdist(ds.z, ds.z), cdist(e2, e2)
        iu = np.triu_indices(n, 1)
        near = dz[iu] <= np.median(dz[iu])
        got = E.distance_fidelity_split(e2, ds.z)
        assert got[0] == pytest.approx(np.corrcoef(dz[iu][near], de[iu][near])[0, 1], abs=1e-9) and got[1] == pytest.approx(np.corrcoef(dz[iu][~near], de[iu][~near])[0, 1], abs=1e-9)
        a, b = rng.normal(size=(n, 2)), rng.normal(size=(n, 2))
        assert procrustes_disparity(a, b) == pytest.approx(sp_procrustes(a, b)[2], abs=1e-9)
        assert procrustes_disparity(a, 2.5 * a @ np.array([[0.0, 1.0], [-1.0, 0.0]]) + 3) == pytest.approx(0.0, abs=1e-9)
        u = np.linalg.svd(E._quad_features(e2), full_matrices=False)[0]
        r2 = 1 - ((ds.z - u @ (u.T @ ds.z)) ** 2).sum() / ((ds.z - ds.z.mean(0)) ** 2).sum()
        assert E.r2_quadratic(e2, ds.z) == pytest.approx(r2, abs=1e-7)


def test_out_of_sample_r2_uses_uncentred_residuals_and_transform_reaches_a_stationary_point():
    ds = S.generate_dataset(100, 2, 1.0, 0.25, 0, 11)
    out = E.out_of_sample(ds, E.Settings(n_iter=90))
    zt = ds.z[out["test"]]
    for name, train_emb, y in (("pacmap", out["model"].embedding, out["y_pacmap"]), ("umap", out["umap_train"], out["y_umap"]), ("tsne", out["tsne_train"], out["y_tsne"])):
        beta = np.linalg.pinv(E._quad_features(train_emb)) @ ds.z[out["train"]]
        resid = zt - E._quad_features(y) @ beta
        assert out["r2_" + name] == pytest.approx(1 - (resid ** 2).sum() / ((zt - zt.mean(0)) ** 2).sum(), abs=1e-6)
    m = out["model"]
    xmin, xmax, xmean = m.prep
    Xn = (((ds.X[out["test"]] - m.mean) / m.scale) - xmin) / xmax - xmean
    idx = np.argsort(cdist(Xn, m.Xp), axis=1, kind="stable")[:, :m.n_neighbors]

    def nb_loss(Yq, i):
        d = 1 + ((Yq[i] - m.embedding[idx[i]]) ** 2).sum(1)
        return float((d / (10 + d)).sum())
    y = out["y_pacmap"]
    for i in range(4):
        for c in range(2):
            e = np.zeros_like(y)
            e[i, c] = 1e-5
            assert abs((nb_loss(y + e, i) - nb_loss(y - e, i)) / 2e-5) < 0.05                                     # Adam mit Lernrate 1 endet nahe, nicht exakt im Minimum
