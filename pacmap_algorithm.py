"""PaCMAP (Pairwise Controlled Manifold Approximation, Wang, Huang, Rudin & Shaposhnik 2021) von Grund auf (numpy, exaktes kNN, für n ≤ 600).

Folgt der Referenzimplementierung `pacmap` (Quellcode gelesen, im Test gegengeprüft):

1. **Drei Paartypen**: **nahe Paare (NB)** - je Tour die n_neighbors nächsten Nachbarn unter *skalierten* Abständen d²ᵢⱼ/(σᵢσⱼ), σᵢ = Mittel der Abstände zum 4.-6. Nachbarn (Kandidaten: die k + 50 nächsten);
   **mittlere Paare (MN)** - 6 Zufallspunkte ziehen, der *zweitnächste* wird gewählt; **ferne Paare (FP)** - Zufallspunkte, die keine nahen Paare sind. Anzahl MN = MN_ratio · k, FP = FP_ratio · k.
2. **Verlust** (d̃ = 1 + ‖yᵢ − yⱼ‖²): NB `w_NB · d̃/(10 + d̃)`, MN `w_MN · d̃/(10000 + d̃)`, FP `w_FP / (1 + d̃)` - nahe und mittlere Paare ziehen an, ferne stoßen ab.
3. **Gewichtsschema in drei Phasen** (Standard 100 - 100 - 250 Iterationen): Phase 1 (w_MN 1000 → 3, w_NB 2, w_FP 1) ordnet die globale Struktur, Phase 2 (w_MN 3, w_NB 3, w_FP 1) balanciert, Phase 3 (w_MN 0, w_NB 1, w_FP 1) verfeinert lokal.
4. **Adam** (Lernrate 1, β = 0.9/0.999) über alle Paare gleichzeitig; Start: 0.01 · PCA (nach der Vorverarbeitung des Originals: globales Min-Max, zentriert) oder zufällig · 1e-4.
5. **Neue Punkte** (`transform`, wie `PaCMAP.transform`): Paare zu den nächsten Trainings-Touren, nur Anziehung (Trainings-Einbettung fest), Adam mit dem NB-Gewichtsschema. Startpunkt hier: gewichteter Mittelwert der Nachbar-Koordinaten (die Referenz startet an der PCA).

Anders als die Referenz ist das kNN exakt (die Referenz nutzt näherungsweise Suche) - Ergebnisse sind gleichwertig, nicht bitgleich."""

from dataclasses import dataclass

import numpy as np

from pacmap_isomap import pairwise_distances

W_MN_INIT = 1000.0
BETA1, BETA2 = 0.9, 0.999
LEARNING_RATE = 1.0
PHASES = (2, 2, 5)                                 # Anteile der drei Phasen (450 Iterationen: 100 - 100 - 250)


def phase_iterations(n_iter):
    unit = n_iter // sum(PHASES)
    return unit * PHASES[0], unit * PHASES[1], n_iter - unit * (PHASES[0] + PHASES[1])


def find_weight(itr, iters, w_mn_init=W_MN_INIT):
    """Gewichte (w_MN, w_NB, w_FP) in Iteration `itr` (0-basiert) - identisch zu `pacmap.find_weight`."""
    p1, p2, _ = iters
    if itr < p1:
        return (1 - itr / p1) * w_mn_init + itr / p1 * 3.0, 2.0, 1.0
    if itr < p1 + p2:
        return 3.0, 3.0, 1.0
    return 0.0, 1.0, 1.0


def preprocess(Z):
    """Vorverarbeitung wie im Original für Daten mit höchstens 100 Merkmalen: globales Min-Max, dann zentrieren. -> (Xp, xmin, xmax, xmean)."""
    xmin = float(Z.min())
    X = Z - xmin
    xmax = float(X.max())
    X = X / xmax
    xmean = X.mean(0)
    return X - xmean, xmin, xmax, xmean


def decide_num_pairs(n, n_neighbors, mn_ratio, fp_ratio):
    """Anzahl (NB, MN, FP) je Tour, wie `PaCMAP.decide_num_pairs`."""
    n_mn = int(round(n_neighbors * mn_ratio))
    n_fp = int(round(n_neighbors * fp_ratio))
    n_neighbors = min(n_neighbors, n - 1)
    n_fp = min(n_fp, n - 1 - n_neighbors)
    n_mn = min(n_mn, n - 1)
    if n_neighbors + n_mn + n_fp >= n:
        total = 1 + mn_ratio + fp_ratio
        n_neighbors, n_mn, n_fp = int(n / total), int(n / total * mn_ratio), int(n / total * fp_ratio)
    if n_neighbors < 1 or n_fp < 1:
        raise ValueError("zu wenige Touren für diese Paar-Anzahlen")
    return n_neighbors, n_mn, n_fp


def neighbour_pairs(dist, n_neighbors):
    """Nahe Paare: k nächste Nachbarn unter skalierten Abständen d²/(σᵢσⱼ) aus den k + 50 nächsten. -> Paare [n·k, 2]."""
    n = len(dist)
    extra = min(n_neighbors + 50, n - 1)
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    nbrs = np.argsort(d, axis=1, kind="stable")[:, :extra]
    kd = np.take_along_axis(dist, nbrs, axis=1)
    sig = np.maximum(kd[:, 3:6].mean(1), 1e-10)
    scaled = kd ** 2 / sig[:, None] / sig[nbrs]
    pick = np.argsort(scaled, axis=1, kind="stable")[:, :n_neighbors]
    chosen = np.take_along_axis(nbrs, pick, axis=1)
    return np.column_stack([np.repeat(np.arange(n), n_neighbors), chosen.ravel()])


def _distinct_random(n, count, forbidden, rng):
    """Je Zeile `count` verschiedene gleichverteilte Zufallsindizes aus 0..n−1 ohne die Indizes in `forbidden` [n, m] (und ohne die Zeile selbst)."""
    keys = rng.random((n, n))
    keys[np.arange(n), np.arange(n)] = np.inf
    if forbidden is not None and forbidden.size:
        keys[np.repeat(np.arange(n), forbidden.shape[1]), forbidden.ravel()] = np.inf
    return np.argpartition(keys, count - 1, axis=1)[:, :count] if count < n else np.argsort(keys, axis=1)[:, :count]


def mid_near_pairs(dist, n_mn, rng):
    """Mittlere Paare: 6 Zufallspunkte (nicht die Tour selbst, nicht schon gewählte), der ZWEITNÄCHSTE wird gewählt - wiederholt n_mn-mal je Tour. -> Paare [n·n_mn, 2]."""
    n = len(dist)
    picked = np.empty((n, 0), dtype=int)
    for _ in range(n_mn):
        sample = _distinct_random(n, 6, picked, rng)
        d = np.take_along_axis(dist, sample, axis=1)
        second = np.argsort(d, axis=1, kind="stable")[:, 1]
        picked = np.column_stack([picked, sample[np.arange(n), second]])
    return np.column_stack([np.repeat(np.arange(n), n_mn), picked.ravel()]) if n_mn else np.zeros((0, 2), dtype=int)


def further_pairs(n, n_fp, nb_pairs, n_neighbors, rng):
    """Ferne Paare: Zufallspunkte, die weder die Tour selbst noch ihre nahen Nachbarn sind (einmal gezogen, nicht je Iteration neu). -> Paare [n·n_fp, 2]."""
    nb = nb_pairs[:, 1].reshape(n, n_neighbors)
    chosen = _distinct_random(n, n_fp, nb, rng)
    return np.column_stack([np.repeat(np.arange(n), n_fp), chosen.ravel()])


def pair_gradient(Y, nb, mn, fp, w_nb, w_mn, w_fp):
    """Gradient des PaCMAP-Verlusts und die Verlustanteile (NB, MN, FP) - Anziehung bei NB/MN, Abstoßung bei FP; wie `pacmap.pacmap_grad`."""
    n = len(Y)
    grad = np.zeros_like(Y)
    losses = np.zeros(3)
    for k, (pairs, w, kind) in enumerate(((nb, w_nb, "nb"), (mn, w_mn, "mn"), (fp, w_fp, "fp"))):
        if len(pairs) == 0 or w == 0:
            continue
        i, j = pairs[:, 0], pairs[:, 1]
        yij = Y[i] - Y[j]
        d = 1.0 + (yij ** 2).sum(1)
        if kind == "nb":
            losses[k] = w * (d / (10.0 + d)).sum()
            coeff = w * 20.0 / (10.0 + d) ** 2
        elif kind == "mn":
            losses[k] = w * (d / (10000.0 + d)).sum()
            coeff = w * 20000.0 / (10000.0 + d) ** 2
        else:
            losses[k] = w * (1.0 / (1.0 + d)).sum()
            coeff = -w * 2.0 / (1.0 + d) ** 2
        g = coeff[:, None] * yij
        for c in range(Y.shape[1]):
            grad[:, c] += np.bincount(i, weights=g[:, c], minlength=n) - np.bincount(j, weights=g[:, c], minlength=n)
    return grad, losses


def adam_step(Y, grad, m, v, itr, lr=LEARNING_RATE):
    """Ein Adam-Schritt (in-place) wie `pacmap.update_embedding_adam`."""
    lr_t = lr * np.sqrt(1.0 - BETA2 ** (itr + 1)) / (1.0 - BETA1 ** (itr + 1))
    m += (1 - BETA1) * (grad - m)
    v += (1 - BETA2) * (grad ** 2 - v)
    Y -= lr_t * m / (np.sqrt(v) + 1e-7)


def snapshot_iterations(iters):
    p1, p2, p3 = iters
    total = p1 + p2 + p3
    marks = {0, 10, 30, 60, p1, p1 + 20, p1 + 40, p1 + 70, p1 + p2, p1 + p2 + 50, p1 + p2 + 100, total}
    return sorted(m for m in marks if 0 <= m <= total)


@dataclass(frozen=True)
class PaCMAPModel:
    embedding: np.ndarray            # [n, 2]
    snapshots: dict                  # Iteration -> Einbettung (Kopie), inkl. 0 und n_iter
    losses: np.ndarray               # [n_iter, 3]: Verlust je Iteration (NB, MN, FP), mit den Gewichten der Iteration
    weights: np.ndarray              # [n_iter, 3]: (w_MN, w_NB, w_FP) je Iteration
    nb_pairs: np.ndarray
    mn_pairs: np.ndarray
    fp_pairs: np.ndarray
    n_neighbors: int
    n_mn: int
    n_fp: int
    iters: tuple                     # Iterationen je Phase
    init: str
    seed: int
    mean: np.ndarray                 # Standardisierung der Trainingsdaten (für transform)
    scale: np.ndarray
    Z: np.ndarray
    Xp: np.ndarray                   # vorverarbeitete Trainingsdaten
    prep: tuple                      # (xmin, xmax, xmean)

    @property
    def n(self):
        return len(self.Z)

    @property
    def n_iter(self):
        return int(sum(self.iters))


def fit_pacmap(X, n_neighbors=10, mn_ratio=0.5, fp_ratio=2.0, n_iter=450, init="pca", seed=0):
    X = np.asarray(X, dtype=float)
    if init not in ("pca", "random"):
        raise ValueError("init in {pca, random}")
    if n_iter < sum(PHASES):
        raise ValueError(f"n_iter muss mindestens {sum(PHASES)} sein")
    mean = X.mean(0)
    scale = X.std(0, ddof=1)
    scale = np.where(scale > 0, scale, 1.0)
    Z = (X - mean) / scale
    n = len(Z)
    Xp, xmin, xmax, xmean = preprocess(Z)
    k, n_mn, n_fp = decide_num_pairs(n, n_neighbors, mn_ratio, fp_ratio)
    dist = pairwise_distances(Xp)
    rng = np.random.default_rng(seed + 2_000_003)
    nb = neighbour_pairs(dist, k)
    mn = mid_near_pairs(dist, n_mn, rng)
    fp = further_pairs(n, n_fp, nb, k, rng)
    if init == "pca":
        _, _, vt = np.linalg.svd(Xp, full_matrices=False)
        Y = 0.01 * (Xp @ vt[:2].T)
    else:
        Y = np.random.default_rng(seed).normal(size=(n, 2)) * 1e-4
    iters = phase_iterations(n_iter)
    total = sum(iters)
    wanted = set(snapshot_iterations(iters))
    snapshots = {0: Y.copy()} if 0 in wanted else {}
    m, v = np.zeros_like(Y), np.zeros_like(Y)
    losses, weights = np.zeros((total, 3)), np.zeros((total, 3))
    for itr in range(total):
        w_mn, w_nb, w_fp = find_weight(itr, iters)
        grad, losses[itr] = pair_gradient(Y, nb, mn, fp, w_nb, w_mn, w_fp)
        weights[itr] = (w_mn, w_nb, w_fp)
        adam_step(Y, grad, m, v, itr)
        if itr + 1 in wanted:
            snapshots[itr + 1] = Y.copy()
    return PaCMAPModel(embedding=Y, snapshots=snapshots, losses=losses, weights=weights, nb_pairs=nb, mn_pairs=mn, fp_pairs=fp, n_neighbors=k, n_mn=n_mn, n_fp=n_fp, iters=iters,
                       init=init, seed=int(seed), mean=mean, scale=scale, Z=Z, Xp=Xp, prep=(xmin, xmax, xmean))


def transform(model, X_new, n_iter=None):
    """Neue Touren einbetten (Behelf in Anlehnung an `PaCMAP.transform`): nahe Paare zu den Trainings-Touren, nur Anziehung, Trainings-Einbettung fest, Adam mit dem NB-Gewichtsschema."""
    Zn = (np.asarray(X_new, dtype=float) - model.mean) / model.scale
    xmin, xmax, xmean = model.prep
    Xn = (Zn - xmin) / xmax - xmean
    d = np.sqrt(np.maximum(((Xn[:, None, :] - model.Xp[None, :, :]) ** 2).sum(-1), 0.0))
    k = model.n_neighbors
    idx = np.argsort(d, axis=1, kind="stable")[:, :k]
    w = 1.0 / np.maximum(np.take_along_axis(d, idx, axis=1), 1e-9)
    Y = np.einsum("ik,ikc->ic", w / w.sum(1, keepdims=True), model.embedding[idx])
    iters = model.iters if n_iter is None else phase_iterations(n_iter)
    m, v = np.zeros_like(Y), np.zeros_like(Y)
    rows = np.repeat(np.arange(len(Y)), k)
    cols = idx.ravel()
    for itr in range(sum(iters)):
        w_nb = find_weight(itr, iters)[1]
        diff = Y[rows] - model.embedding[cols]
        dd = 1.0 + (diff ** 2).sum(1)
        g = (w_nb * 20.0 / (10.0 + dd) ** 2)[:, None] * diff
        grad = np.stack([np.bincount(rows, weights=g[:, c], minlength=len(Y)) for c in range(2)], axis=1)
        adam_step(Y, grad, m, v, itr)
    return Y
