"""Defaults, Slider-Grenzen und Presets für die PaCMAP-Demo. Merkmale und Erzeugungs-Konstanten sind wortgleich aus pca-demo übernommen
(dieselben Lieferrouten - dieselbe gekrümmte Fläche, an der PCA scheiterte); alles Übrige ist neu."""

# --- Merkmale: 12 Kennzahlen je Tour in 4 Gruppen zu je 3 (Name, Einheit, Mittelwert, typische Streuung in Einheiten) ------------
FEATURES = (
    ("Distanz", "m", 45000.0, 15000.0),
    ("Stopps", "Anzahl", 60.0, 20.0),
    ("Ladegewicht", "kg", 1200.0, 400.0),
    ("Zeitfenster-Enge", "min", 90.0, 30.0),
    ("Verspätung", "min", 12.0, 8.0),
    ("Überstunden", "min", 25.0, 15.0),
    ("Fahrzeit je km", "s", 90.0, 25.0),
    ("Stop-and-go-Anteil", "%", 22.0, 10.0),
    ("Parkzeit", "min", 35.0, 12.0),
    ("Retourenquote", "Anteil", 0.06, 0.02),
    ("Sonderwünsche", "Anzahl", 4.0, 2.0),
    ("Zustellversuche", "Anzahl", 1.3, 0.5),
)
N_FEATURES = len(FEATURES)
FEATURE_NAMES = tuple(f[0] for f in FEATURES)
FEATURE_LABELS = tuple(f"{f[0]} [{f[1]}]" for f in FEATURES)
GROUPS = ("Größe", "Zeitdruck", "Verkehr", "Sonderfälle")     # je 3 aufeinanderfolgende Merkmale
GROUP_OF_FEATURE = tuple(i // 3 for i in range(N_FEATURES))

# --- Regler ------------------------------------------------------------------------------------------------------------
DEFAULT_N_TOURS = 300
N_TOURS_MIN, N_TOURS_MAX = 100, 600
DEFAULT_Q = 2
Q_MIN, Q_MAX = 1, 4
DEFAULT_CURVATURE = 1.0
CURVATURE_MIN, CURVATURE_MAX = 0.0, 1.0
DEFAULT_NOISE = 0.25
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_OUTLIER_PCT = 0
OUTLIER_PCT_MIN, OUTLIER_PCT_MAX = 0, 10
DEFAULT_N_NEIGHBORS = 10
N_NEIGHBORS_MIN, N_NEIGHBORS_MAX = 2, 50           # die obere Grenze folgt zusätzlich der Tourenzahl (k · (1 + MN + FP) < n)
MN_RATIO_CHOICES = (0.0, 0.25, 0.5, 1.0, 2.0)
DEFAULT_MN_RATIO = 0.5
FP_RATIO_CHOICES = (0.5, 1.0, 2.0, 4.0)
DEFAULT_FP_RATIO = 2.0
DEFAULT_N_ITER = 450
N_ITER_MIN, N_ITER_MAX = 90, 900                   # Vielfache von 9: Phasen 2/9 - 2/9 - 5/9 (wie 100 - 100 - 250 bei 450)
INITS = ("pca",) + tuple(f"random:{i}" for i in range(5))
INIT_LABELS = {"pca": "PCA (deterministisch)", **{f"random:{i}": f"zufällig, Start {i + 1}" for i in range(5)}}
DEFAULT_INIT = "pca"
DEFAULT_SEED = 7

# --- Erzeugung ---------------------------------------------------------------------------------------------------------
OUTLIER_SCALE = 10.0                   # Sonderfahrten: latenter Faktor um diesen Faktor vergrößert
CROSS_LOADING = 0.15                   # kleine Querladungen zwischen Merkmalsgruppen
WITHIN_LOADINGS = (0.95, 0.9, 0.85)    # Ladung der drei Merkmale einer Gruppe auf ihren Faktor
CURVATURE_FREQUENCY = 1.6              # Frequenz der sin/cos-Terme der Krümmung
CURVATURE_AMPLITUDE = 2.0              # Länge jeder Spalte der Krümmungsmatrix (in z-Einheiten bei Krümmung 1)
LAYOUT_SEED = 20240915                 # feste Ladungs- und Krümmungsmatrizen (unabhängig vom Seed der Touren)


# --- Auswertung --------------------------------------------------------------------------------------------------------
TRUST_NEIGHBORS = 10
UMAP_N_NEIGHBORS, UMAP_MIN_DIST, UMAP_N_EPOCHS = 15, 0.1, 500      # Vergleichsverfahren mit ihren guten Einstellungen (siehe umap-demo / tsne-demo / isomap-demo / lle-demo)
TSNE_PERPLEXITY, TSNE_N_ITER = 30, 500
ISOMAP_K = 10
LLE_K, LLE_REG = 14, 1e-2
SWEEP_SEEDS = tuple(100_000 + i for i in range(3))                 # feste Sweep-Seeds, unabhängig vom Demo-Seed
SWEEP_N_NEIGHBORS = (2, 3, 5, 10, 20, 40)
SWEEP_MN_RATIOS = (0.0, 0.25, 0.5, 1.0, 2.0)
SWEEP_FP_RATIOS = (0.5, 1.0, 2.0, 4.0)
SWEEP_N_TOURS = 200
STABILITY_INITS = ("random:0", "random:1", "random:2", "random:3")
STABILITY_SEEDS = tuple(100_000 + i for i in range(4))             # feste Datensätze für den fairen Stabilitätsvergleich
HOLDOUT_FRACTION = 0.2
TIMING_NS = (100, 200, 400, 600)
CONVERGED_ITERS = 450                              # Referenzlauf für die Konvergenz-Prüfung bei wenigen Iterationen

_BASE = {"n_tours": DEFAULT_N_TOURS, "q": DEFAULT_Q, "curvature": DEFAULT_CURVATURE, "noise": DEFAULT_NOISE, "outlier_pct": DEFAULT_OUTLIER_PCT, "n_neighbors": DEFAULT_N_NEIGHBORS,
         "mn_ratio": DEFAULT_MN_RATIO, "fp_ratio": DEFAULT_FP_RATIO, "n_iter": DEFAULT_N_ITER, "init": DEFAULT_INIT, "seed": DEFAULT_SEED}
PRESETS = {
    "Gekrümmte Fläche: PaCMAP entrollt": {**_BASE},
    "Ohne mittlere Paare: globale Ordnung fehlt": {**_BASE, "mn_ratio": 0.0},
    "Sonderfahrten: dieselbe Schwäche wie UMAP": {**_BASE, "outlier_pct": 5},
    "n_neighbors zu klein": {**_BASE, "n_neighbors": 2},
    "Rauschen: PaCMAP hält": {**_BASE, "noise": 0.8},
    "Gerade Daten: kein Vorteil": {**_BASE, "curvature": 0.0},
}
PRESET_HELP = {
    "Gekrümmte Fläche: PaCMAP entrollt": "Dieselbe gebogene Fläche wie in den Demos davor: PaCMAP gewinnt die versteckten Faktoren gut zurück (R² ≈ 0.80 gegen 0.50 der PCA; UMAP 0.88, t-SNE 0.93, Isomap 0.98) - bei fernen Tourenpaaren aber nur mit Abstandstreue 0.39, klar unter UMAP (0.56).",
    "Ohne mittlere Paare: globale Ordnung fehlt": "Ohne mittlere Paare (MN-Verhältnis 0) fehlt die Kraft, die ferne Regionen grob ordnet: Abstandstreue ferner Paare -0.04 statt 0.39, R² 0.62 statt 0.80 (Seed 7; über 3 Seeds im Mittel 0.06 statt 0.33). Das bestätigt den Mechanismus - nicht den Anspruch, dass PaCMAP damit globaler als UMAP wäre.",
    "Sonderfahrten: dieselbe Schwäche wie UMAP": "5 % Sonderfahrten mit extremen Werten: PCA (R² ≈ 0.76) und Isomap (0.75) behalten die Größenordnung, PaCMAP fällt auf 0.14 - genau wie UMAP (0.14) und t-SNE (0.12). Auch PaCMAPs mittlere Paare retten die globale Struktur hier nicht.",
    "n_neighbors zu klein": "Mit nur 2 nahen Paaren je Tour fehlt die lokale Verankerung: R² der Faktoren 0.49 statt 0.80 mit dem Standard (10 Nachbarn; über 4 Datensätze 0.37-0.49 gegen 0.78-0.89). Die Demo rechnet einen Referenzlauf mit den Standard-Paaren daneben.",
    "Rauschen: PaCMAP hält": "Mit viel Rauschen (0.8) liegt PaCMAP bei R² ≈ 0.87 - etwa auf dem Niveau von UMAP (0.89), t-SNE (0.85) und Isomap (0.84), deutlich über der PCA (0.45).",
    "Gerade Daten: kein Vorteil": "Krümmung 0: die Kennzahlen hängen linear von den Faktoren ab, die PCA ist optimal (R² 0.98) - PaCMAP erreicht 0.93, bei fernen Paaren nur 0.35 gegen 0.94.",
}
PRESET_EXPECTED_BANDS = {
    "Gekrümmte Fläche: PaCMAP entrollt": {"verdict": "pacmap_wins", "r2": (0.68, 0.95), "r2_umap": (0.8, 0.96), "r2_iso": (0.94, 1.0), "r2_pca": (0.4, 0.6), "far": (0.2, 0.6), "far_umap": (0.4, 0.75)},
    "Ohne mittlere Paare: globale Ordnung fehlt": {"verdict": "mn_missing", "far": (-0.3, 0.2), "far_ref": (0.2, 0.6), "r2": (0.45, 0.78), "r2_ref": (0.68, 0.95), "n_mn": (0, 0)},
    "Sonderfahrten: dieselbe Schwäche wie UMAP": {"verdict": "global_structure", "r2": (0.0, 0.35), "r2_umap": (0.0, 0.35), "r2_pca": (0.6, 0.9), "r2_iso": (0.6, 0.9), "far": (-0.1, 0.35), "far_pca": (0.85, 1.0)},
    "n_neighbors zu klein": {"verdict": "neighbors_small", "r2": (0.2, 0.65), "r2_ref": (0.68, 0.95)},
    "Rauschen: PaCMAP hält": {"verdict": "pacmap_wins", "r2": (0.75, 0.95), "r2_umap": (0.78, 0.96), "r2_tsne": (0.7, 0.92), "r2_pca": (0.3, 0.6)},
    "Gerade Daten: kein Vorteil": {"verdict": "no_advantage", "r2": (0.82, 0.97), "r2_pca": (0.94, 1.0), "far": (0.15, 0.6), "far_pca": (0.85, 1.0)},
}
