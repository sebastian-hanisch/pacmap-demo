"""Plotly-Visualisierungen der PaCMAP-Demo: die drei Paartypen einer Tour, Verlustterme und Gewichtsschema, Einbettungen, Optimierungsverlauf, Sweeps, Abstandstreue, Stabilitätsvergleich, Out-of-sample und Rechenzeit.
Alle Figuren laufen durch `lock_axes` (Touch-Scrolling-Konvention des Portfolios: keine Zoom-/Pan-Gesten im Chart)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98", "#8e5fbf"
PHASE_COLORS = ("rgba(214,138,46,0.10)", "rgba(31,119,180,0.10)", "rgba(44,160,44,0.10)")


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _scatter(coords, color, name="Touren", size=7, showscale=False, label="latenter Faktor 1", opacity=1.0):
    return go.Scatter(
        x=coords[:, 0], y=coords[:, 1], mode="markers", name=name, hoverinfo="skip",
        marker=dict(color=color, colorscale="Viridis", size=size, showscale=showscale, opacity=opacity, line=dict(width=0.5, color="white"),
                    colorbar=dict(title=label) if showscale else None),
    )


def build_pairs_view(coords, color, focus, near, mid, far):
    """2-D-Ansicht (erste zwei Hauptkomponenten): gewählte Tour und ihre drei Paartypen - nahe (orange), mittlere (grün), ferne (rot) Partner."""
    fig = go.Figure(_scatter(coords, color, showscale=True, opacity=0.4))
    for idx, name, col, size in ((far, "ferne Paare (stoßen ab)", RED, 9), (mid, "mittlere Paare (ziehen an)", GREEN, 12), (near, "nahe Paare (ziehen an)", ORANGE, 12)):
        if len(idx):
            fig.add_trace(go.Scatter(x=coords[idx, 0], y=coords[idx, 1], mode="markers", name=name, hoverinfo="skip", marker=dict(color=col, size=size, line=dict(width=1, color="#14233B"))))
    fig.add_trace(go.Scatter(x=[coords[focus, 0]], y=[coords[focus, 1]], mode="markers", name="gewählte Tour", hoverinfo="skip",
                             marker=dict(color="#111111", size=15, symbol="star", line=dict(width=1, color="white"))))
    fig.update_xaxes(title="PC1 (Ansicht)")
    fig.update_yaxes(title="PC2 (Ansicht)")
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_pair_losses(phase_weights):
    """Verlust je Paar gegen den Abstand d im Bild (d̃ = 1 + d²), gewichtet für drei Zeitpunkte: Anfang Phase 1, Phase 2, Phase 3. `phase_weights`: [(Titel, w_MN, w_NB, w_FP)]."""
    d = np.linspace(0, 8, 200)
    dt = 1.0 + d ** 2
    fig = make_subplots(rows=1, cols=len(phase_weights), subplot_titles=[p[0] for p in phase_weights], shared_yaxes=False)
    for col, (_, w_mn, w_nb, w_fp) in enumerate(phase_weights, start=1):
        fig.add_trace(go.Scatter(x=d, y=w_nb * dt / (10.0 + dt), mode="lines", line=dict(color=ORANGE, width=3), name="nah (zieht an)", showlegend=col == 1), row=1, col=col)
        if w_mn > 0:
            fig.add_trace(go.Scatter(x=d, y=w_mn * dt / (10000.0 + dt), mode="lines", line=dict(color=GREEN, width=3), name="mittel (zieht an)", showlegend=col == 1), row=1, col=col)
        fig.add_trace(go.Scatter(x=d, y=w_fp / (1.0 + dt), mode="lines", line=dict(color=RED, width=3), name="fern (stößt ab)", showlegend=col == 1), row=1, col=col)
    fig.update_xaxes(title_text="Abstand im Bild")
    fig.update_yaxes(title_text="gewichteter Verlust je Paar", col=1)
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_weight_schedule(weights, iters):
    """Gewichtsschema über die Iterationen: w_MN (links, log) fällt in Phase 1 von 1000 auf 3, in Phase 3 auf 0; w_NB und w_FP springen zwischen den Phasen."""
    its = np.arange(1, len(weights) + 1)
    fig = go.Figure()
    p1, p2, p3 = iters
    for k, (x0, x1) in enumerate(((0, p1), (p1, p1 + p2), (p1 + p2, p1 + p2 + p3))):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=PHASE_COLORS[k], line_width=0, annotation_text=f"Phase {k + 1}", annotation_position="top left")
    fig.add_trace(go.Scatter(x=its, y=np.maximum(weights[:, 0], 1e-3), mode="lines", line=dict(color=GREEN, width=3), name="w mittlere Paare"))
    fig.add_trace(go.Scatter(x=its, y=weights[:, 1], mode="lines", line=dict(color=ORANGE, width=3), name="w nahe Paare"))
    fig.add_trace(go.Scatter(x=its, y=weights[:, 2], mode="lines", line=dict(color=RED, width=3, dash="dot"), name="w ferne Paare"))
    fig.update_xaxes(title="Iteration")
    fig.update_yaxes(title="Gewicht (log)", type="log", range=[-0.5, 3.3])
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_embedding(coords, color, title_x, title_y, label="latenter Faktor 1", height=380):
    fig = go.Figure(_scatter(coords, color, showscale=True, label=label))
    fig.update_xaxes(title=title_x)
    fig.update_yaxes(title=title_y)
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=20, b=10))
    return lock_axes(fig)


def build_optimization(losses, snapshot_r2, snapshot_far, iters, marker=None):
    """Optimierungsverlauf: Verlustanteile je Paartyp (log), R² der wahren Faktoren und Abstandstreue ferner Paare an den Schnappschüssen; die drei Phasen sind hinterlegt."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Verlust je Paartyp", "R² der wahren Faktoren", "Abstandstreue ferner Paare"))
    its = np.arange(1, len(losses) + 1)
    fig.add_trace(go.Scatter(x=its, y=np.maximum(losses[:, 0], 1e-3), mode="lines", line=dict(color=ORANGE, width=3), name="nah"), row=1, col=1)
    fig.add_trace(go.Scatter(x=its, y=np.maximum(losses[:, 1], 1e-3), mode="lines", line=dict(color=GREEN, width=3), name="mittel"), row=1, col=1)
    fig.add_trace(go.Scatter(x=its, y=np.maximum(losses[:, 2], 1e-3), mode="lines", line=dict(color=RED, width=3), name="fern"), row=1, col=1)
    r2 = sorted(snapshot_r2.items())
    far = sorted(snapshot_far.items())
    fig.add_trace(go.Scatter(x=[i for i, _ in r2], y=[v for _, v in r2], mode="lines+markers", line=dict(color=PURPLE, width=3), showlegend=False, hovertemplate="Iteration %{x}: %{y:.2f}<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[i for i, _ in far], y=[v for _, v in far], mode="lines+markers", line=dict(color=BLUE, width=3), showlegend=False, hovertemplate="Iteration %{x}: %{y:.2f}<extra></extra>"), row=1, col=3)
    p1, p2, p3 = iters
    for col in (1, 2, 3):
        for k, (x0, x1) in enumerate(((0, p1), (p1, p1 + p2), (p1 + p2, p1 + p2 + p3))):
            fig.add_vrect(x0=x0, x1=x1, fillcolor=PHASE_COLORS[k], line_width=0, row=1, col=col)
        if marker is not None:
            fig.add_vline(x=max(marker, 1), line_dash="dot", line_color=GRAY, row=1, col=col)
    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(type="log", col=1)
    fig.update_yaxes(range=[-0.2, 1.02], col=2)
    fig.update_yaxes(range=[-0.2, 1.02], col=3)
    fig.update_layout(template="plotly_white", height=330, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_sweep(rows, xkey, current, xtitle, log=False):
    """Sweep über einen Regler: R², Abstandstreue ferner Paare und Trustworthiness."""
    xs = [r[xkey] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("R² der wahren Faktoren", "Abstandstreue (fern) und Trustworthiness"))
    fig.add_trace(go.Scatter(x=xs, y=[r["r2"] for r in rows], mode="lines+markers", line=dict(color=ORANGE, width=3), name="R²"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["far"] for r in rows], mode="lines+markers", line=dict(color=RED, width=3), name="Abstandstreue (fern)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=xs, y=[r["trust"] for r in rows], mode="lines+markers", line=dict(color=GREEN, width=3), name="Trustworthiness"), row=1, col=2)
    for col in (1, 2):
        if current is not None:
            fig.add_vline(x=current, line_dash="dot", line_color=GRAY, row=1, col=col)
    fig.update_xaxes(title_text=xtitle, tickvals=xs, ticktext=[f"{x:g}" for x in xs], type="log" if log else "linear")
    fig.update_yaxes(range=[-0.2, 1.02])
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_distance_fidelity(latent_pairs, panels):
    """Paarabstände der 2-D-Einbettung gegen die Abstände der wahren Faktoren (je auf Mittelwert 1 normiert). `panels`: [(Titel, Abstände, Farbe)]; auf der Diagonalen ist die Einbettung abstandstreu."""
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[0] for p in panels])
    for col, (_, pairs, color) in enumerate(panels, start=1):
        top = float(max(latent_pairs.max(), pairs.max())) * 1.05
        fig.add_trace(go.Scatter(x=latent_pairs, y=pairs, mode="markers", marker=dict(color=color, size=5, opacity=0.35), hoverinfo="skip", showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=GRAY, dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=col)
    fig.update_xaxes(title_text="Abstand der wahren Faktoren (normiert)")
    fig.update_yaxes(title_text="Abstand in der Einbettung (normiert)", col=1)
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_stability_compare(rows):
    """Median der paarweisen Procrustes-Abstände zufälliger Starts je Datensatz für PaCMAP, UMAP und t-SNE (niedriger = stabiler)."""
    labels = [f"Datensatz {i + 1}" for i in range(len(rows))]
    fig = go.Figure()
    for key, name, color in (("pacmap", "PaCMAP", ORANGE), ("umap", "UMAP", PURPLE), ("tsne", "t-SNE", RED)):
        fig.add_trace(go.Bar(x=labels, y=[r[key] for r in rows], name=name, marker_color=color, hovertemplate="%{x}: %{y:.2f}<extra>" + name + "</extra>"))
    fig.update_yaxes(title="mittlerer Procrustes-Abstand", range=[0, 1])
    fig.update_layout(template="plotly_white", barmode="group", height=320, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_out_of_sample(panels, train_color, test_color):
    """Zurückgehaltene Touren (Sterne) in den Einbettungen der drei Verfahren. `panels`: [(Titel, Training, Neu)]."""
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[0] for p in panels])
    for col, (_, train, test) in enumerate(panels, start=1):
        fig.add_trace(go.Scatter(x=train[:, 0], y=train[:, 1], mode="markers", hoverinfo="skip", showlegend=False, marker=dict(color=train_color, colorscale="Viridis", size=5, opacity=0.5)), row=1, col=col)
        fig.add_trace(go.Scatter(x=test[:, 0], y=test[:, 1], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(color=test_color, colorscale="Viridis", cmin=float(train_color.min()), cmax=float(train_color.max()), size=11, symbol="star",
                                             line=dict(width=1.2, color="#14233B"))), row=1, col=col)
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_timing(rows):
    ns = np.array([r["n"] for r in rows], dtype=float)
    fig = go.Figure()
    for key, label, color in (("pacmap", "PaCMAP", ORANGE), ("umap", "UMAP", PURPLE), ("tsne", "t-SNE", RED), ("isomap", "Isomap", BLUE), ("lle", "LLE", "#8c564b"), ("pca", "PCA", GREEN)):
        fig.add_trace(go.Scatter(x=ns, y=[max(r[key], 1e-6) for r in rows], mode="lines+markers", name=label, line=dict(color=color, width=3)))
    fig.update_xaxes(title="Anzahl Touren n", type="log")
    fig.update_yaxes(title="Rechenzeit (s)", type="log")
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)
