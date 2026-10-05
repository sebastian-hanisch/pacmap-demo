# PaCMAP an Lieferrouten-Kennzahlen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-pacmap-demo.streamlit.app/)**

Sechstes Stück der **Dimensionsreduktion-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **PaCMAP** – an einem wachsenden Beispiel. Vehikel: **dieselben 12
Lieferrouten-Kennzahlen wie in [pca-demo](../pca-demo), [isomap-demo](../isomap-demo), [lle-demo](../lle-demo), [tsne-demo](../tsne-demo) und [umap-demo](../umap-demo)** (der Generator ist wortgleich kopiert und
per Test gegen dessen Ausgabe eingefroren), erzeugt aus wenigen versteckten Faktoren – dieselbe gekrümmte Fläche, an der PCA scheiterte. UMAP, t-SNE, Isomap, LLE und PCA stehen als Vergleich daneben.

**Einordnung in die Reihe (die Kanten des Graphen):** PaCMAP ist die **Fortsetzung von UMAP** und schließt die Kette t-SNE → UMAP → PaCMAP. Statt Wahrscheinlichkeiten und Graphen nutzt es **drei feste Paarlisten**
(nahe, mittlere, ferne Paare) mit einem **Gewichtsschema in drei Phasen**; es soll UMAPs verbliebene Schwäche, die **globale Struktur** und die **lokal-global-Balance**, beheben. Die Demo **misst**, was davon auf diesen
Daten stimmt – **das Ergebnis ist überwiegend negativ**. PaCMAP hat außerdem weniger Verbreitung als UMAP und keinen etablierten Standard-Status.
```
pca-demo → isomap-demo | lle-demo | tsne-demo → umap-demo → pacmap-demo    (Kette t-SNE → UMAP → PaCMAP)
pca-demo → autoencoder-demo   (unabhängiger Ast, Stück 7 der Linie, gebaut)
```

| Versprechen / Frage | Ergebnis (300 Touren, 4 feste Datensätze, wenn nicht anders angegeben) |
|---|---|
| Bessere globale Struktur als UMAP | ❌ Abstandstreue ferner Paare im Mittel **0.34** (UMAP 0.60, t-SNE 0.58, PCA 0.38, Isomap 0.89) – PaCMAP liegt am schlechtesten der drei Verfahren |
| Extreme (Sonderfahrten) erhalten | ❌ 5 %: R² **0.30** (UMAP 0.36, t-SNE 0.35, PCA 0.67, Isomap 0.77) – dieselbe Schwäche |
| Mittlere Paare ordnen die globale Struktur (Mechanismus) | ✅ ohne MN-Paare: ferne Paare **0.06** statt 0.33 (3 Datensätze) |
| Geschwindigkeit | ✅ am schnellsten: n = 600 **0.4–0.5 s** gegen UMAP 1.3–1.7 s und t-SNE 4.9–5.1 s; auch bei n = 100–200 mindestens gleichauf |
| Stabilität (Start egal) | ⚠️ zwischen t-SNE und UMAP (200 Touren, q = 2, Median der paarweisen Abweichungen zufälliger Starts): PaCMAP 0.10–0.43, UMAP 0.01–0.22, t-SNE 0.37–0.53; bei q = 3 gemischt |
| Neue Touren einbetten | ⚠️ nur als Behelf: R² 0.79–0.91, UMAP `transform` 0.88–0.93; beim Neu-Rechnen verschieben sich die Trainings-Touren um Procrustes 0.06–0.56 |

## Was die Demo zeigt

1. **PaCMAP in Aktion** (Schritt-Slider + Abspielen): die **drei Paartypen** einer Tour (nah/mittel/fern, Anzahl und mittlerer Abstand im Original) → **Verlust je Paar** in den drei Phasen und das **Gewichtsschema** → **Optimierung**
   (Schnappschüsse, Verlust je Paartyp, R² und Abstandstreue ferner Paare je Schnappschuss, Phasen hinterlegt) → Ergebnis neben der PCA.
2. **Was PaCMAP gefunden hat – und die anderen fünf Verfahren auf denselben Daten:** sechs Einbettungen, R² der wahren Faktoren, Abstandstreue (gesamt/nah/fern), Trustworthiness, Paare je Tour.
3. **📐 Wie stark hängen die Ergebnisse von n_neighbors, den Paar-Verhältnissen und den Iterationen ab?** (live über feste Sweep-Seeds ab 100000, unabhängig vom Demo-Seed), mit Verdict (zu wenige nahe Paare →
   keine mittleren Paare → keine globale Struktur bei Sonderfahrten → kein Vorteil → PaCMAP entrollt).
4. **🆚 Löst PaCMAP sein Versprechen gegenüber UMAP ein?** – gemessene Tabelle, Abstände nah/fern; Experimente auf Abruf (Knopf): Out-of-sample (Behelf gegen UMAP-`transform` und t-SNE-Näherung), **fairer Stabilitätsvergleich**
   (PaCMAP/UMAP/t-SNE auf denselben Datensätzen), Rechenzeit.

Regler: Touren, wahre Faktoren q, Krümmung, Rauschen, **Sonderfahrten**, n_neighbors, MN-Verhältnis, FP-Verhältnis, Iterationen, Initialisierung (PCA / zufällig).

Messwerte (Seed 7, 300 Touren, q = 2, n_neighbors 10, MN 0.5, FP 2, 450 Iterationen, PCA-Start, wenn nicht anders angegeben; die Presets prüfen sie mit weiten Bändern):

| Situation | Messung |
|---|---|
| Gekrümmte Fläche | R² der wahren Faktoren **0.80** (PaCMAP) gegen 0.88 (UMAP), 0.93 (t-SNE), 0.98 (Isomap), 0.50 (PCA); Trustworthiness 0.98; **Abstandstreue ferner Paare 0.39** gegen 0.56 (UMAP) |
| Ohne mittlere Paare | Abstandstreue ferner Paare **−0.04** statt 0.39, R² 0.62 statt 0.80 (über 3 Seeds 0.06 statt 0.33) |
| 5 % Sonderfahrten | PaCMAP **R² 0.14**, UMAP 0.14, t-SNE 0.12, PCA 0.76, Isomap 0.75; ferne Paare 0.07 gegen 0.95 (PCA) |
| n_neighbors 2 | R² **0.49** statt 0.80 (4 Datensätze: 0.37–0.49 gegen 0.78–0.89) |
| Rauschen 0.8 | R² **0.87** – etwa wie UMAP (0.89), t-SNE (0.85), Isomap (0.84); PCA 0.45 |
| Gerade Daten (Krümmung 0) | **kein Vorteil**: R² 0.93 gegen 0.98 (PCA); ferne Paare 0.35 gegen 0.94 |

**Fairer Vergleich über 4 Datensätze** (300 Touren, feste Seeds 100000–100003; Mittel von R² / nahe Paare / ferne Paare / Trustworthiness): Standard – PaCMAP 0.89 / 0.53 / 0.34 / 0.98, UMAP 0.92 / 0.72 / 0.60 / 0.98,
t-SNE 0.89 / 0.68 / 0.58 / 0.99, Isomap 0.98 / 0.91 / 0.89 / 0.99, PCA 0.51 / 0.61 / 0.38 / 0.86. q = 3: R² 0.41 (PaCMAP), 0.55 (UMAP), 0.38 (t-SNE); ferne Paare −0.04 gegen 0.23 / 0.23. Rauschen 0.8: R² 0.88 / 0.90 / 0.80.
2 % Sonderfahrten: R² 0.58 / 0.62 / 0.66 (PCA 0.66, Isomap 0.78); ferne Paare 0.15 / 0.24 / 0.28 (PCA 0.86). Gerade Daten: 0.91 / 0.92 / 0.94 (PCA 0.99).

**Hyperparameter** (300 Touren, 3 Datensätze, Mittel R² / ferne Paare): n_neighbors 3: 0.64 / 0.18, 5: 0.75 / 0.28, 10: 0.88 / 0.33, 20: 0.88 / 0.40, 40: 0.92 / 0.64. MN-Verhältnis 0: 0.73 / 0.06, 0.5: 0.88 / 0.33, 2: 0.84 / 0.33.
FP-Verhältnis 0.5: 0.86 / 0.40, 2: 0.88 / 0.33, 4: 0.82 / 0.25. Iterationen 90: 0.83 / 0.37, 180: 0.89 / 0.47, 270: 0.88 / 0.38, 450: 0.88 / 0.33, 900: 0.86 / 0.29. Die **Live-Sweeps der App** (200 Touren, Seeds 100000–100002)
zeichnen ein zackigeres Bild – z. B. ferne Paare bei MN 0 / 0.25 / 0.5 / 1 / 2: 0.01 / −0.04 / 0.14 / 0.34 / 0.26 (mehr mittlere Paare helfen dort bis Verhältnis 1), bei Iterationen 90 / 180: 0.02 / −0.02, bei n_neighbors 2 / 3 / 5 / 10 / 20 / 40:
0.11 / 0.36 / 0.20 / 0.14 / 0.39 / 0.39. **Die Abstandstreue ferner Paare ist eine unruhige Kennzahl, die stark zwischen Datensätzen und Tourenzahlen schwankt**; belastbar sind: zu wenige nahe Paare schaden (R²), fehlende mittlere Paare
schaden der fernen Ordnung, und das FP-Verhältnis sowie Iterationen über etwa 270 ändern das R² kaum.

**Phasen** (ferne Paare im Mittel über 6 Datensätze, Iteration 60 / 100 / 200 / 300 / 450): 300 Touren 0.51 / 0.53 / 0.44 / 0.41 / 0.36; 200 Touren 0.33 / 0.34 / 0.28 / 0.19 / 0.19 – die globale Ordnung ist meist **am Ende von Phase 1** am höchsten
und erodiert danach (in 10 von 12 Datensätzen). Ein einzelner Datensatz kann davon abweichen.

**Stabilität** (Median der sechs paarweisen Procrustes-Abstände von vier zufälligen Starts, 200 Touren, 4 feste Datensätze; PaCMAP / UMAP / t-SNE): q = 2: 0.11 / 0.01 / 0.48, 0.10 / 0.01 / 0.53, 0.30 / 0.22 / 0.37, 0.43 / 0.11 / 0.43;
q = 3: 0.51 / 0.16 / 0.58, 0.37 / 0.35 / 0.76, 0.10 / 0.05 / 0.68, 0.80 / 0.25 / 0.72. UMAP ist am stabilsten, PaCMAP liegt bei q = 2 klar unter t-SNE, bei q = 3 gemischt (0.80 gegen 0.72 im vierten Datensatz). Auch die
**Paar-Ziehung** beeinflusst das Ergebnis: bei PCA-Start und zwei verschiedenen Paar-Seeds lag der Procrustes-Abstand bei q = 2 zwischen 0.11 und 0.33, bei q = 3 zwischen 0.05 und 0.70.

**Out-of-sample** (letzte 20 % zurückgehalten, 4 feste Seeds): Behelf-`transform` R² 0.79 / 0.91 / 0.84 / 0.82, die offizielle `PaCMAP.transform` 0.81 / 0.93 / 0.58 / 0.81, UMAP `transform` 0.88 / 0.93 / 0.88 / 0.93, t-SNE-Näherung
0.77 / 0.67 / 0.72 / 0.85; Verschiebung der Trainings-Touren beim Neu-Rechnen (PaCMAP) 0.06 / 0.20 / 0.56 / 0.47. Die Referenz weist selbst darauf hin, dass `transform` neue Punkte wie einen zusätzlichen Datensatz behandelt.

**Rechenzeit** (lokale Messung, Standard-Einstellungen, ein Lauf je n; PaCMAP / UMAP / t-SNE): n = 100: 0.07 s / 0.33 s / 0.08 s; n = 200: 0.13 / 0.50 / 0.23–0.28; n = 400: 0.25 / 0.89–0.95 / 2.2–2.5; n = 600: **0.39–0.5 / 1.3–1.7 / 4.9–5.1**.

## Modell und Verfahren

- **Generator** (`pacmap_scenario.py`): wortgleich aus pca-demo; latente Faktoren, 12 Merkmale in 4 Gruppen, Krümmung `κ·B·h(z)`, Rauschen, Sonderfahrten. PaCMAP arbeitet auf z-Werten, danach – wie das Original – global auf [0, 1] skaliert und zentriert.
- **PaCMAP** (`pacmap_algorithm.py`, numpy, ohne `pacmap`, exaktes kNN, n ≤ 600): folgt dem Quellcode der Referenz `pacmap`: nahe Paare = k nächste unter skalierten Abständen `d²/(σᵢσⱼ)` (σ = Mittel der Abstände zum 4.–6. Nachbarn) aus den
  k + 50 nächsten; mittlere Paare = zweitnächste von 6 Zufallstouren; ferne Paare = Zufallstouren ohne nahe Paare (einmal gezogen); Verlust `d̃/(10 + d̃)`, `d̃/(10000 + d̃)`, `1/(1 + d̃)` mit `d̃ = 1 + ‖yᵢ − yⱼ‖²`; Gewichtsschema
  100/100/250 (proportional skaliert); Adam (Lernrate 1); PCA-Start `0.01·PCA`. `transform` als Behelf im Geist von `PaCMAP.transform` (Start am gewichteten Mittel der Nachbarn statt an der PCA). Die Referenz sucht Nachbarn
  näherungsweise (Faiss), hier wird exakt gesucht – gleichwertig, nicht bitgleich.
- **UMAP / t-SNE / Isomap / LLE** (`pacmap_umap.py`, `pacmap_tsne.py`, `pacmap_isomap.py`, `pacmap_lle.py`): wortgleich aus umap-demo / tsne-demo / isomap-demo / lle-demo kopiert (nur Vergleichsverfahren, feste gute Einstellungen).
- **Auswertung** (`pacmap_evaluation.py`): R² der wahren Faktoren aus den zwei Koordinaten per quadratischer Regression; Abstandstreue = Pearson-Korrelation der Paarabstände mit den Faktor-Paarabständen, getrennt für die untere (nah)
  und obere Hälfte (fern); Trustworthiness; Procrustes-Abstand; Sweeps, Stabilitäts- und Out-of-sample-Vergleich, Zeitmessung. Verdicts stützen sich auf einen **Referenzlauf mit den Standard-Paaren** oder auf die Vergleichsverfahren.

## Was nicht funktioniert hat / Grenzen

- **Der Anspruch "bessere globale Struktur als UMAP" hat sich nicht bestätigt** – weder im Standardfall (ferne Paare 0.34 gegen 0.60 über 4 Datensätze) noch bei Sonderfahrten. Auch die Sonderfahrten-Schwäche bleibt.
- **Vorher/nachher-Kurven in den Iterationen waren nicht robust:** ein erster Befund "180 Iterationen erhalten die globale Ordnung besser als 450" (300 Touren, 3 Seeds) ließ sich mit 200 Touren nicht reproduzieren (dort 0.02 / −0.02 bei 90 / 180). Belastbar ist
  nur der Mittelwert-Trend "am Ende von Phase 1 am höchsten"; die App behauptet nichts Feineres. Ebenso zeigten sich beim FP-Verhältnis je nach Datensatzgröße gegenläufige Muster – die Hilfetexte nennen dafür keinen Effekt.
- **Ein "UMAP liegt vorn"-Verdict wurde verworfen:** der R²-Abstand zu UMAP (0.08 im Standardfall bei Seed 7) liegt zu nah an jeder Schwelle und kippt plattformabhängig; die Erfolgsmeldung nennt stattdessen die UMAP-Zahlen und sagt, dass PaCMAP bei fernen Paaren nicht vorn liegt.
- **Chaotischer Optimierer:** die Einzelwerte (z. B. Seed 7: R² 0.80 gegen 0.67 der Referenz) schwanken zwischen Implementierungen und Plattformen; Tests prüfen daher Mittelwerte über mehrere Datensätze und weite Bänder.
- **Grenzen (Text):** Achsen und Abstände im Bild haben keine feste Bedeutung; Clustergrößen und Abstände zwischen Clustern sollten nicht interpretiert werden. Exaktes kNN ist auf ≈ 600 Touren ausgelegt; größere Datensätze brauchen Näherungen,
  die hier nicht gebaut sind. Es gibt kein Konvergenzkriterium – nach der letzten Phase wird abgebrochen.

## Verifikation

- **Gewichtsschema** identisch zu `pacmap.find_weight`; **Gradient und Verlust** gegen `pacmap.pacmap_grad` (1e-3) und finite Differenzen; **Adam-Schritt** gegen `pacmap.update_embedding_adam`; **Paar-Anzahlen** gegen `PaCMAP.decide_num_pairs`
  (inkl. Umordnung bei kleinem n); nahe Paare gegen eine Handrechnung des skalierten kNN und zu > 90 % gegen die Referenz; Struktur der mittleren und fernen Paare (verschieden, keine nahen Paare, nah < mittel < fern im Original).
- Qualität gegen die Referenz über drei Datensätze (R² und Trustworthiness vergleichbar); `transform`: Trainings-Touren als "neu" nahe ihrer Koordinate, deterministisch.
- Generator bit-identisch zu pca-demo, Isomap-/LLE-/t-SNE-/UMAP-Kopien gegen eingefrorene Referenzwerte (mit chaos-tauglichen Toleranzen); Trustworthiness gegen `sklearn.manifold.trustworthiness` (1e-9).
- Die Tabelle "Versprechen" ist als Tests hinterlegt (globale Struktur, Sonderfahrten, Mechanismus ohne MN-Paare, Phasen-Verlauf, Stabilität, Out-of-sample, Rechenzeit); Verdict-Codes; alle 6 Presets in weiten Bändern; AppTest-Rauchtests
  (Default, jedes Preset, jeder Schritt, Randgrößen, n_neighbors folgt n und den Verhältnissen, Experimente auf Abruf), Achsensperre aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🆚 Vergleich mit UMAP, Mathe |
| `pacmap_algorithm.py` | PaCMAP von Grund auf (Paare, Verlust, Gewichtsschema, Adam, Transform) |
| `pacmap_umap.py`, `pacmap_tsne.py`, `pacmap_isomap.py`, `pacmap_lle.py` | Vergleichsverfahren (wortgleich aus umap-demo / tsne-demo / isomap-demo / lle-demo) |
| `pacmap_scenario.py`, `pacmap_constants.py` | Lieferrouten-Generator (wortgleich aus pca-demo), Konstanten, Presets |
| `pacmap_evaluation.py` | Kennzahlen, Sweeps, Verdict, Stabilität, Out-of-sample, Zeitmessung |
| `pacmap_presets.py`, `pacmap_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | pacmap-/umap-/sklearn-/scipy-Kreuzvergleiche, Generator-Referenz, Auswertung, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Dimensionsreduktion: von PCA bis Autoencoder](https://sebastianhanisch.net/konzepte-dimensionsreduktion.html).
