"""
Cas concret : contrôle qualité d'espresso.
Un café est bon si sa température et sa pression sont proches des valeurs idéales.
Il est mauvais si l'une des deux est trop extrême dans n'importe quelle direction.

Run:  python example.py
Output: example.png
"""
import math
import random
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from main import train, forward

BG, AX_BG, GRID = "#0f0f1a", "#13132b", "#1e1e3a"

def _style(ax, title):
    ax.set_facecolor(AX_BG)
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.tick_params(colors="#888888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.6)


# ─────────────────────────────────────────────
# MAPPING  espace normalisé  <->  unités réelles
#
# Valeurs idéales : 93°C, 9 bars
# 1 unité normalisée = 8°C  /  2 bars
# ─────────────────────────────────────────────
TEMP_CENTER, TEMP_SCALE = 93.0, 8.0   # °C
PRES_CENTER, PRES_SCALE =  9.0, 2.0   # bars

def to_norm(temp, pres):
    return [(temp - TEMP_CENTER) / TEMP_SCALE,
            (pres - PRES_CENTER) / PRES_SCALE]

def to_real(x_norm):
    return [TEMP_CENTER + x_norm[0] * TEMP_SCALE,
            PRES_CENTER + x_norm[1] * PRES_SCALE]


# ─────────────────────────────────────────────
# DONNÉES
# Bon café   (classe 0) : r faible  — proche des valeurs idéales
# Mauvais    (classe 1) : r grand   — trop loin des valeurs idéales
# ─────────────────────────────────────────────
random.seed(42)

def make_coffee_data(n=200, noise=0.12):
    xs, ys = [], []
    for _ in range(n // 2):
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(0.3, 1.0) + random.gauss(0, noise)
        xs.append([r * math.cos(angle), r * math.sin(angle)])
        ys.append(0)
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(2.0, 2.8) + random.gauss(0, noise)
        xs.append([r * math.cos(angle), r * math.sin(angle)])
        ys.append(1)
    return xs, ys

xs_norm, ys_data = make_coffee_data(200)


# ─────────────────────────────────────────────
# ENTRAÎNEMENT
# ─────────────────────────────────────────────
history = train(xs_norm, ys_data, lr=0.1, epochs=500)
W1_f, b1_f, W2_f, b2_f = history[-1][0], history[-1][1], history[-1][2], history[-1][3]
losses = [h[4] for h in history]


# ─────────────────────────────────────────────
# FRONTIÈRE DE DÉCISION
# ─────────────────────────────────────────────
def _boundary(W1, b1, W2, b2, res=80):
    lim  = 3.5
    grid = [lim * (2 * k / res - 1) for k in range(res)]
    ZZ   = [[forward([xv, yv], W1, b1, W2, b2)[1] for xv in grid] for yv in grid]
    return ([TEMP_CENTER + v * TEMP_SCALE for v in grid],
            [PRES_CENTER + v * PRES_SCALE for v in grid],
            ZZ)

temp_grid, pres_grid, ZZ = _boundary(W1_f, b1_f, W2_f, b2_f)

temps_0 = [to_real(p)[0] for p, y in zip(xs_norm, ys_data) if y == 0]
pres_0  = [to_real(p)[1] for p, y in zip(xs_norm, ys_data) if y == 0]
temps_1 = [to_real(p)[0] for p, y in zip(xs_norm, ys_data) if y == 1]
pres_1  = [to_real(p)[1] for p, y in zip(xs_norm, ys_data) if y == 1]


# ─────────────────────────────────────────────
# NOUVEAUX EXPRESSOS À CLASSIFIER
# temp (°C), pression (bars)
# ─────────────────────────────────────────────
new_coffees = [
    (94,  9.2,  "#c77dff", "Espresso A"),   # proche idéal    → bon
    (75,  9.0,  "#06d6a0", "Espresso B"),   # trop froid       → mauvais
    (93, 14.0,  "#ffd166", "Espresso C"),   # trop haute pression → mauvais
    (110, 12.5, "#ff6b6b", "Espresso D"),   # trop chaud + surpression → mauvais
]


# ─────────────────────────────────────────────
# PRÉCISION GLOBALE
# ─────────────────────────────────────────────
correct = sum(
    1 for x, y in zip(xs_norm, ys_data)
    if (forward(x, W1_f, b1_f, W2_f, b2_f)[1] > 0.5) == bool(y)
)
accuracy = correct / len(ys_data)


# ─────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor(BG)
fig.suptitle("Contrôle qualité d'espresso — température vs pression d'extraction",
             color="white", fontsize=13)

gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)


# ── Gauche : frontière de décision ───────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
_style(ax1, "Zone de qualité apprise\nbleu = bon café  /  rouge = mauvais café")

ax1.imshow(ZZ, origin="lower", aspect="auto",
           extent=[temp_grid[0], temp_grid[-1], pres_grid[0], pres_grid[-1]],
           cmap="RdBu_r", vmin=0, vmax=1, alpha=0.55)

ax1.scatter(temps_0, pres_0, color="#7ec8e3", s=22, alpha=0.8, zorder=3,
            label="bon café (classe 0)")
ax1.scatter(temps_1, pres_1, color="#ff6b6b", s=22, alpha=0.8, zorder=3,
            label="mauvais café (classe 1)")

ax1.axvline(TEMP_CENTER, color="#888888", lw=0.8, linestyle=":", alpha=0.5)
ax1.axhline(PRES_CENTER, color="#888888", lw=0.8, linestyle=":", alpha=0.5)
ax1.plot(TEMP_CENTER, PRES_CENTER, "*", color="white", markersize=12, zorder=5,
         label=f"idéal ({TEMP_CENTER}°C, {PRES_CENTER} bars)")

for temp, pres, col, lbl in new_coffees:
    x_norm    = to_norm(temp, pres)
    _, y_hat  = forward(x_norm, W1_f, b1_f, W2_f, b2_f)
    qualite   = "mauvais" if y_hat > 0.5 else "bon"
    ax1.plot(temp, pres, "^", color=col, markersize=11, zorder=6,
             markeredgecolor="white", markeredgewidth=0.8)
    ax1.annotate(f"{lbl}\n{y_hat:.0%} → {qualite}",
                 xy=(temp, pres), xytext=(temp + 2, pres + 0.45),
                 color=col, fontsize=7.5,
                 arrowprops=dict(arrowstyle="->", color=col, lw=1.2))

ax1.set_xlabel("Température d'extraction (°C)", color="#888888", fontsize=8)
ax1.set_ylabel("Pression (bars)", color="#888888", fontsize=8)
ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7.5, loc="upper right")


# ── Droite : courbe de loss ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
_style(ax2, "Loss sur l'entraînement")

ax2.plot(range(len(losses)), losses, color="#7ec8e3", lw=2)
ax2.set_xlabel("epoch", color="#888888", fontsize=8)
ax2.set_ylabel("BCE loss", color="#888888", fontsize=8)

ax2.text(len(losses) * 0.38, losses[0] * 0.78,
         f"précision : {accuracy:.1%}\nloss finale : {losses[-1]:.3f}\n\n"
         f"Le réseau a appris que\nla qualité = distance aux\nvaleurs idéales.\n"
         f"Impossible avec une droite.",
         color="white", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", edgecolor="#333355"))


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "example.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved -> {out}")
print(f"Précision : {accuracy:.1%}")
for temp, pres, col, lbl in new_coffees:
    x_norm = to_norm(temp, pres)
    _, y_hat = forward(x_norm, W1_f, b1_f, W2_f, b2_f)
    print(f"  {lbl} ({temp}°C, {pres} bars)  →  {y_hat:.1%}  ({'mauvais' if y_hat > 0.5 else 'bon'})")
