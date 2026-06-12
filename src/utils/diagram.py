"""
python diagram.py  ->  diagram.png
Diagramme technique : architecture du réseau + cycle d'entraînement.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BG = "#0f0f1a"

fig, (ax_arch, ax_cycle) = plt.subplots(1, 2, figsize=(18, 9))
fig.patch.set_facecolor(BG)
fig.suptitle("Neural Network — architecture & cycle d'entraînement", color="white", fontsize=14)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def node(ax, x, y, color, label, sublabel=None, r=0.48):
    circle = plt.Circle((x, y), r, color=color, ec="white", lw=1.2, zorder=4)
    ax.add_patch(circle)
    ax.text(x, y + (0.18 if sublabel else 0), label,
            ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=5)
    if sublabel:
        ax.text(x, y - 0.28, sublabel, ha="center", va="center",
                color="#cccccc", fontsize=7, zorder=5)

def box(ax, x, y, w, h, color, label, sublabel=None):
    rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                   boxstyle="round,pad=0.08",
                                   facecolor=color, edgecolor="white", lw=1.2, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.18 if sublabel else 0), label,
            ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - 0.3, sublabel, ha="center", va="center",
                color="#cccccc", fontsize=7.5, zorder=4)

def arrow(ax, x1, y1, x2, y2, color="white"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, mutation_scale=14),
                zorder=2)

def code(ax, x, y, txt, color="#aaaaff"):
    ax.text(x, y, txt, color=color, fontsize=8, ha="center", va="center",
            fontfamily="monospace", zorder=5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                      edgecolor="#333366", linewidth=1))


# ─────────────────────────────────────────────
# GAUCHE : Architecture réseau
# ─────────────────────────────────────────────
ax = ax_arch
ax.set_facecolor(BG)
ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 10.5)
ax.axis("off")
ax.set_title("Architecture", color="white", fontsize=12, pad=12)

inp_pos    = [(1.5, 7.0), (1.5, 3.0)]
hidden_pos = [(5.0, 8.5), (5.0, 6.0), (5.0, 3.5), (5.0, 1.0)]
out_pos    = (8.5, 5.0)

# Connexions input → hidden
for (ix, iy) in inp_pos:
    for (hx, hy) in hidden_pos:
        ax.plot([ix + 0.48, hx - 0.48], [iy, hy], color="#2a2a4a", lw=0.9, zorder=1)

# Connexions hidden → output
for (hx, hy) in hidden_pos:
    ax.plot([hx + 0.48, out_pos[0] - 0.48], [hy, out_pos[1]], color="#2a2a4a", lw=0.9, zorder=1)

# Noeuds
for i, (x, y) in enumerate(inp_pos):
    node(ax, x, y, "#1d3557", f"x{i+1}")
for j, (x, y) in enumerate(hidden_pos):
    node(ax, x, y, "#2d6a4f", f"h{j+1}", "ReLU")
node(ax, *out_pos, "#7b2d00", "ŷ", "sigmoid")

# Étiquettes couches
ax.text(1.5, 9.8, "INPUT\n(n_in)", ha="center", color="#7ec8e3", fontsize=9)
ax.text(5.0, 9.8, "HIDDEN\n(n_hidden)", ha="center", color="#7ec8e3", fontsize=9)
ax.text(8.5, 9.8, "OUTPUT\n(1)", ha="center", color="#7ec8e3", fontsize=9)

# Formules sur les connexions
ax.text(3.2, 8.2, "z = Σ W1[j][i]·x[i] + b1[j]\nh = ReLU(z)",
        color="#ffd166", fontsize=8, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1500", edgecolor="#ffd166", lw=1))
ax.text(7.0, 8.2, "z = Σ W2[j]·h[j] + b2\nŷ = sigmoid(z)",
        color="#ff6b6b", fontsize=8, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#2a0000", edgecolor="#ff6b6b", lw=1))

# Dimensions des poids
ax.text(3.2, 1.8, "W1 : (n_hidden × n_in)\nb1 : (n_hidden,)",
        color="#888888", fontsize=8, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#111122", edgecolor="#333355", lw=1))
ax.text(7.0, 1.8, "W2 : (n_hidden,)\nb2 : scalaire",
        color="#888888", fontsize=8, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#111122", edgecolor="#333355", lw=1))


# ─────────────────────────────────────────────
# DROITE : Cycle d'entraînement
# ─────────────────────────────────────────────
ax2 = ax_cycle
ax2.set_facecolor(BG)
ax2.set_xlim(0, 10)
ax2.set_ylim(-0.5, 10.5)
ax2.axis("off")
ax2.set_title("Cycle d'entraînement", color="white", fontsize=12, pad=12)

# Boîtes
box(ax2, 1.8, 8.5, 2.8, 1.1, "#1d3557",  "1. DATA",      "(xs, ys)")
box(ax2, 5.2, 8.5, 2.8, 1.1, "#2d6a4f",  "2. FORWARD",   "h, ŷ = forward(x, W, b)")
box(ax2, 8.2, 6.5, 2.8, 1.1, "#7b2d00",  "3. LOSS",      "binary cross-entropy")
box(ax2, 8.2, 4.0, 2.8, 1.1, "#4a1942",  "4. GRADIENTS", "backpropagation")
box(ax2, 5.2, 4.0, 2.8, 1.1, "#1a3a4a",  "5. UPDATE",    "W -= lr × dW")
box(ax2, 1.8, 4.0, 2.8, 1.1, "#2a2a00",  "6. LOOP",      "× epochs")

# Snippets de code
code(ax2, 5.2, 7.2, "h, y_hat = forward(x, W1, b1, W2, b2)")
code(ax2, 8.2, 5.3, "L = -Σ[ y·log(ŷ) + (1-y)·log(1-ŷ) ] / N")
code(ax2, 8.2, 2.8, "dW1, db1, dW2, db2 = gradients(...)")
code(ax2, 5.2, 2.8, "W1 -= lr·dW1\nb1 -= lr·db1\nW2 -= lr·dW2\nb2 -= lr·db2")

# Flèches principales
arrow(ax2, 3.2, 8.5, 3.8, 8.5, "#7ec8e3")
arrow(ax2, 6.6, 8.5, 6.8, 7.0, "#7ec8e3")
arrow(ax2, 8.2, 6.0, 8.2, 4.6, "#7ec8e3")
arrow(ax2, 6.8, 4.0, 6.6, 4.0, "#7ec8e3")
arrow(ax2, 3.8, 4.0, 3.2, 4.0, "#7ec8e3")

# Boucle d'entraînement
ax2.annotate("", xy=(5.2, 7.95), xytext=(1.8, 4.56),
             arrowprops=dict(arrowstyle="-|>", color="#ffd166",
                             lw=2.0, mutation_scale=14,
                             connectionstyle="arc3,rad=0.35"), zorder=2)
ax2.text(2.0, 6.3, "boucle\nd'entraînement",
         color="#ffd166", fontsize=8, ha="center",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1500", edgecolor="#ffd166"))

# Résultat final
ax2.text(5.0, 0.5,
         "Après N epochs : W1, b1, W2, b2 convergent  →  le réseau sépare les classes",
         color="#00ff99", fontsize=8.5, ha="center",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#001a0d", edgecolor="#00ff99", lw=1))


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "diagram.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved -> {out}")
