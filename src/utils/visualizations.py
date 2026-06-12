"""
All visualizations in a single PNG.
Run:  python visualizations.py
Output: training.png

YOUR TASK: code forward(), loss(), gradients(), and train() in src/main.py,
then replace _run_training() with your own function.
"""
import math
import random
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
random.seed(42)

def _make_circles(n=200, noise=0.12):
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

xs_data, ys_data = _make_circles(200)

xs_0  = [p[0] for p, y in zip(xs_data, ys_data) if y == 0]
ys_c0 = [p[1] for p, y in zip(xs_data, ys_data) if y == 0]
xs_1  = [p[0] for p, y in zip(xs_data, ys_data) if y == 1]
ys_c1 = [p[1] for p, y in zip(xs_data, ys_data) if y == 1]


# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
BG       = "#0f0f1a"
AX_BG    = "#13132b"
GRID_COL = "#1e1e3a"

def _style(ax, title):
    ax.set_facecolor(AX_BG)
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.tick_params(colors="#888888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.5)


# ─────────────────────────────────────────────
# TRAINING PLACEHOLDER
# Replace with:
#   from main import train
#   history = train(xs_data, ys_data, lr=0.1, epochs=500)
# once you have implemented train() in src/main.py.
# Expected history format: [(W1, b1, W2, b2, loss), ...]
# ─────────────────────────────────────────────
def _relu(x):    return max(0.0, x)
def _sigmoid(x): return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))

def _run_training(xs, ys, n_hidden=4, lr=0.1, epochs=500):
    random.seed(0)
    n_in = len(xs[0])
    W1   = [[random.gauss(0, 0.5) for _ in range(n_in)] for _ in range(n_hidden)]
    b1   = [0.0] * n_hidden
    W2   = [random.gauss(0, 0.5) for _ in range(n_hidden)]
    b2   = 0.0
    N    = len(xs)
    history = []

    for _ in range(epochs):
        dW1     = [[0.0] * n_in for _ in range(n_hidden)]
        db1     = [0.0] * n_hidden
        dW2     = [0.0] * n_hidden
        db2_acc = 0.0
        loss_val = 0.0

        for x, y in zip(xs, ys):
            z1    = [sum(W1[j][i] * x[i] for i in range(n_in)) + b1[j] for j in range(n_hidden)]
            h     = [_relu(z) for z in z1]
            z2    = sum(W2[j] * h[j] for j in range(n_hidden)) + b2
            y_hat = _sigmoid(z2)

            loss_val += -(y * math.log(max(y_hat, 1e-15)) + (1 - y) * math.log(max(1 - y_hat, 1e-15)))

            delta_out = y_hat - y
            for j in range(n_hidden):
                dW2[j] += delta_out * h[j]
            db2_acc += delta_out

            delta_h  = [delta_out * W2[j] for j in range(n_hidden)]
            delta_z1 = [delta_h[j] * (1.0 if z1[j] > 0 else 0.0) for j in range(n_hidden)]

            for j in range(n_hidden):
                for i in range(n_in):
                    dW1[j][i] += delta_z1[j] * x[i]
                db1[j] += delta_z1[j]

        W1 = [[W1[j][i] - lr * dW1[j][i] / N for i in range(n_in)] for j in range(n_hidden)]
        b1 = [b1[j]   - lr * db1[j]   / N for j in range(n_hidden)]
        W2 = [W2[j]   - lr * dW2[j]   / N for j in range(n_hidden)]
        b2 = b2        - lr * db2_acc  / N

        history.append((W1, b1, W2, b2, loss_val / N))

    return history


# ─────────────────────────────────────────────
# DECISION BOUNDARY
# ─────────────────────────────────────────────
def _boundary(W1, b1, W2, b2, res=60):
    lim  = 3.5
    grid = [lim * (2 * k / res - 1) for k in range(res)]
    ZZ   = []
    for y_val in grid:
        row = []
        for x_val in grid:
            x     = [x_val, y_val]
            n_h   = len(b1)
            z1    = [sum(W1[j][i] * x[i] for i in range(len(x))) + b1[j] for j in range(n_h)]
            h     = [_relu(z) for z in z1]
            z2    = sum(W2[j] * h[j] for j in range(n_h)) + b2
            row.append(_sigmoid(z2))
        ZZ.append(row)
    return grid, ZZ


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
history = _run_training(xs_data, ys_data)
losses  = [h[4] for h in history]

snap_epochs = [0, 24, 99, len(history) - 1]
snap_colors = ["#ffd166", "#f4a261", "#e76f51", "#00ff99"]
snap_labels = ["epoch 1", "epoch 25", "epoch 100", f"epoch {len(history)} (final)"]


# ─────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(17, 10))
fig.patch.set_facecolor(BG)
fig.suptitle("Neural Network — du problème à la convergence",
             color="white", fontsize=14, y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)


# Panel 1 — données brutes
ax1 = fig.add_subplot(gs[0, 0])
_style(ax1, "Step 1 — Les données\nNon-linéairement séparables")
ax1.scatter(xs_0, ys_c0, color="#7ec8e3", s=22, alpha=0.85, zorder=3, label="classe 0 (intérieur)")
ax1.scatter(xs_1, ys_c1, color="#ff6b6b", s=22, alpha=0.85, zorder=3, label="classe 1 (extérieur)")
ax1.set_aspect("equal")
ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7, loc="upper right")


# Panel 2 — pourquoi une droite échoue
ax2 = fig.add_subplot(gs[0, 1])
_style(ax2, "Step 2 — Une droite ne peut pas séparer\nIl faut déformer l'espace")
ax2.scatter(xs_0, ys_c0, color="#7ec8e3", s=22, alpha=0.7, zorder=3)
ax2.scatter(xs_1, ys_c1, color="#ff6b6b", s=22, alpha=0.7, zorder=3)
for slope in [0.0, 0.8, -0.8, 2.0]:
    ax2.axline((0, 0), slope=slope, color="#ffd166", lw=1.2, alpha=0.4, linestyle="--")
ax2.text(0, 2.8, "n'importe quelle\ndroite échoue", color="#ffd166", fontsize=8, ha="center",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1500", edgecolor="#ffd166"))
ax2.set_xlim(-3.5, 3.5)
ax2.set_ylim(-3.5, 3.5)
ax2.set_aspect("equal")


# Panel 3 — frontière epoch 1
ax3 = fig.add_subplot(gs[0, 2])
W1_e, b1_e, W2_e, b2_e, _ = history[snap_epochs[0]]
_, ZZ = _boundary(W1_e, b1_e, W2_e, b2_e)
_style(ax3, "Step 3 — Epoch 1\nPoids aléatoires, frontière aléatoire")
ax3.imshow(ZZ, origin="lower", aspect="auto",
           extent=[-3.5, 3.5, -3.5, 3.5], cmap="RdBu_r", vmin=0, vmax=1, alpha=0.65)
ax3.scatter(xs_0, ys_c0, color="#7ec8e3", s=18, alpha=0.9, zorder=3)
ax3.scatter(xs_1, ys_c1, color="#ff6b6b", s=18, alpha=0.9, zorder=3)
ax3.set_aspect("equal")


# Panel 4 — frontière epoch 25
ax4 = fig.add_subplot(gs[1, 0])
W1_m, b1_m, W2_m, b2_m, _ = history[snap_epochs[1]]
_, ZZ = _boundary(W1_m, b1_m, W2_m, b2_m)
_style(ax4, "Step 4 — Epoch 25\nLa frontière commence à prendre forme")
ax4.imshow(ZZ, origin="lower", aspect="auto",
           extent=[-3.5, 3.5, -3.5, 3.5], cmap="RdBu_r", vmin=0, vmax=1, alpha=0.65)
ax4.scatter(xs_0, ys_c0, color="#7ec8e3", s=18, alpha=0.9, zorder=3)
ax4.scatter(xs_1, ys_c1, color="#ff6b6b", s=18, alpha=0.9, zorder=3)
ax4.set_aspect("equal")


# Panel 5 — frontière finale
ax5 = fig.add_subplot(gs[1, 1])
W1_f, b1_f, W2_f, b2_f, _ = history[snap_epochs[3]]
_, ZZ = _boundary(W1_f, b1_f, W2_f, b2_f)
_style(ax5, f"Step 5 — Epoch {len(history)} (final)\nLes cercles sont séparés")
ax5.imshow(ZZ, origin="lower", aspect="auto",
           extent=[-3.5, 3.5, -3.5, 3.5], cmap="RdBu_r", vmin=0, vmax=1, alpha=0.65)
ax5.scatter(xs_0, ys_c0, color="#7ec8e3", s=18, alpha=0.9, zorder=3)
ax5.scatter(xs_1, ys_c1, color="#ff6b6b", s=18, alpha=0.9, zorder=3)
ax5.set_aspect("equal")


# Panel 6 — courbe de loss
ax6 = fig.add_subplot(gs[1, 2])
_style(ax6, "Step 6 — Loss par epoch\nConverge vers le minimum")
ax6.plot(range(len(losses)), losses, color="#7ec8e3", lw=2)

for ep, col in zip(snap_epochs, snap_colors):
    ax6.axvline(ep, color=col, linestyle=":", lw=1, alpha=0.7)
    ax6.plot(ep, losses[ep], "o", color=col, markersize=6, zorder=5)

ax6.text(len(losses) * 0.4, losses[0] * 0.75,
         f"loss finale : {losses[-1]:.3f}",
         color="white", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="#333355"))
ax6.set_xlabel("epoch", color="#888888", fontsize=8)
ax6.set_ylabel("BCE loss", color="#888888", fontsize=8)


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "training.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved -> {out}")
