"""Reproduce the figures in the Noncommutative Shear Flows poster.

Run: python code/generate_figures.py
Requirements: numpy, matplotlib. These are area-preserving discrete maps,
with no molecular diffusion.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True, parents=True)

INK = "#23333B"
NAVY = "#21556B"
RUST = "#B35E40"
MUTED = "#5B6970"
BORDER = "#BEC9CC"
N = 16_000
SEED = 630
CENTER = (0.27, 0.39)
RADIUS = .12
STEPS = (0, 2, 4, 6)

A = np.array([[1, 1], [0, 1]], dtype=np.int64)
B = np.array([[1, 0], [1, 1]], dtype=np.int64)
M = A @ B
C = A @ np.array([[1, 0], [-1, 1]], dtype=np.int64)


def initial_patch():
    rng = np.random.default_rng(SEED)
    radii = RADIUS * np.sqrt(rng.random(N))
    angles = 2 * np.pi * rng.random(N)
    xy = np.column_stack((CENTER[0] + radii * np.cos(angles),
                          CENTER[1] + radii * np.sin(angles)))
    return xy, xy[:, 0] < CENTER[0]


def advance(xy, matrix, repetitions):
    out = xy.copy()
    for _ in range(repetitions):
        out = (out @ matrix.T) % 1.
    return out


def grid_variance(xy, bins=32):
    counts, _, _ = np.histogram2d(xy[:, 0], xy[:, 1],
                                  bins=bins, range=[[0, 1], [0, 1]])
    expected = len(xy) / (bins * bins)
    return float(np.mean((counts / expected - 1.) ** 2))


def save(fig, name, dpi=300):
    fig.savefig(FIGDIR / f"{name}.pdf", dpi=dpi, facecolor="white")
    fig.savefig(FIGDIR / f"{name}.png", dpi=180, facecolor="white")
    plt.close(fig)


def make_shear_schematic():
    rect = np.array([[-.28, -.17], [.28, -.17], [.28, .17], [-.28, .17]])
    examples = [("initial", np.eye(2, dtype=int)),
                ("A", A), ("B", B), ("AB", M), ("BA", B @ A)]
    fig, axes = plt.subplots(1, 5, figsize=(14.0, 3.15))
    for ax, (name, matrix) in zip(axes, examples):
        corners = rect @ matrix.T
        loop = np.vstack((corners, corners[:1]))
        ax.axvline(0, color="#DBE1E0", linewidth=.9, zorder=0)
        ax.axhline(0, color="#DBE1E0", linewidth=.9, zorder=0)
        ax.add_patch(Polygon(corners, closed=True,
                             facecolor=NAVY if name != "BA" else RUST,
                             edgecolor="none", alpha=.21))
        ax.plot(loop[:, 0], loop[:, 1], lw=2.5,
                color=NAVY if name != "BA" else RUST)
        ax.set(xlim=(-.82, .82), ylim=(-.82, .82))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=21, color=INK, pad=10, fontweight="medium")
        for spine in ax.spines.values():
            spine.set_linewidth(.85)
            spine.set_edgecolor(BORDER)
    fig.subplots_adjust(left=.013, right=.988, bottom=.025, top=.85, wspace=.18)
    save(fig, "01_shear_schematic")


def make_protocol_comparison():
    points, left = initial_patch()
    cases = [
        ("ONE DIRECTION", r"$A^{2n}$", "algebraic", A, 2),
        ("ALTERNATING", r"$(AB)^n$", "exponential", M, 1),
        ("OPPOSITE SIGN", r"$C^n$", "periodic", C, 1),
    ]
    fig, axs = plt.subplots(3, 4, figsize=(18.0, 10.25),
                            sharex=True, sharey=True)
    for row, (name, expr, behavior, matrix, stages) in enumerate(cases):
        for col, n in enumerate(STEPS):
            ax = axs[row, col]
            loc = advance(points, matrix, n * stages)
            ax.scatter(loc[left, 0], loc[left, 1], s=2.4,
                       marker=".", c=NAVY, alpha=.93,
                       linewidths=0, rasterized=True)
            ax.scatter(loc[~left, 0], loc[~left, 1], s=2.4,
                       marker=".", c=RUST, alpha=.93,
                       linewidths=0, rasterized=True)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([0, .5, 1.])
            ax.set_yticks([0, .5, 1.])
            ax.tick_params(length=3., width=.75, colors=MUTED,
                           labelsize=12, pad=4)
            if row != 2:
                ax.tick_params(labelbottom=False)
            if col != 0:
                ax.tick_params(labelleft=False)
            for s in ax.spines.values():
                s.set_color(BORDER)
                s.set_linewidth(.9)
            if row == 0:
                ax.set_title(f"{n} cycles", fontsize=21, color=INK, pad=15)
        bb = axs[row, 0].get_position()
        yy = (bb.y0 + bb.y1) / 2
        fig.text(.012, yy+.039, name, fontsize=17.7,
                 color=INK, fontweight="bold", ha="left", va="center")
        fig.text(.012, yy-.005, expr, fontsize=17.0,
                 color=NAVY, ha="left", va="center")
        fig.text(.012, yy-.047, behavior, fontsize=15.0,
                 color=MUTED, ha="left", va="center")
    fig.subplots_adjust(left=.205, right=.990, top=.951, bottom=.052,
                        wspace=.075, hspace=.13)
    save(fig, "02_protocol_comparison", dpi=330)


def verify():
    assert np.array_equal(M, [[2, 1], [1, 1]])
    assert np.array_equal(C, [[0, 1], [-1, 1]])
    assert np.array_equal(np.linalg.matrix_power(C, 6), np.eye(2, dtype=np.int64))
    assert int(round(np.linalg.det(M))) == 1
    assert int(round(np.linalg.det(C))) == 1
    points, _ = initial_patch()
    results = {}
    for name, mat, stages in (("initial", np.eye(2, dtype=np.int64), 0),
                              ("one_direction", A, 2),
                              ("alternating", M, 1),
                              ("opposite_sign", C, 1)):
        results[name] = grid_variance(advance(points, mat, 6 * stages))
    assert abs(results["initial"] - results["opposite_sign"]) < 1e-9
    assert abs(results["alternating"] - 0.066544) < 2e-6
    print("Grid variance (32 x 32 cells, n = 6):")
    for k, v in results.items():
        print(f"{k:>15s}: {v:.6f}")
    print(f"uniform-sampling benchmark: {(32*32-1)/N:.7f}")


if __name__ == "__main__":
    verify()
    make_shear_schematic()
    make_protocol_comparison()
