"""Vistas de la nube con matplotlib Agg, sin ventana gráfica."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import labels_to_rgb, palette


def scatter_views(xyz, rgb_or_labels, path, title, max_points=200000, seed=0):
    """Guarda PNG con vista superior, lateral y 3D de la misma muestra."""
    xyz = np.asarray(xyz)
    values = np.asarray(rgb_or_labels)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz debe tener forma (N, 3)")
    if values.shape not in ((len(xyz),), xyz.shape):
        raise ValueError("Se espera una etiqueta o tres canales RGB por punto")
    if not isinstance(max_points, (int, np.integer)) or max_points < 1:
        raise ValueError("max_points debe ser positivo")
    if len(xyz) > max_points:
        idx = np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)
        xyz, values = xyz[idx], values[idx]
    colors = labels_to_rgb(values) / 255.0 if values.ndim == 1 else values.astype(float) / 255.0
    fig = plt.figure(figsize=(18, 6))
    try:
        for position, (vertical, ylabel, caption) in enumerate(
                ((1, "y", "Superior"), (2, "z", "Lateral")), start=1):
            ax = fig.add_subplot(1, 3, position)
            ax.scatter(xyz[:, 0], xyz[:, vertical], c=colors, s=0.5, linewidths=0)
            ax.set(xlabel="x", ylabel=ylabel, title=caption, aspect="equal")
        ax = fig.add_subplot(1, 3, 3, projection="3d")
        ax.scatter(*xyz.T, c=colors, s=0.5, linewidths=0)
        ax.set(xlabel="x", ylabel="y", zlabel="z", title="3D")
        if len(xyz):
            ax.set_box_aspect(np.maximum(np.ptp(xyz, axis=0), 1e-9))
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, format="png", dpi=150)
    finally:
        plt.close(fig)
