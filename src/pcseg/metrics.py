"""Métricas sin ground truth; -1 representa ruido."""

from itertools import combinations

import numpy as np
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             silhouette_score)


def cluster_summary(labels):
    labels = np.asarray(labels)
    clusters, counts = np.unique(labels[labels != -1], return_counts=True)
    n_noise = int(np.count_nonzero(labels == -1))
    return {"n_clusters": len(clusters), "n_noise": n_noise,
            "noise_frac": n_noise / len(labels) if len(labels) else 0.0,
            "sizes": {int(label): int(count) for label, count in zip(clusters, counts)}}


def _without_noise(X, labels):
    X, labels = np.asarray(X), np.asarray(labels)
    if X.ndim != 2 or labels.shape != (len(X),):
        raise ValueError("X debe ser (N, d) y labels (N,)")
    keep = labels != -1
    return X[keep], labels[keep]


def _valid_clusters(labels):
    return 2 <= len(np.unique(labels)) < len(labels)


def silhouette_sample(X, labels, n=10000, seed=0):
    """Silhouette sobre una muestra reproducible, excluyendo ruido."""
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n debe ser un entero positivo")
    X, labels = _without_noise(X, labels)
    if len(X) > n:
        idx = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
        X, labels = X[idx], labels[idx]
    if not _valid_clusters(labels):
        return float("nan")
    return float(silhouette_score(X, labels))


def davies_bouldin(X, labels):
    """Davies-Bouldin sin ruido; NaN si la partición no permite calcularlo."""
    X, labels = _without_noise(X, labels)
    if not _valid_clusters(labels):
        return float("nan")
    return float(davies_bouldin_score(X, labels))


def stability_ari(fit_fn, X, seeds):
    """fit_fn(X, seed) devuelve etiquetas; compara todos los pares de corridas.

    Incluye ruido como etiqueta en ARI para detectar cambios de asignación.
    Devuelve (media, std poblacional), o (NaN, NaN) sin pares suficientes.
    """
    labels = [np.asarray(fit_fn(X, seed)) for seed in seeds]
    if any(run.shape != (len(X),) for run in labels):
        raise ValueError("fit_fn debe devolver una etiqueta por fila de X")
    scores = [adjusted_rand_score(a, b) for a, b in combinations(labels, 2)]
    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.std(scores))
