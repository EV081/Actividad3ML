"""Propaga etiquetas desde una muestra a la nube completa."""

import numpy as np
from scipy.spatial import cKDTree


def propagate_labels(xyz_sample, labels_sample, xyz_all, k=1):
    """Nearest neighbor o mayoría de k vecinos; empates: etiqueta menor."""
    xyz_sample = np.asarray(xyz_sample, dtype=np.float64)
    xyz_all = np.asarray(xyz_all, dtype=np.float64)
    labels_sample = np.asarray(labels_sample)
    if (xyz_sample.ndim != 2 or xyz_sample.shape[1] != 3
            or xyz_all.ndim != 2 or xyz_all.shape[1] != 3):
        raise ValueError("Las coordenadas deben tener forma (N, 3)")
    if labels_sample.shape != (len(xyz_sample),) or not np.issubdtype(labels_sample.dtype, np.integer):
        raise ValueError("Debe haber una etiqueta entera por punto de la muestra")
    if not isinstance(k, (int, np.integer)) or k < 1 or k > len(xyz_sample):
        raise ValueError("k debe estar entre 1 y el tamaño de la muestra")
    _, neighbors = cKDTree(xyz_sample).query(xyz_all, k=k, workers=-1)
    labels = labels_sample[neighbors]
    if k == 1:
        return labels.astype(int)
    # Ordenar permite contar etiquetas arbitrarias, incluyendo -1, sin bincount.
    labels = np.sort(labels, axis=1)
    best = labels[:, 0].copy()
    best_count = np.zeros(len(labels), dtype=int)
    for column in range(k):
        counts = (labels == labels[:, column, None]).sum(axis=1)
        better = counts > best_count
        best[better] = labels[better, column]
        best_count[better] = counts[better]
    return best.astype(int)
