"""Voxel downsampling y features geométricas y de color."""

import numpy as np
from sklearn.preprocessing import StandardScaler


def _voxel_partition(xyz, voxel):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("xyz debe tener forma (N, 3) y valores finitos")
    if not np.isscalar(voxel) or not np.isfinite(voxel) or voxel <= 0:
        raise ValueError("voxel debe ser positivo y finito")
    cells = np.floor(xyz / voxel)
    _, idx, inverse = np.unique(cells, axis=0, return_index=True, return_inverse=True)
    return idx, inverse


def voxel_downsample(xyz, voxel):
    """Devuelve el índice del primer punto de cada voxel, en orden espacial."""
    return _voxel_partition(xyz, voxel)[0]


def voxel_ids(xyz, voxel):
    """IDs contiguos: el ID i corresponde al representante idx[i]."""
    return _voxel_partition(xyz, voxel)[1]


def _srgb_to_lab(rgb):
    # sRGB de 8 bits -> RGB lineal -> XYZ D65 -> CIE Lab.
    srgb = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92,
                      ((srgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
    relative = (linear @ matrix.T) / [0.95047, 1.0, 1.08883]
    delta = 6 / 29
    f = np.where(relative > delta**3, np.cbrt(relative),
                 relative / (3 * delta**2) + 4 / 29)
    return np.column_stack((116 * f[:, 1] - 16,
                            500 * (f[:, 0] - f[:, 1]),
                            200 * (f[:, 1] - f[:, 2])))


def build_features(xyz, rgb, normals, use_xyz=True, use_rgb=False,
                   use_normals=False, use_height=False, standardize=True):
    """Concatena las features elegidas y opcionalmente aplica StandardScaler."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz debe tener forma (N, 3)")
    blocks = []
    if use_xyz:
        blocks.append(xyz)
    for enabled, values, name in ((use_rgb, rgb, "rgb"),
                                   (use_normals, normals, "normals")):
        if enabled:
            values = np.asarray(values, dtype=np.float64)
            if values.shape != xyz.shape:
                raise ValueError(f"{name} debe tener forma (N, 3)")
            blocks.append(_srgb_to_lab(values) if name == "rgb" else values)
    if use_height:
        if len(xyz) == 0:
            raise ValueError("height requiere al menos un punto")
        blocks.append((xyz[:, 2] - np.percentile(xyz[:, 2], 1))[:, None])
    if not blocks:
        raise ValueError("Selecciona al menos un grupo de features")
    features = np.concatenate(blocks, axis=1)
    return StandardScaler().fit_transform(features) if standardize else features
