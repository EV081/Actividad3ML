"""Pruebas de la API compartida; el PLY real se usa sólo en lectura/round-trip."""

from pathlib import Path
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

# Permite ejecutar pytest sin instalar el paquete ni modificar configuración.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcseg.features import build_features, voxel_downsample, voxel_ids
from pcseg.io import labels_to_rgb, palette, read_ply, write_ply
from pcseg.metrics import (cluster_summary, davies_bouldin, silhouette_sample,
                           stability_ari)
from pcseg.propagate import propagate_labels
from pcseg.viz import scatter_views


def test_read_real_ply_and_round_trip(tmp_path):
    cloud = read_ply(ROOT / "data/FARO_Scan_Full_Sub1_down3_with_normals.ply")
    assert set(cloud) == {"xyz", "rgb", "normals"}
    for key, dtype in (("xyz", np.float32), ("rgb", np.uint8), ("normals", np.float32)):
        assert cloud[key].shape == (1647021, 3)
        assert cloud[key].dtype == dtype
    output = tmp_path / "round_trip.ply"
    write_ply(output, **cloud)
    restored = read_ply(output)
    for key in cloud:
        assert_array_equal(restored[key], cloud[key])


def test_read_handcrafted_binary(tmp_path):
    # Fixture independiente del writer: verifica orden de bytes y propiedades.
    import struct

    path = tmp_path / "fixture.ply"
    header = ("ply\nformat binary_little_endian 1.0\ncomment ejemplo\n"
              "element vertex 1\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar r\nproperty uchar g\nproperty uchar b\n"
              "property float nx\nproperty float ny\nproperty float nz\nend_header\n")
    path.write_bytes(header.encode("ascii") + struct.pack("<fffBBBfff", 1, -2, 3, 4, 5, 6, 0, 0, 1))
    cloud = read_ply(path)
    assert_array_equal(cloud["xyz"], [[1, -2, 3]])
    assert_array_equal(cloud["rgb"], [[4, 5, 6]])
    assert_array_equal(cloud["normals"], [[0, 0, 1]])


def test_write_without_normals(tmp_path):
    path = tmp_path / "no_normals.ply"
    xyz = np.array([[1, 2, 3]], dtype=np.float32)
    rgb = np.array([[10, 20, 30]], dtype=np.uint8)
    write_ply(path, xyz, rgb)
    cloud = read_ply(path)
    assert_array_equal(cloud["xyz"], xyz)
    assert_array_equal(cloud["rgb"], rgb)
    assert_array_equal(cloud["normals"], np.zeros((1, 3)))


def test_read_rejects_truncated_payload(tmp_path):
    path = tmp_path / "truncated.ply"
    write_ply(path, np.ones((2, 3)), np.zeros((2, 3)))
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncados"):
        read_ply(path)


def test_labels_to_rgb():
    labels = np.array([-1, 0, 1, 20, 21, 100])
    colors = labels_to_rgb(labels)
    assert colors.dtype == np.uint8
    assert colors.shape == (6, 3)
    assert_array_equal(colors, labels_to_rgb(labels))
    assert_array_equal(colors[0], [45, 45, 45])
    assert_array_equal(colors[1], palette[0])
    assert_array_equal(colors[1:3], colors[3:5])
    custom = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    assert_array_equal(labels_to_rgb(labels, custom)[1], custom[0])
    assert_array_equal(labels_to_rgb(labels, custom)[0], [45, 45, 45])


def test_voxel_downsample_and_ids():
    xyz = np.array([[0.2, 0, 0], [0.9, 0, 0], [1.2, 0, 0],
                    [-0.1, 0, 0], [-0.9, 0, 0]])
    idx = voxel_downsample(xyz, 1.0)
    ids = voxel_ids(xyz, 1.0)
    assert_array_equal(idx, [3, 0, 2])
    assert_array_equal(ids, [1, 1, 2, 0, 0])
    assert len(idx) < len(xyz)
    assert_array_equal(idx, voxel_downsample(xyz, 1.0))
    assert_array_equal(ids[idx], np.arange(len(idx)))
    assert_array_equal(np.floor(xyz[idx[ids]]), np.floor(xyz))


@pytest.mark.parametrize("voxel", [0, -1, np.nan, np.inf])
def test_invalid_voxel(voxel):
    with pytest.raises(ValueError):
        voxel_downsample(np.zeros((2, 3)), voxel)


def test_propagation_on_sample():
    xyz = np.random.default_rng(0).normal(size=(100, 3))
    labels = np.arange(100) % 5 - 1
    predicted = propagate_labels(xyz, labels, xyz)
    assert np.issubdtype(predicted.dtype, np.integer)
    assert_array_equal(predicted, labels)
    queries = xyz[[7, 3]] + 1e-9
    assert_array_equal(propagate_labels(xyz, labels, queries), labels[[7, 3]])


def test_propagation_majority_and_ties():
    xyz = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    labels = np.array([7, -1, -1])
    query = np.array([[0.1, 0, 0]])
    assert_array_equal(propagate_labels(xyz, labels, query, k=3), [-1])
    assert_array_equal(propagate_labels(xyz, labels, query, k=2), [-1])
    with pytest.raises(ValueError):
        propagate_labels(xyz, labels, query, k=4)


def test_cluster_summary():
    assert cluster_summary(np.array([-1, 2, 2, 8, -1])) == {
        "n_clusters": 2, "n_noise": 2, "noise_frac": 0.4, "sizes": {2: 2, 8: 1}}
    assert cluster_summary(np.array([], dtype=int)) == {
        "n_clusters": 0, "n_noise": 0, "noise_frac": 0.0, "sizes": {}}


@pytest.mark.parametrize("flags, dimension", [
    ({}, 3), ({"use_rgb": True}, 6),
    ({"use_rgb": True, "use_normals": True, "use_height": True}, 10),
    ({"use_xyz": False, "use_height": True}, 1),
])
def test_build_features(flags, dimension):
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(100, 3))
    rgb = rng.integers(0, 256, size=(100, 3), dtype=np.uint8)
    normals = rng.normal(size=(100, 3))
    features = build_features(xyz, rgb, normals, **flags)
    assert features.shape == (100, dimension)
    assert features.dtype == np.float64
    assert_allclose(features.mean(axis=0), 0, atol=1e-12)
    assert_allclose(features.std(axis=0), 1, atol=1e-12)


def test_lab_and_height_known_values():
    xyz = np.array([[0, 0, 0], [1, 2, 10], [3, 4, 20]])
    rgb = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0]], dtype=np.uint8)
    lab = build_features(xyz, rgb, None, use_xyz=False, use_rgb=True, standardize=False)
    assert_allclose(lab, [[0, 0, 0], [100, 0, 0], [53.2408, 80.0925, 67.2032]], atol=1e-3)
    height = build_features(xyz, None, None, use_xyz=False, use_height=True, standardize=False)
    assert_allclose(height[:, 0], [-0.2, 9.8, 19.8])


def test_metrics_exclude_noise_and_are_deterministic():
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    X = np.array([[0, 0], [0, 1], [10, 0], [10, 1], [5, 500]])
    labels = np.array([2, 2, 8, 8, -1])
    assert silhouette_sample(X, labels) == pytest.approx(silhouette_score(X[:4], labels[:4]))
    assert davies_bouldin(X, labels) == pytest.approx(davies_bouldin_score(X[:4], labels[:4]))
    rng = np.random.default_rng(0)
    large_X = rng.normal(size=(200, 3))
    large_labels = np.arange(200) % 2
    assert silhouette_sample(large_X, large_labels, n=30) == silhouette_sample(large_X, large_labels, n=30)


@pytest.mark.parametrize("labels", [[-1, -1], [0, 0], [0, 1], []])
def test_metrics_undefined(labels):
    X = np.zeros((len(labels), 2))
    assert np.isnan(silhouette_sample(X, labels))
    assert np.isnan(davies_bouldin(X, labels))


def test_stability_ari():
    X = np.zeros((4, 2))
    runs = {0: [0, 0, 1, 1], 1: [7, 7, 3, 3], 2: [0, 1, 0, 1]}
    seen = []

    def fit_fn(X, seed):
        seen.append(seed)
        assert X.shape == (4, 2)
        return runs[seed]

    mean, std = stability_ari(fit_fn, X, [0, 1, 2])
    assert seen == [0, 1, 2]
    assert mean == pytest.approx(0)
    assert std == pytest.approx(np.std([1, -0.5, -0.5]))
    assert all(np.isnan(value) for value in stability_ari(fit_fn, X, [0]))


@pytest.mark.parametrize("as_labels", [True, False])
def test_scatter_views_png(tmp_path, as_labels):
    import matplotlib.pyplot as plt
    from pcseg import viz

    xyz = np.random.default_rng(0).normal(size=(100, 3))
    labels = np.arange(100) % 3 - 1
    values = labels if as_labels else labels_to_rgb(labels)
    path = tmp_path / "views.png"
    before = plt.get_fignums()
    scatter_views(xyz, values, path, "Nube de prueba", max_points=30)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert plt.get_fignums() == before
    assert viz.palette is palette
