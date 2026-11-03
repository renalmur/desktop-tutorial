"""Генератор учебных снимков и обучение (маленький быстрый прогон)."""
import warnings

import numpy as np
from PIL import Image

from cloudai.features import N_FEATURES
from cloudai.net import CloudNet
from cloudai.synth import SCENE_KINDS, make_scene, scene_weights
from cloudai.training import LABEL_CLEAR, LABEL_CLOUD, build_synthetic_dataset, fit_network, load_real_samples
from conftest import flat_image


def test_scene_is_reproducible_and_consistent():
    a, b = make_scene(42, size=128), make_scene(42, size=128)
    assert np.array_equal(a["rgb"], b["rgb"]) and a["percent"] == b["percent"]
    assert a["rgb"].shape == (128, 128, 3) and a["rgb"].min() >= 0 and a["rgb"].max() <= 1
    assert a["percent"] == 100.0 * a["mask"].sum() / a["valid"].sum()


def test_requested_coverage_is_respected():
    for coverage in (0.0, 0.3, 0.8, 1.0):
        scene = make_scene(5, size=192, coverage=coverage, kind="океан", nodata=False, cloud_style="слоистые")
        assert abs(scene["percent"] - 100 * coverage) < 6


def test_every_scene_kind_can_be_drawn():
    for i, kind in enumerate(SCENE_KINDS):
        assert make_scene(100 + i, size=96, kind=kind)["kind"] == kind


def test_scene_weights_sum_to_one():
    for share in (0.0, 0.25, 1.0):
        assert abs(sum(scene_weights(share)) - 1.0) < 1e-9
    assert scene_weights(0.0)[SCENE_KINDS.index("морской лёд")] == 0.0


def test_tiny_training_run_learns_something(tmp_path):
    X, y = build_synthetic_dataset(n_scenes=24, pixels_per_scene=300, seed=3, log=None)
    assert X.shape == (24 * 300, N_FEATURES) and set(np.unique(y)) <= {0, 1}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # сеть не успевает сойтись за 15 шагов — это ожидаемо
        net, accuracy = fit_network(X, y, max_iter=15)
    assert accuracy > 0.85
    net.save(tmp_path / "tiny.npz")
    assert np.allclose(CloudNet.load(tmp_path / "tiny.npz").predict_proba(X[:100]), net.predict_proba(X[:100]))


def test_scribbles_and_full_masks_become_training_pixels(tmp_path):
    real, labels, masks = tmp_path / "real", tmp_path / "labels", tmp_path / "masks"
    for folder in (real, labels, masks):
        folder.mkdir()
    flat_image(real / "a.png", (240, 240, 240), size=(100, 80))
    scribble = np.zeros((80, 100), dtype=np.uint8)
    scribble[:10, :10] = LABEL_CLOUD
    scribble[70:, :20] = LABEL_CLEAR
    Image.fromarray(scribble).save(labels / "a.png.png")
    flat_image(real / "b.jpg", (20, 80, 20), size=(100, 80))
    Image.fromarray(np.zeros((80, 100), dtype=np.uint8)).save(masks / "b.png")       # полная маска: облаков нет
    flat_image(real / "c_unlabeled.png", (100, 100, 100))

    X, y, used = load_real_samples(real, labels, masks)
    assert used == ["a.png", "b.jpg"]
    assert X.shape[1] == N_FEATURES
    assert int(y.sum()) == 100                           # облачных пикселей — только мазок 10×10
    assert len(y) == 100 + 200 + 6000                    # мазки первого снимка + маска второго (не больше 6000 пикселей со снимка)
