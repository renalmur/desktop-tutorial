"""Нейросеть: здравый смысл и точность на учебных снимках."""
import numpy as np
import pytest

from cloudai.features import N_FEATURES
from cloudai.net import CloudNet, relu, sigmoid
from cloudai.synth import make_scene


def test_activation_functions():
    assert relu(np.array([-2.0, 0.0, 3.0])).tolist() == [0.0, 0.0, 3.0]
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)
    assert 0.0 < sigmoid(np.array([-100.0]))[0] < 1e-6        # не ломается на больших числах


def test_model_matches_feature_count(model):
    assert model.weights[0].shape[0] == N_FEATURES
    assert model.weights[-1].shape[1] == 1


def test_probabilities_are_between_zero_and_one(model):
    rgb = np.random.default_rng(1).random((64, 64, 3)).astype(np.float32)
    result = model.analyze(rgb)
    assert result["probability"].min() >= 0.0 and result["probability"].max() <= 1.0
    assert 0.0 <= result["cloud_percent"] <= 100.0


@pytest.mark.parametrize("color, expected", [
    ((0.10, 0.25, 0.10), "ясно"),      # лес
    ((0.03, 0.08, 0.15), "ясно"),      # океан
    ((0.60, 0.45, 0.30), "ясно"),      # почва
    ((0.95, 0.95, 0.97), "облачно"),   # сплошная белая облачность
    ((0.80, 0.80, 0.82), "облачно"),   # сплошная серая облачность
])
def test_flat_scenes(model, color, expected):
    rgb = np.ones((120, 120, 3), dtype=np.float32) * np.array(color, dtype=np.float32)
    percent = model.analyze(rgb)["cloud_percent"]
    if expected == "ясно":
        assert percent < 5.0
    else:
        assert percent > 95.0


def test_accuracy_on_unseen_scenes(model):
    """40 снимков, которых сеть не видела при обучении: средняя ошибка процента — не больше 4 пунктов."""
    kinds = ["океан", "побережье", "лес и поля", "пустыня", "город и поля"]
    errors, decisions = [], []
    for i in range(40):
        scene = make_scene(7_000_000 + i, size=256, coverage=i / 39.0, kind=kinds[i % 5], nodata=False)
        predicted = model.analyze(scene["rgb"])["cloud_percent"]
        errors.append(abs(predicted - scene["percent"]))
        if abs(scene["percent"] - 75.0) > 8.0:                 # вдали от порога решение обязано совпасть
            decisions.append((predicted > 75.0) == (scene["percent"] > 75.0))
    assert np.mean(errors) < 4.0
    assert all(decisions)


def test_nodata_is_excluded_from_percent(model):
    """Половина кадра — чёрное поле, вторая половина — сплошные облака: облачность 100 %, а не 50 %."""
    rgb = np.zeros((120, 120, 3), dtype=np.float32)
    rgb[:, 60:] = 0.93
    result = model.analyze(rgb)
    assert result["valid_fraction"] == pytest.approx(0.5, abs=0.02)
    assert result["cloud_percent"] > 95.0


def test_save_and_load_give_same_answers(model, tmp_path):
    model.save(tmp_path / "copy.npz")
    copy = CloudNet.load(tmp_path / "copy.npz")
    feats = np.random.default_rng(2).random((500, N_FEATURES)).astype(np.float32)
    assert np.allclose(model.predict_proba(feats), copy.predict_proba(feats))
    assert (tmp_path / "copy.json").exists()


def test_missing_model_gives_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        CloudNet.load(tmp_path / "нет_такой.npz")
