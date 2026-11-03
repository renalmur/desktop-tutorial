"""Нейросеть для поиска облаков.

Нейросеть здесь маленькая и честная: 14 чисел на входе, два скрытых слоя и один
выход — «вероятность того, что пиксель — облако». Вся работа сети — умножить,
сложить и «отсечь отрицательное». Ниже это написано тремя строками numpy.

    слой 1:  h1 = ReLU(x · W1 + b1)
    слой 2:  h2 = ReLU(h1 · W2 + b2)
    выход:   p  = sigmoid(h2 · W3 + b3)      число от 0 до 1

W и b — «веса». Их никто не придумывает руками: они подбираются при обучении
(файл train.py) и хранятся в models/cloud_net.npz.
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from cloudai.features import FEATURE_NAMES, N_FEATURES, compute_features, find_nodata

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cloud_net.npz"


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU: отрицательные числа заменяем нулём. Без этого сеть была бы обычной линейкой."""
    return np.maximum(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Сжимает любое число в диапазон 0..1 — получается вероятность."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class CloudNet:
    def __init__(self, weights: list[np.ndarray], biases: list[np.ndarray],
                 mean: np.ndarray, std: np.ndarray, info: dict | None = None):
        if weights[0].shape[0] != N_FEATURES:
            raise ValueError(f"модель ждёт {weights[0].shape[0]} признаков, а код считает {N_FEATURES}")
        self.weights = [w.astype(np.float32) for w in weights]
        self.biases = [b.astype(np.float32) for b in biases]
        self.mean = mean.astype(np.float32)        # средние значения признаков на обучении
        self.std = std.astype(np.float32)          # и их разброс
        self.info = info or {}

    # ---------- сохранение и загрузка ----------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"mean": self.mean, "std": self.std, "n_layers": np.array(len(self.weights))}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"W{i}"] = w
            arrays[f"b{i}"] = b
        np.savez_compressed(path, **arrays)
        info = dict(self.info, feature_names=FEATURE_NAMES,
                    layers=[int(self.weights[0].shape[0])] + [int(w.shape[1]) for w in self.weights])
        path.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "CloudNet":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Нет файла модели {path}. Сначала обучите сеть: python train.py")
        data = np.load(path, allow_pickle=False)
        n = int(data["n_layers"])
        info_path = path.with_suffix(".json")
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        return cls([data[f"W{i}"] for i in range(n)], [data[f"b{i}"] for i in range(n)],
                   data["mean"], data["std"], info)

    # ---------- сама сеть ----------
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Таблица признаков (..., 14) -> вероятность облака для каждого пикселя (...)."""
        shape = features.shape[:-1]
        x = (features.reshape(-1, N_FEATURES) - self.mean) / self.std     # приводим признаки к одному масштабу
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = relu(x @ w + b)                                           # скрытые слои
        out = x @ self.weights[-1] + self.biases[-1]                      # выходной слой
        return sigmoid(out[:, 0]).reshape(shape)

    def analyze(self, rgb: np.ndarray, alpha: np.ndarray | None = None) -> dict:
        """Полный разбор одного снимка.

        Возвращает словарь:
            probability   — карта вероятностей облака (0..1)
            mask          — True там, где облако
            valid         — True там, где есть данные (не чёрное поле)
            cloud_percent — процент облачности от площади с данными
        """
        probability = self.predict_proba(compute_features(rgb))
        # Сглаживаем карту окном 3×3: одиночный «облачный» пиксель среди леса — скорее ошибка,
        # чем облако. Окно маленькое, чтобы не стереть мелкие кучевые облака.
        probability = ndimage.median_filter(probability, size=3, mode="nearest")
        valid = ~find_nodata(rgb, alpha)
        mask = (probability > 0.5) & valid
        n_valid = int(valid.sum())
        cloud_percent = 100.0 * float(mask.sum()) / n_valid if n_valid else 0.0
        return {"probability": probability, "mask": mask, "valid": valid,
                "cloud_percent": cloud_percent, "valid_fraction": n_valid / valid.size}
