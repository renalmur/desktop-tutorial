"""Обучение нейросети.

Обучение — это подбор весов. Сети показывают пиксель (14 признаков) и правильный
ответ («облако» или «не облако»). Сеть ошибается, и алгоритм чуть-чуть двигает
веса так, чтобы в следующий раз ошибка стала меньше. Так сотни тысяч раз.

Откуда берутся примеры:
    1) учебные снимки из генератора (cloudai/synth.py) — там ответ известен точно;
    2) настоящие снимки, которые вы разметили во вкладке «Разметка» (data/labels);
    3) готовые маски, если они у вас есть (data/masks): белое — облако, чёрное — нет.
"""
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from cloudai.features import N_FEATURES, compute_features
from cloudai.imageio import ImageReadError, is_image_file, load_image
from cloudai.net import DEFAULT_MODEL_PATH, CloudNet
from cloudai.synth import make_scene

ROOT = Path(__file__).resolve().parent.parent
SYNTH_CACHE = ROOT / "models" / "synth_cache.npz"      # учебные пиксели для обучения «по умолчанию»

# Значения в файле разметки (data/labels/<имя снимка>.png)
LABEL_UNKNOWN, LABEL_CLEAR, LABEL_CLOUD = 0, 1, 2


def build_synthetic_dataset(n_scenes: int = 600, pixels_per_scene: int = 400, seed: int = 0,
                            snow_share: float = 0.0, log=print) -> tuple[np.ndarray, np.ndarray]:
    """Рисует n_scenes учебных снимков и берёт из каждого случайные пиксели с ответами."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    started = time.time()
    for i in range(n_scenes):
        scene = make_scene(seed * 1_000_003 + i, size=int(rng.choice([256, 320, 384])), snow_share=snow_share)
        feats = compute_features(scene["rgb"])[scene["valid"]]
        labels = scene["mask"][scene["valid"]]
        pick = rng.choice(len(labels), size=min(pixels_per_scene, len(labels)), replace=False)
        xs.append(feats[pick])
        ys.append(labels[pick])
        if log and (i + 1) % 100 == 0:
            log(f"  нарисовано снимков: {i + 1} из {n_scenes} ({time.time() - started:.0f} с)")
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.uint8)


def load_synthetic_cached(n_scenes: int = 600, seed: int = 0, snow_share: float = 0.0,
                          log=print) -> tuple[np.ndarray, np.ndarray]:
    """Учебные пиксели из сохранённого файла (чтобы дообучение не ждало генератор) или заново.

    Сохраняется только набор «по умолчанию»; для опытов с другими настройками снимки рисуются заново.
    """
    is_default = n_scenes == 600 and seed == 0 and snow_share == 0.0
    if is_default and SYNTH_CACHE.exists():
        data = np.load(SYNTH_CACHE, allow_pickle=False)
        if data["X"].shape[1] == N_FEATURES:
            return data["X"].astype(np.float32), data["y"]
    X, y = build_synthetic_dataset(n_scenes, seed=seed, snow_share=snow_share, log=log)
    if is_default:
        SYNTH_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(SYNTH_CACHE, X=X.astype(np.float16), y=y)
    return X, y


def label_path_for(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / (image_path.name + ".png")


def load_real_samples(real_dir: Path, labels_dir: Path, masks_dir: Path | None = None,
                      max_per_image: int = 6000, seed: int = 1) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Собирает размеченные пиксели настоящих снимков.

    Возвращает (признаки, ответы, список использованных файлов).
    """
    rng = np.random.default_rng(seed)
    xs, ys, used = [], [], []
    if not real_dir.exists():
        return np.zeros((0, N_FEATURES), np.float32), np.zeros(0, np.uint8), used
    for image_path in sorted(p for p in real_dir.iterdir() if is_image_file(p)):
        label_file = label_path_for(image_path, labels_dir)
        mask_file = (masks_dir / (image_path.stem + ".png")) if masks_dir else None
        if not label_file.exists() and not (mask_file and mask_file.exists()):
            continue
        try:
            image = load_image(image_path)
        except ImageReadError:
            continue
        h, w = image["rgb"].shape[:2]
        labels = np.zeros((h, w), dtype=np.uint8)
        if mask_file and mask_file.exists():                      # полная маска: размечен каждый пиксель
            full = np.asarray(Image.open(mask_file).convert("L").resize((w, h), Image.NEAREST))
            labels = np.where(full > 127, LABEL_CLOUD, LABEL_CLEAR).astype(np.uint8)
        if label_file.exists():                                   # мазки из вкладки «Разметка» важнее маски
            scribble = np.asarray(Image.open(label_file).convert("L").resize((w, h), Image.NEAREST))
            labels = np.where(scribble > 0, scribble, labels).astype(np.uint8)
        known = np.flatnonzero((labels == LABEL_CLEAR) | (labels == LABEL_CLOUD))
        if len(known) == 0:
            continue
        pick = rng.choice(known, size=min(max_per_image, len(known)), replace=False)
        feats = compute_features(image["rgb"]).reshape(-1, N_FEATURES)
        xs.append(feats[pick])
        ys.append((labels.reshape(-1)[pick] == LABEL_CLOUD).astype(np.uint8))
        used.append(image_path.name)
    if not xs:
        return np.zeros((0, N_FEATURES), np.float32), np.zeros(0, np.uint8), used
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys), used


def fit_network(X: np.ndarray, y: np.ndarray, hidden: tuple[int, ...] = (32, 16), seed: int = 0,
                max_iter: int = 60) -> tuple[CloudNet, float]:
    """Подбирает веса сети. Возвращает сеть и точность на отложенных пикселях (0..1)."""
    from sklearn.neural_network import MLPClassifier       # scikit-learn нужен только для обучения

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xn = (X - mean) / std
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(Xn))
    n_val = max(1000, len(Xn) // 8)                        # каждый восьмой пиксель сеть при обучении не видит
    val, train = order[:n_val], order[n_val:]
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", alpha=1e-4, batch_size=512,
                        learning_rate_init=2e-3, max_iter=max_iter, early_stopping=True,
                        n_iter_no_change=6, random_state=seed)
    clf.fit(Xn[train], y[train])
    accuracy = float(clf.score(Xn[val], y[val]))
    net = CloudNet(clf.coefs_, clf.intercepts_, mean, std)
    return net, accuracy


def train_and_save(model_path: Path = DEFAULT_MODEL_PATH, n_scenes: int = 600, seed: int = 0,
                   snow_share: float = 0.0, real_dir: Path | None = None, labels_dir: Path | None = None,
                   masks_dir: Path | None = None, log=print) -> dict:
    """Полный цикл: собрать примеры -> обучить -> сохранить модель. Возвращает сводку."""
    real_dir = real_dir or ROOT / "data" / "real"
    labels_dir = labels_dir or ROOT / "data" / "labels"
    masks_dir = masks_dir or ROOT / "data" / "masks"
    started = time.time()

    log("1/3. Готовлю учебные снимки из генератора…")
    X, y = load_synthetic_cached(n_scenes, seed, snow_share, log)
    n_synth = len(y)

    Xr, yr, used = load_real_samples(real_dir, labels_dir, masks_dir)
    if len(yr):
        # Настоящих пикселей обычно мало. Повторяем их, чтобы они весили не меньше половины учебных.
        repeat = int(np.clip(round(0.5 * n_synth / len(yr)), 1, 50))
        X = np.concatenate([X] + [Xr] * repeat)
        y = np.concatenate([y] + [yr] * repeat)
        log(f"     + настоящие снимки: {len(used)} шт., размеченных пикселей: {len(yr)} (повторены ×{repeat})")
    else:
        log("     настоящих размеченных снимков пока нет — учусь только на учебных")

    log(f"2/3. Обучаю нейросеть на {len(y)} пикселях…")
    net, accuracy = fit_network(X, y, seed=seed)
    net.info = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "synthetic_scenes": n_scenes,
        "snow_share": snow_share,
        "synthetic_pixels": int(n_synth),
        "real_pixels": int(len(yr)),
        "real_images": used,
        "pixel_accuracy_holdout": round(accuracy, 4),
        "seconds": round(time.time() - started, 1),
    }
    log("3/3. Сохраняю модель…")
    net.save(model_path)
    log(f"Готово за {net.info['seconds']} с. Точность на отложенных пикселях: {accuracy * 100:.1f} %")
    return net.info
