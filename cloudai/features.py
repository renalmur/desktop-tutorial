"""Признаки: что нейросеть узнаёт о каждом пикселе.

Один пиксель сам по себе — это три числа (R, G, B). Этого мало: белый пиксель
может быть облаком, снегом или крышей склада. Поэтому мы добавляем сведения
о соседях: насколько ярко вокруг, гладко там или пёстро, есть ли рядом тёмное.

Как человек отличает облако:
    * оно светлое            -> признаки «яркость», «самый тёмный канал»
    * оно белое или серое    -> признак «насыщенность цвета» (у облака маленькая)
    * оно большое и мягкое   -> средняя яркость вокруг, разброс яркости, минимум вокруг
Нейросеть получает эти 14 чисел и сама учится, как их сочетать.
"""
import numpy as np
from scipy import ndimage

FEATURE_NAMES = [
    "красный",                  # 0  канал R
    "зелёный",                  # 1  канал G
    "синий",                    # 2  канал B
    "яркость (max)",            # 3  самый яркий из трёх каналов
    "самый тёмный канал",       # 4  у белого он большой, у цветного — маленький
    "насыщенность",             # 5  0 = серый/белый, 1 = чистый цвет
    "синий минус красный",      # 6  вода и дымка синее, пустыня и почва краснее
    "яркость вокруг 5",         # 7  средняя яркость в окне 5×5
    "яркость вокруг 15",        # 8  ... в окне 15×15
    "яркость вокруг 41",        # 9  ... в окне 41×41 (крупный план)
    "пестрота 5",               # 10 разброс яркости в окне 5×5 (гладко или рябит)
    "пестрота 15",              # 11 ... в окне 15×15
    "минимум вокруг 7",         # 12 самое тёмное место рядом: у маленькой белой крыши рядом темно
    "пестрота 41",              # 13 разброс яркости в большом окне 41×41
]
N_FEATURES = len(FEATURE_NAMES)


def _local_std(gray: np.ndarray, size: int) -> np.ndarray:
    """Разброс яркости в окне size×size: 0 — идеально гладко, больше — пёстро."""
    mean = ndimage.uniform_filter(gray, size, mode="nearest")
    mean_sq = ndimage.uniform_filter(gray * gray, size, mode="nearest")
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def compute_features(rgb: np.ndarray) -> np.ndarray:
    """Снимок высота × ширина × 3 (числа 0..1) -> таблица признаков высота × ширина × 14."""
    rgb = np.asarray(rgb, dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    brightest = rgb.max(axis=-1)
    darkest = rgb.min(axis=-1)
    gray = rgb.mean(axis=-1)
    saturation = (brightest - darkest) / (brightest + 0.05)

    feats = np.empty(rgb.shape[:2] + (N_FEATURES,), dtype=np.float32)
    feats[..., 0] = r
    feats[..., 1] = g
    feats[..., 2] = b
    feats[..., 3] = brightest
    feats[..., 4] = darkest
    feats[..., 5] = saturation
    feats[..., 6] = b - r
    feats[..., 7] = ndimage.uniform_filter(gray, 5, mode="nearest")
    feats[..., 8] = ndimage.uniform_filter(gray, 15, mode="nearest")
    feats[..., 9] = ndimage.uniform_filter(gray, 41, mode="nearest")
    feats[..., 10] = _local_std(gray, 5)
    feats[..., 11] = _local_std(gray, 15)
    feats[..., 12] = ndimage.minimum_filter(gray, 7, mode="nearest")
    feats[..., 13] = _local_std(gray, 41)
    return feats


def find_nodata(rgb: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray:
    """Маска «нет данных»: чёрные поля по краям кадра и прозрачные пиксели.

    Спутник иногда снимает под углом, и по краям файла остаются чёрные треугольники.
    Это не Земля и не облака, поэтому в подсчёте процента они не участвуют.
    Чёрное пятно посреди снимка (например, ночное озеро) полем не считается:
    мы берём только чёрные области, которые касаются края кадра.
    """
    nodata = np.zeros(rgb.shape[:2], dtype=bool)
    if alpha is not None:
        nodata |= alpha < 0.5
    black = rgb.max(axis=-1) < 0.03
    if black.any():
        labels, count = ndimage.label(black)
        if count:
            border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
            sizes = ndimage.sum_labels(black, labels, index=np.arange(1, count + 1))
            min_size = 0.005 * black.size          # мелкие чёрные точки у края — просто тени
            for label in border[border > 0]:
                if sizes[label - 1] >= min_size:
                    nodata |= labels == label
    return nodata
