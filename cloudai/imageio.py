"""Чтение снимков с диска.

Компьютер хранит цветную картинку как три таблицы чисел: красную (R), зелёную (G)
и синюю (B). Число 0 — канал выключен, 255 — горит на полную. Мы переводим всё
в диапазон от 0 до 1: так нейросети удобнее считать.
"""
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

# Спутниковые снимки бывают очень большими. Разрешаем до 400 мегапикселей,
# а не отключаем защиту совсем: гигантский «битый» файл не должен повесить компьютер.
Image.MAX_IMAGE_PIXELS = 400_000_000

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Нейросеть работает с уменьшенной копией снимка: длинная сторона — 384 пикселя.
# Облака — крупные объекты, им не нужна каждая мелкая деталь, а считать в сотни раз быстрее.
WORK_SIZE = 384


class ImageReadError(Exception):
    """Файл не удалось прочитать как картинку."""


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def _to_float_rgb(img: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
    """Любой формат картинки -> массив высота × ширина × 3 с числами 0..1 и (если есть) прозрачность."""
    alpha = None
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        # 16-битные снимки (часто TIFF): числа до 65535. Растягиваем по самым ярким пикселям.
        arr = np.asarray(img, dtype=np.float32)
        top = float(np.percentile(arr, 99.9)) or 1.0
        gray = np.clip(arr / top, 0.0, 1.0)
        return np.stack([gray, gray, gray], axis=-1), None
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        alpha = np.asarray(rgba, dtype=np.float32)[..., 3] / 255.0
        img = rgba.convert("RGB")
    else:
        img = img.convert("RGB")           # чёрно-белый снимок превращается в три одинаковых канала
    return np.asarray(img, dtype=np.float32) / 255.0, alpha


def load_image(path: str | Path, work_size: int = WORK_SIZE) -> dict:
    """Читает снимок и готовит уменьшенную копию для нейросети.

    Возвращает словарь:
        rgb    — уменьшенный снимок, числа 0..1, размер высота × ширина × 3
        alpha  — прозрачность (или None): прозрачные пиксели — это «нет данных»
        width, height — размер исходного снимка в пикселях
    """
    path = Path(path)
    try:
        with Image.open(path) as img:
            width, height = img.size
            if width < 8 or height < 8:
                raise ImageReadError(f"снимок слишком маленький: {width}×{height}")
            img.load()
            scale = work_size / max(width, height)
            if scale < 1.0:
                new_size = (max(8, round(width * scale)), max(8, round(height * scale)))
                if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
                    img = img.resize(new_size, Image.BILINEAR)
                else:
                    # LANCZOS аккуратно усредняет соседние пиксели — мелкие детали не «рябят»
                    img = img.resize(new_size, Image.LANCZOS)
            rgb, alpha = _to_float_rgb(img)
    except ImageReadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ImageReadError(f"файл не читается как картинка ({exc.__class__.__name__})") from exc
    return {"rgb": rgb, "alpha": alpha, "width": width, "height": height}
