"""Генератор учебных снимков.

Чтобы научить нейросеть, нужны примеры с ответами: снимок и маска, где отмечен
каждый облачный пиксель. Размечать сотни настоящих снимков вручную долго,
поэтому первую «школу» сеть проходит на нарисованных программой снимках:
программа сама рисует землю, море, снег и облака и поэтому точно знает,
где облако. Потом сеть дообучается на настоящих снимках (см. вкладку «Разметка»).

Главный инструмент — фрактальный шум: случайное поле, в котором есть и крупные
пятна, и мелкая рябь. Из него получаются и горы, и береговая линия, и облака.
"""
import io

import numpy as np
from PIL import Image
from scipy import ndimage

SCENE_KINDS = ["океан", "побережье", "лес и поля", "пустыня", "горы со снегом",
               "зима (всё в снегу)", "морской лёд", "город и поля"]
SNOW_KINDS = ("горы со снегом", "зима (всё в снегу)", "морской лёд")


def scene_weights(snow_share: float = 0.0) -> list[float]:
    """Как часто каждый тип местности попадает в обучение.

    snow_share — доля снежных и ледовых снимков (0..1). По умолчанию 0: мы проверили, что
    сеть, которую учили на снегу, начинает пропускать настоящие белые облака (см. GUIDE_RU.md).
    """
    plain = {"океан": 0.18, "побережье": 0.22, "лес и поля": 0.22, "пустыня": 0.18, "город и поля": 0.20}
    snow = {"горы со снегом": 0.4, "зима (всё в снегу)": 0.4, "морской лёд": 0.2}
    return [snow[k] * snow_share if k in snow else plain[k] * (1 - snow_share) for k in SCENE_KINDS]


# Облаком считаем пиксель, где облако закрывает землю хотя бы на 35 %.
CLOUD_ALPHA_THRESHOLD = 0.35


def fractal(shape: tuple[int, int], beta: float, rng: np.random.Generator,
            stretch: tuple[float, float] = (1.0, 1.0), detail: float = 0.0) -> np.ndarray:
    """Фрактальный шум 0..1.

    beta    — больше: пятна крупнее и мягче;
    stretch — вытягивает шум в полосы;
    detail  — сколько самых крупных пятен убрать (0 — ничего; 6 — остаются пятна мельче 1/6 кадра).
    """
    h, w = shape
    spectrum = np.fft.fft2(rng.standard_normal((h, w)))
    fy = np.fft.fftfreq(h)[:, None] * stretch[0]
    fx = np.fft.fftfreq(w)[None, :] * stretch[1]
    k = np.sqrt(fx * fx + fy * fy)
    k[0, 0] = 1.0
    amplitude = 1.0 / k ** beta
    if detail > 0:
        amplitude = amplitude * (1.0 - np.exp(-(k * max(h, w) / detail) ** 2))
    field = np.real(np.fft.ifft2(spectrum * amplitude))
    field -= field.min()
    peak = field.max()
    return (field / peak if peak > 0 else field).astype(np.float32)


def _mix(colors: list[tuple[float, float, float]], t: np.ndarray) -> np.ndarray:
    """Плавный переход между цветами палитры по значению t (0..1)."""
    palette = np.asarray(colors, dtype=np.float32)
    pos = np.clip(t, 0.0, 1.0) * (len(palette) - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(palette) - 1)
    frac = (pos - i0)[..., None]
    return palette[i0] * (1 - frac) + palette[i1] * frac


def _hillshade(elevation: np.ndarray, strength: float) -> np.ndarray:
    """Свет сбоку: склоны к солнцу светлее, от солнца темнее. Так на снимке проступают горы."""
    gy, gx = np.gradient(elevation)
    shade = 1.0 + strength * (gx * 0.8 + gy * 0.6) * elevation.shape[1]
    return np.clip(shade, 0.45, 1.6).astype(np.float32)


def _fields(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Лоскутное одеяло сельхозполей: прямоугольники разных оттенков."""
    h, w = shape
    cell = int(rng.integers(8, 28))
    angle = rng.uniform(0, np.pi / 2)
    yy, xx = np.mgrid[0:h, 0:w]
    u = (xx * np.cos(angle) + yy * np.sin(angle)) // cell
    v = (-xx * np.sin(angle) + yy * np.cos(angle)) // (cell * rng.uniform(0.6, 1.8))
    ids = (u.astype(np.int64) * 7919 + v.astype(np.int64) * 104729) % 9973
    table = rng.random(9973).astype(np.float32)
    palette = [(0.20, 0.33, 0.12), (0.45, 0.42, 0.20), (0.62, 0.55, 0.32), (0.30, 0.22, 0.14),
               (0.36, 0.50, 0.20), (0.70, 0.66, 0.45)]
    return _mix(palette, table[ids])


def make_ground(shape: tuple[int, int], kind: str, rng: np.random.Generator) -> np.ndarray:
    """Рисует поверхность Земли без облаков. Возвращает картинку 0..1."""
    h, w = shape
    elevation = fractal(shape, rng.uniform(2.0, 2.6), rng)
    moisture = fractal(shape, rng.uniform(1.8, 2.4), rng)
    fine = fractal(shape, 1.0, rng) - 0.5                       # мелкая рябь

    # --- суша ---
    if kind == "пустыня":
        land = _mix([(0.80, 0.68, 0.48), (0.72, 0.56, 0.38), (0.62, 0.42, 0.28), (0.86, 0.80, 0.66)], moisture)
    elif kind in ("лес и поля", "город и поля"):
        land = _mix([(0.07, 0.17, 0.07), (0.14, 0.27, 0.10), (0.26, 0.36, 0.14), (0.42, 0.40, 0.22)], moisture)
    else:
        land = _mix([(0.08, 0.18, 0.08), (0.22, 0.32, 0.13), (0.46, 0.42, 0.24), (0.60, 0.45, 0.30),
                     (0.74, 0.64, 0.46)], moisture)
    if kind in ("лес и поля", "город и поля") or rng.random() < 0.2:
        flat = ndimage.gaussian_filter((moisture > np.quantile(moisture, rng.uniform(0.3, 0.6))).astype(np.float32), 2)
        land = land * (1 - flat[..., None]) + _fields(shape, rng) * flat[..., None]
    land = land * (1 + 0.5 * fine[..., None])
    land = land * _hillshade(elevation, rng.uniform(0.3, 1.2))[..., None]

    if kind == "город и поля":
        town = fractal(shape, 2.4, rng)
        town = np.clip((town - np.quantile(town, rng.uniform(0.75, 0.92))) * 12, 0, 1)
        speckle = rng.random(shape).astype(np.float32)
        gray = 0.32 + 0.30 * speckle
        land = land * (1 - town[..., None]) + np.stack([gray, gray * 0.98, gray * 0.95], -1) * town[..., None]
        roofs = (speckle > 0.985) & (town > 0.4)                 # яркие крыши складов — «ловушка» для сети
        land[roofs] = rng.uniform(0.85, 1.0)

    # --- снег ---
    if kind == "горы со снегом":
        snowline = np.quantile(elevation, rng.uniform(0.45, 0.8))
        snow = np.clip((elevation - snowline) * rng.uniform(15, 40), 0, 1)
    elif kind == "зима (всё в снегу)":
        snow = np.clip(0.75 + 0.5 * (moisture - 0.5) + rng.uniform(-0.1, 0.25), 0, 1)
        forest = np.clip((fractal(shape, 2.2, rng) - rng.uniform(0.45, 0.7)) * 10, 0, 1)
        snow = snow * (1 - 0.85 * forest)                        # тёмная тайга торчит из-под снега
    else:
        snow = np.zeros(shape, dtype=np.float32)
    if snow.any():
        snow_color = rng.uniform(0.74, 0.97) * _hillshade(elevation, rng.uniform(0.6, 1.6)) * (1 + 0.15 * fine)
        if rng.random() < 0.7:                                   # реки и дороги — тонкие тёмные линии на снегу
            lines = np.abs(fractal(shape, rng.uniform(2.0, 2.6), rng) - 0.5) < rng.uniform(0.004, 0.012)
            snow_color = snow_color * (1 - 0.45 * ndimage.gaussian_filter(lines.astype(np.float32), 0.6))
        snow_rgb = np.stack([snow_color, snow_color, snow_color * rng.uniform(1.0, 1.05)], -1)
        land = land * (1 - snow[..., None]) + snow_rgb * snow[..., None]

    # --- вода ---
    if kind == "океан":
        sea_level = 2.0
    elif kind in ("побережье", "морской лёд"):
        sea_level = np.quantile(elevation, rng.uniform(0.35, 0.8))
    elif rng.random() < 0.35:
        sea_level = np.quantile(elevation, rng.uniform(0.03, 0.2))      # озёра и реки
    else:
        sea_level = -1.0
    water = elevation < sea_level
    if water.any():
        depth = np.clip((sea_level - elevation) * rng.uniform(4, 12), 0, 1)
        deep = np.array([0.02, 0.06, 0.13]) * rng.uniform(0.7, 1.5)
        shallow = np.array(rng.choice([(0.08, 0.28, 0.30), (0.20, 0.30, 0.26), (0.45, 0.33, 0.20)]))  # бирюза, ил, муть
        sea = shallow * (1 - depth[..., None]) + deep * depth[..., None]
        sea = sea * (1 + 0.25 * fine[..., None])
        if rng.random() < 0.25:                                  # солнечный блик на воде — ещё одна «ловушка»
            yy, xx = np.mgrid[0:h, 0:w]
            cy, cx = rng.uniform(0, h), rng.uniform(0, w)
            radius = rng.uniform(0.25, 0.6) * max(h, w)
            glint = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / radius ** 2)) * rng.uniform(0.15, 0.5)
            sea = sea + glint[..., None] * np.array([0.9, 0.95, 1.0])
        if kind == "морской лёд":
            floes = fractal(shape, rng.uniform(1.6, 2.2), rng)
            ice = np.clip((floes - np.quantile(floes, rng.uniform(0.2, 0.7))) * 40, 0, 1)   # льдины с резкими краями
            ice_color = rng.uniform(0.82, 0.98) * (1 + 0.2 * fine)
            sea = sea * (1 - ice[..., None]) + np.stack([ice_color, ice_color, ice_color * 1.03], -1) * ice[..., None]
        soft = ndimage.gaussian_filter(water.astype(np.float32), 0.8)[..., None]
        land = land * (1 - soft) + sea * soft
    return np.clip(land, 0, 1).astype(np.float32)


CLOUD_STYLES = ["слоистые", "кучевые", "мелкие кучевые", "смесь"]


def make_clouds(shape: tuple[int, int], coverage: float, rng: np.random.Generator,
                style: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Рисует слой облаков. Возвращает (плотность 0..1, яркость 0..1).

    coverage — какую долю кадра облака должны занять (0..1);
    style    — вид облаков из CLOUD_STYLES (если не задан, выбирается случайно).
    """
    density = np.zeros(shape, dtype=np.float32)
    style = style or str(rng.choice(CLOUD_STYLES, p=[0.3, 0.2, 0.25, 0.25]))
    if coverage > 0.003:
        scale = float(rng.choice([0.0, 0.0, 2.0, 4.0]))                  # одно большое поле облаков или несколько
        base = fractal(shape, rng.uniform(2.0, 3.0), rng, detail=scale)
        if style == "мелкие кучевые":
            # Россыпь маленьких плотных облаков («попкорн»). Рисуем так: бросаем на кадр случайные
            # точки и размываем каждую в пятнышко. Соседние пятнышки сливаются в облачка.
            radius = rng.uniform(1.3, 4.0)
            seeds = (rng.random(shape) < rng.uniform(0.01, 0.04)).astype(np.float32)
            puffs = ndimage.gaussian_filter(seeds, radius)
            big = ndimage.gaussian_filter((rng.random(shape) < 0.006).astype(np.float32), radius * 2.0)
            puffs = puffs / (puffs.max() + 1e-6) + 0.7 * big / (big.max() + 1e-6)
            base = base * rng.uniform(0.2, 0.9) + puffs / puffs.max()     # base решает, где облачков больше
        elif style in ("кучевые", "смесь"):
            puffs = fractal(shape, rng.uniform(2.3, 3.0), rng, detail=float(rng.choice([4.0, 8.0, 14.0])))  # «барашки»
            base = base * rng.uniform(0.3, 0.7) + puffs * rng.uniform(0.4, 0.8)
        level = np.quantile(base, 1.0 - min(coverage, 0.9995))
        softness = rng.uniform(0.015, 0.16) if style in ("слоистые", "смесь") else rng.uniform(0.01, 0.05)
        density = np.clip((base - level) / softness + CLOUD_ALPHA_THRESHOLD, 0, 1)
        if style in ("кучевые", "мелкие кучевые"):
            # Кучевое облако плотное: сквозь него земли не видно даже у самого края.
            density = np.clip((density - CLOUD_ALPHA_THRESHOLD) * 3.0 + CLOUD_ALPHA_THRESHOLD, 0, 1)
        density *= rng.uniform(0.75, 1.0)                                # тонкие или плотные
    if rng.random() < 0.25:                                              # перистые: длинные прозрачные полосы
        angle = rng.uniform(0, np.pi)
        streaks = fractal(shape, rng.uniform(2.0, 2.6), rng,
                          stretch=(1 + 5 * abs(np.sin(angle)), 1 + 5 * abs(np.cos(angle))))
        cirrus = np.clip((streaks - rng.uniform(0.45, 0.7)) * 3, 0, 1) * rng.uniform(0.1, 0.3)
        density = np.maximum(density, cirrus.astype(np.float32))
    blur = rng.uniform(0.3, 1.2) if style in ("слоистые", "смесь") else rng.uniform(0.2, 0.6)   # кучевые резче
    density = ndimage.gaussian_filter(density, blur)
    billows = fractal(shape, 1.8, rng)
    brightness = rng.uniform(0.86, 1.12) * (0.88 + 0.12 * billows) * (0.85 + 0.15 * density)  # края облака серее середины
    return density.astype(np.float32), brightness.astype(np.float32)


def apply_sensor(img: np.ndarray, rng: np.random.Generator, max_gain: float = 1.3) -> np.ndarray:
    """Имитация настоящей камеры: выдержка, оттенок, дымка, шум, размытие и сжатие JPEG."""
    img = img * rng.uniform(0.75, max_gain)                                # светлее или темнее
    img = img * rng.uniform(0.95, 1.05, size=3).astype(np.float32)         # лёгкий цветной оттенок
    # Из космоса краски бледнее, чем на земле: между камерой и поверхностью — вся атмосфера.
    fade = rng.uniform(0.0, 0.5)
    img = img * (1 - fade) + img.mean(axis=-1, keepdims=True) * fade       # цвета ближе к серому
    haze = rng.uniform(0.0, 0.18) if rng.random() < 0.7 else 0.0           # атмосферная дымка
    img = img * (1 - haze) + haze * np.array([0.78, 0.86, 1.0], dtype=np.float32)
    img = np.clip(img, 0, 1) ** rng.uniform(0.8, 1.25)                     # гамма: контраст в тенях
    img = ndimage.gaussian_filter(img, (rng.uniform(0, 0.9),) * 2 + (0,))  # чуть не в фокусе
    img = img + rng.normal(0, rng.uniform(0, 0.02), img.shape)             # шум матрицы
    img = np.clip(img, 0, 1)
    if rng.random() < 0.7:                                                 # сжатие JPEG, как в реальных файлах
        buf = io.BytesIO()
        Image.fromarray((img * 255).astype(np.uint8)).save(buf, "JPEG", quality=int(rng.integers(55, 96)))
        img = np.asarray(Image.open(io.BytesIO(buf.getvalue())), dtype=np.float32) / 255.0
    return img.astype(np.float32)


def make_scene(seed: int, size: int = 256, coverage: float | None = None, kind: str | None = None,
               nodata: bool | None = None, cloud_style: str | None = None, snow_share: float = 0.0) -> dict:
    """Один учебный снимок с ответом.

    Возвращает словарь:
        rgb     — снимок (0..1), высота × ширина × 3
        mask    — True, где облако (правильный ответ)
        valid   — True, где есть данные
        percent — истинный процент облачности
        kind    — тип местности
    """
    rng = np.random.default_rng(seed)
    shape = (size, size)
    kind = kind or str(rng.choice(SCENE_KINDS, p=scene_weights(snow_share)))
    if coverage is None:
        roll = rng.random()
        coverage = 0.0 if roll < 0.08 else 1.0 if roll < 0.14 else float(rng.uniform(0, 1))

    ground = make_ground(shape, kind, rng)
    density, brightness = make_clouds(shape, coverage, rng, cloud_style)

    # Тень облака: тот же рисунок, сдвинутый в сторону от солнца.
    shift = (int(rng.integers(-10, 11)), int(rng.integers(-10, 11)))
    shadow = np.roll(density, shift, axis=(0, 1)) * rng.uniform(0.15, 0.5)
    ground = ground * (1 - shadow[..., None])

    tint = np.array([1.0, 1.0, rng.uniform(1.0, 1.04)], dtype=np.float32)
    img = ground * (1 - density[..., None]) + brightness[..., None] * tint * density[..., None]
    # На снежных снимках камера обычно убавляет выдержку, и снег не «выгорает» в ровное белое пятно.
    # Иногда (каждый четвёртый раз) оставляем пересвет: такие снимки сеть честно путает с облаками.
    snowy = kind in SNOW_KINDS
    img = apply_sensor(img, rng, max_gain=1.0 if snowy and rng.random() < 0.75 else 1.3)
    mask = density >= CLOUD_ALPHA_THRESHOLD

    valid = np.ones(shape, dtype=bool)
    if nodata is None:
        nodata = rng.random() < 0.08
    if nodata:                                                   # чёрный угол кадра, как у повёрнутого снимка
        yy, xx = np.mgrid[0:size, 0:size]
        corner = (xx * rng.choice([-1, 1]) + yy * rng.choice([-1, 1]))
        valid = corner < np.quantile(corner, rng.uniform(0.75, 0.92))
        img[~valid] = 0.0

    n_valid = int(valid.sum())
    percent = 100.0 * float((mask & valid).sum()) / n_valid if n_valid else 0.0
    return {"rgb": img, "mask": mask & valid, "valid": valid, "percent": percent, "kind": kind, "seed": seed}


def save_scene(scene: dict, path: str, quality: int = 92) -> None:
    """Сохраняет снимок на диск (JPEG или PNG — по расширению)."""
    image = Image.fromarray((np.clip(scene["rgb"], 0, 1) * 255).astype(np.uint8))
    if str(path).lower().endswith((".jpg", ".jpeg")):
        image.save(path, quality=quality)
    else:
        image.save(path)
