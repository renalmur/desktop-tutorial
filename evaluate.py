"""Проверка качества нейросети.  Запуск:  python evaluate.py

Скрипт честно отвечает на вопрос «насколько сети можно верить»:
    1) рисует снимки, которых сеть при обучении не видела, и сравнивает её ответ с правильным;
    2) отдельно проверяет трудные случаи — снег и лёд;
    3) сравнивает сеть с простым правилом «яркий пиксель = облако»;
    4) проверяет устойчивость: тёмный и пересвеченный снимок, сильное сжатие, малый размер;
    5) проверяет настоящий снимок Sentinel-2 (data/real_check);
    6) проверяет ваши размеченные снимки, если есть полные маски в data/masks.

Результат: таблица в консоли и страница reports/quality_report.html с картинками.
"""
import argparse
import io
import json
import sys
import time
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image

from cloudai.imageio import is_image_file, load_image
from cloudai.net import CloudNet
from cloudai.pipeline import DEFAULT_THRESHOLD, make_overlay
from cloudai.synth import SNOW_KINDS, make_scene

ROOT = Path(__file__).resolve().parent
PLAIN_KINDS = ["океан", "побережье", "лес и поля", "пустыня", "город и поля"]


# ------------------------------------------------------------------ наборы снимков

def make_test_scenes(kinds: list[str], per_kind: int, first_seed: int) -> list[dict]:
    """Проверочные снимки: доля облаков равномерно от 0 до 100 %, номера (seed) не пересекаются с обучением."""
    scenes = []
    for k, kind in enumerate(kinds):
        for i in range(per_kind):
            coverage = i / (per_kind - 1)
            scenes.append(make_scene(first_seed + k * 10_000 + i, size=384, coverage=coverage, kind=kind))
    return scenes


def brightness_rule(rgb: np.ndarray, level: float) -> np.ndarray:
    """Простое правило без ИИ: пиксель — облако, если даже самый тёмный его канал ярче level."""
    return rgb.min(axis=-1) > level


def tune_brightness_rule(scenes: list[dict]) -> float:
    """Подбираем лучший уровень для простого правила — чтобы сравнение с сетью было честным."""
    best_level, best_accuracy = 0.5, 0.0
    for level in np.arange(0.30, 0.95, 0.025):
        accuracy = np.mean([(brightness_rule(s["rgb"], level) == s["mask"])[s["valid"]].mean() for s in scenes])
        if accuracy > best_accuracy:
            best_level, best_accuracy = float(level), float(accuracy)
    return best_level


# ------------------------------------------------------------------ метрики

def measure(scenes: list[dict], predict, threshold: float) -> dict:
    """Считает метрики. predict(снимок, valid) -> маска облаков."""
    rows = []
    for scene in scenes:
        valid = scene["valid"]
        mask = predict(scene) & valid
        truth = scene["mask"] & valid
        n = max(int(valid.sum()), 1)
        union = int((mask | truth).sum())
        rows.append({
            "kind": scene["kind"],
            "seed": scene.get("seed"),
            "true": scene["percent"],
            "pred": 100.0 * mask.sum() / n,
            "pixel_accuracy": float((mask == truth)[valid].mean()) if valid.any() else 1.0,
            "iou": float((mask & truth).sum() / union) if union else 1.0,
        })
    true = np.array([r["true"] for r in rows])
    pred = np.array([r["pred"] for r in rows])
    true_delete, pred_delete = true > threshold, pred > threshold
    return {
        "count": len(rows),
        "mae": float(np.abs(pred - true).mean()),                       # средняя ошибка, процентные пункты
        "bias": float((pred - true).mean()),                            # + завышает, − занижает
        "p90_error": float(np.percentile(np.abs(pred - true), 90)),     # 9 снимков из 10 ошибаются не больше
        "within_5": float((np.abs(pred - true) <= 5).mean()),
        "pixel_accuracy": float(np.mean([r["pixel_accuracy"] for r in rows])),
        "iou": float(np.mean([r["iou"] for r in rows])),
        "decision_accuracy": float((true_delete == pred_delete).mean()),
        "wrongly_deleted": int((~true_delete & pred_delete).sum()),     # хороший снимок удалён
        "wrongly_kept": int((true_delete & ~pred_delete).sum()),        # облачный снимок оставлен
        "should_delete": int(true_delete.sum()),
        "should_keep": int((~true_delete).sum()),
        "rows": rows,
    }


def by_kind(metrics: dict) -> dict:
    out = {}
    for kind in dict.fromkeys(r["kind"] for r in metrics["rows"]):
        rows = [r for r in metrics["rows"] if r["kind"] == kind]
        err = np.array([r["pred"] - r["true"] for r in rows])
        out[kind] = {"count": len(rows), "mae": float(np.abs(err).mean()), "bias": float(err.mean()),
                     "pixel_accuracy": float(np.mean([r["pixel_accuracy"] for r in rows]))}
    return out


# ------------------------------------------------------------------ искажения для проверки устойчивости

def distort(scene: dict, how: str) -> dict:
    rgb = scene["rgb"]
    if how == "темнее на 30 %":
        rgb = rgb * 0.7
    elif how == "светлее на 30 %":
        rgb = np.clip(rgb * 1.3, 0, 1)
    elif how == "сильное сжатие JPEG":
        buf = io.BytesIO()
        Image.fromarray((rgb * 255).astype(np.uint8)).save(buf, "JPEG", quality=25)
        rgb = np.asarray(Image.open(io.BytesIO(buf.getvalue())), dtype=np.float32) / 255.0
    elif how == "вдвое меньше размер":
        small = Image.fromarray((rgb * 255).astype(np.uint8)).resize((rgb.shape[1] // 2, rgb.shape[0] // 2), Image.LANCZOS)
        rgb = np.asarray(small, dtype=np.float32) / 255.0
        half = lambda m: np.asarray(Image.fromarray(m).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST))  # noqa: E731
        valid, mask = half(scene["valid"]), half(scene["mask"])
        n = max(int(valid.sum()), 1)
        return dict(scene, rgb=rgb, valid=valid, mask=mask & valid, percent=100.0 * (mask & valid).sum() / n)
    rgb = rgb.astype(np.float32).copy()
    rgb[~scene["valid"]] = 0.0
    return dict(scene, rgb=rgb)


# ------------------------------------------------------------------ картинки и страница отчёта

def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def mask_picture(mask: np.ndarray) -> np.ndarray:
    return np.repeat((mask * 255).astype(np.uint8)[..., None], 3, axis=2)


def save_strip(path: Path, scenes_with_pred: list[tuple[dict, np.ndarray]], tile: int = 200) -> None:
    """Полоса картинок: в каждой колонке сверху снимок, ниже правильная маска, ниже ответ сети."""
    columns = []
    for scene, mask in scenes_with_pred:
        parts = [to_uint8(scene["rgb"]), mask_picture(scene["mask"]),
                 np.asarray(make_overlay(scene["rgb"], mask & scene["valid"], scene["valid"]))]
        parts = [np.asarray(Image.fromarray(p).resize((tile, tile), Image.LANCZOS)) for p in parts]
        columns.append(np.concatenate(parts, axis=0))
        columns.append(np.full((tile * 3, 6, 3), 255, dtype=np.uint8))
    Image.fromarray(np.concatenate(columns[:-1], axis=1)).save(path, quality=88)


def scatter_svg(rows: list[dict], threshold: float, size: int = 420) -> str:
    """График «правильный процент → ответ сети». Точки на диагонали — идеальные ответы."""
    pad = 44
    scale = (size - 2 * pad) / 100.0
    x = lambda v: pad + v * scale                      # noqa: E731
    y = lambda v: size - pad - v * scale               # noqa: E731
    parts = [f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img" '
             f'aria-label="График: правильный процент облачности и ответ сети">',
             f'<rect x="{pad}" y="{pad}" width="{size - 2 * pad}" height="{size - 2 * pad}" fill="#f3f6fa" stroke="#c5d2e1"/>']
    for tick in range(0, 101, 25):
        parts.append(f'<line x1="{x(tick)}" y1="{y(0)}" x2="{x(tick)}" y2="{y(100)}" stroke="#dfe8f2"/>')
        parts.append(f'<line x1="{x(0)}" y1="{y(tick)}" x2="{x(100)}" y2="{y(tick)}" stroke="#dfe8f2"/>')
        parts.append(f'<text x="{x(tick)}" y="{y(0) + 16}" font-size="11" text-anchor="middle" fill="#4d5b74">{tick}</text>')
        parts.append(f'<text x="{x(0) - 6}" y="{y(tick) + 4}" font-size="11" text-anchor="end" fill="#4d5b74">{tick}</text>')
    parts.append(f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(100)}" y2="{y(100)}" stroke="#14213d" stroke-width="1"/>')
    parts.append(f'<line x1="{x(threshold)}" y1="{y(0)}" x2="{x(threshold)}" y2="{y(100)}" stroke="#8a5200" stroke-dasharray="5 4"/>')
    parts.append(f'<line x1="{x(0)}" y1="{y(threshold)}" x2="{x(100)}" y2="{y(threshold)}" stroke="#8a5200" stroke-dasharray="5 4"/>')
    for r in rows:
        wrong = (r["true"] > threshold) != (r["pred"] > threshold)
        color, radius = ("#c2156b", 4.5) if wrong else ("#0b6e69", 2.6)
        parts.append(f'<circle cx="{x(r["true"]):.1f}" cy="{y(r["pred"]):.1f}" r="{radius}" fill="{color}" fill-opacity="0.75"/>')
    parts.append(f'<text x="{size / 2}" y="{size - 6}" font-size="12" text-anchor="middle" fill="#14213d">правильный процент облачности</text>')
    parts.append(f'<text x="12" y="{size / 2}" font-size="12" text-anchor="middle" fill="#14213d" '
                 f'transform="rotate(-90 12 {size / 2})">ответ сети</text>')
    parts.append("</svg>")
    return "".join(parts)


def num(value: float, digits: int = 1) -> str:
    text = f"{value:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")                    # «-0,0» выглядит странно
    return text.replace(".", ",")


def metrics_row(name: str, m: dict) -> str:
    return (f"<tr><th>{escape(name)}</th><td>{m['count']}</td><td>{num(m['mae'])}</td><td>{num(m['bias']):s}</td>"
            f"<td>{num(m['pixel_accuracy'] * 100)} %</td><td>{num(m['decision_accuracy'] * 100)} %</td>"
            f"<td>{m['wrongly_deleted']} из {m['should_keep']}</td><td>{m['wrongly_kept']} из {m['should_delete']}</td></tr>")


def write_html(path: Path, data: dict, scatter: str, pictures: str = "quality") -> None:
    main, snow, base_main, base_snow = data["main"], data["snow"], data["baseline_main"], data["baseline_snow"]
    head = ("<tr><th></th><th>снимков</th><th>средняя ошибка, п.п.</th><th>смещение, п.п.</th><th>верных пикселей</th>"
            "<th>верных решений</th><th>хороших удалено</th><th>облачных оставлено</th></tr>")
    kinds = "".join(f"<tr><th>{escape(k)}</th><td>{v['count']}</td><td>{num(v['mae'])}</td><td>{num(v['bias'])}</td>"
                    f"<td>{num(v['pixel_accuracy'] * 100)} %</td></tr>" for k, v in data["by_kind"].items())
    stress = "".join(f"<tr><th>{escape(k)}</th><td>{num(v['mae'])}</td><td>{num(v['bias'])}</td>"
                     f"<td>{num(v['decision_accuracy'] * 100)} %</td></tr>" for k, v in data["stress"].items())
    real = data.get("real_check")
    real_html = ""
    if real:
        tiles = "".join(f"<tr><th>{escape(t['name'])}</th><td>{num(t['reference'])} %</td><td>{num(t['network'])} %</td></tr>"
                        for t in real["tiles"])
        real_html = f"""
<h2>Настоящий снимок Sentinel-2</h2>
<p>Один настоящий снимок спутника Sentinel-2 (устье реки, суша и море). Эталонная маска получена другим алгоритмом —
s2cloudless, который видит 10 спектральных каналов, а не три цвета, и специально расширяет облака с запасом.
Поэтому это сравнение двух алгоритмов, а не сети с истиной.</p>
<p><b>Эталон: {num(real['reference'])} %, наша сеть: {num(real['network'])} %, совпадение по пикселям: {num(real['pixel_agreement'] * 100)} %.</b>
Плотные и кучевые облака сеть находит, полупрозрачную дымку — нет: на трёх цветовых каналах она почти не отличается от земли.</p>
<img src="{pictures}/real_check.jpg" alt="Слева снимок, в середине эталонная маска, справа ответ сети">
<p class="cap">Слева снимок, в середине эталонная маска, справа ответ сети (облака закрашены).</p>
<table><tr><th>часть снимка</th><th>эталон</th><th>сеть</th></tr>{tiles}</table>"""
    user = data.get("user_masks")
    user_html = ""
    if user:
        user_html = f"""
<h2>Ваши размеченные снимки</h2>
<p>Снимки из <code>data/real</code>, для которых в <code>data/masks</code> лежит полная маска.</p>
<table>{head}{metrics_row("ваши снимки", user)}</table>"""
    page = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Проверка качества нейросети</title>
<style>
body {{ font-family: "Segoe UI", system-ui, Roboto, Arial, sans-serif; color: #14213d; background: #e9f0f7; margin: 0; line-height: 1.5; }}
main {{ max-width: 980px; margin: 0 auto; padding: 24px 20px 60px; }}
h1 {{ font-size: 1.7rem; margin: 0 0 4px; }} h2 {{ font-size: 1.2rem; margin: 34px 0 8px; }}
p {{ margin: 8px 0; max-width: 78ch; }} .cap, .muted {{ color: #4d5b74; font-size: 0.92rem; }}
table {{ border-collapse: collapse; background: #fff; margin: 10px 0; font-variant-numeric: tabular-nums; }}
th, td {{ border: 1px solid #c5d2e1; padding: 6px 10px; text-align: right; font-size: 0.93rem; }}
th:first-child {{ text-align: left; }} tr:first-child th {{ background: #f3f6fa; font-weight: 600; }}
img {{ max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #c5d2e1; }}
.big {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 16px 0; }}
.big div {{ background: #fff; border: 1px solid #c5d2e1; border-radius: 10px; padding: 12px 14px; }}
.big b {{ display: block; font-size: 1.8rem; line-height: 1.15; }} .big span {{ color: #4d5b74; font-size: 0.9rem; }}
.two {{ display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start; }} .scroll {{ overflow-x: auto; }}
code {{ background: #dfe8f2; padding: 1px 5px; border-radius: 4px; }}
.bad {{ color: #c2156b; font-weight: 600; }}
</style></head><body><main>
<h1>Проверка качества нейросети</h1>
<p class="muted">Отчёт создан {escape(data['created_at'].replace('T', ' '))}. Порог отсева: {num(data['threshold'], 0)} %.
Модель обучена {escape(str(data['model'].get('trained_at', '—')).replace('T', ' '))},
настоящих размеченных пикселей в обучении: {data['model'].get('real_pixels', 0)}.</p>

<div class="big">
<div><b>{num(main['mae'])} п.п.</b><span>средняя ошибка процента облачности</span></div>
<div><b>{num(main['decision_accuracy'] * 100)} %</b><span>решений «сохранить/удалить» совпали с правильными</span></div>
<div><b>{main['wrongly_deleted']} из {main['should_keep']}</b><span>хороших снимков удалено по ошибке</span></div>
<div><b>{main['wrongly_kept']} из {main['should_delete']}</b><span>облачных снимков оставлено по ошибке</span></div>
</div>
<p>Эти числа получены на {main['count']} учебных снимках, которых сеть не видела при обучении:
океан, побережье, лес и поля, пустыня, город. <b>Это нарисованные программой снимки, а не настоящие.</b>
На снимках Спутникса числа будут другими — проверьте сеть на них (см. конец отчёта).</p>

<h2>Сеть и простое правило</h2>
<p>Простое правило без ИИ: «пиксель — облако, если он ярче уровня {num(data['baseline_level'], 2)}» (уровень подобран так, чтобы правило работало как можно лучше).</p>
<div class="scroll"><table>{head}
{metrics_row("нейросеть, обычные снимки", main)}
{metrics_row("простое правило, обычные снимки", base_main)}
{metrics_row("нейросеть, снег и лёд", snow)}
{metrics_row("простое правило, снег и лёд", base_snow)}
</table></div>
<p class="muted">п.п. — процентные пункты. Смещение со знаком «+» значит, что облачность завышается.</p>

<h2>Где сеть ошибается</h2>
<div class="two">
<div>{scatter}<p class="cap">Каждая точка — один снимок. На диагонали — идеальные ответы.<br>
Пунктир — порог. Крупные малиновые точки — неверное решение.</p></div>
<div><table><tr><th>местность</th><th>снимков</th><th>средняя ошибка</th><th>смещение</th><th>верных пикселей</th></tr>{kinds}</table>
<p class="bad">Снег и лёд сеть принимает за облака.</p>
<p>Белый снег и белое облако на обычной цветной фотографии почти неразличимы. Настоящие спутники
различают их в инфракрасных каналах, которых в обычном снимке нет. Что с этим делать — в GUIDE_RU.md, раздел «Чего сеть не умеет».</p></div>
</div>

<h2>Примеры</h2>
<p>В каждой колонке сверху вниз: снимок, правильная маска облаков, ответ сети (облака закрашены).</p>
<p><b>Самые большие ошибки на обычных снимках</b></p>
<div class="scroll"><img src="{pictures}/worst.jpg" alt="Шесть снимков с самой большой ошибкой"></div>
<p><b>Случайные обычные снимки</b></p>
<div class="scroll"><img src="{pictures}/typical.jpg" alt="Шесть случайных снимков"></div>
<p><b>Снег и лёд</b></p>
<div class="scroll"><img src="{pictures}/snow.jpg" alt="Шесть снимков со снегом и льдом"></div>

<h2>Устойчивость</h2>
<p>Те же снимки, но испорченные: проверяем, не «ломается» ли сеть от плохих условий съёмки.</p>
<table><tr><th>что сделали со снимком</th><th>средняя ошибка, п.п.</th><th>смещение, п.п.</th><th>верных решений</th></tr>
<tr><th>ничего</th><td>{num(main['mae'])}</td><td>{num(main['bias'])}</td><td>{num(main['decision_accuracy'] * 100)} %</td></tr>{stress}</table>
{real_html}
{user_html}
<h2>Скорость</h2>
<p>Один снимок обрабатывается в среднем за {num(data['seconds_per_image'], 2)} с (без чтения файла с диска).</p>

<h2>Как проверить сеть на снимках Спутникса</h2>
<p>1. Положите снимки в <code>data/incoming</code> и запустите <code>python process.py --dry-run</code> — файлы не изменятся.<br>
2. Сравните проценты со своей оценкой на глаз. Найдите снимки, где сеть ошиблась сильнее всего.<br>
3. Разметьте их во вкладке «Разметка и дообучение» и дообучите сеть.<br>
4. Если есть полные маски (белое — облако), положите их в <code>data/masks</code> и запустите этот отчёт ещё раз.</p>
</main></body></html>"""
    path.write_text(page, encoding="utf-8")


# ------------------------------------------------------------------ основная программа

def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка качества нейросети")
    parser.add_argument("--per-kind", type=int, default=60, help="сколько снимков каждого типа местности (по умолчанию 60)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--model", default=None, help="файл модели (по умолчанию models/cloud_net.npz)")
    parser.add_argument("--tag", default="", help="метка отчёта: с --tag opyt отчёт запишется в quality_report_opyt.html")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        net = CloudNet.load(args.model) if args.model else CloudNet.load()
    except FileNotFoundError as exc:
        print(exc)
        return 1
    suffix = ("_" + "".join(c for c in args.tag if c.isalnum() or c in "-_")) if args.tag else ""
    out = ROOT / "reports" / f"quality{suffix}"
    report_path = ROOT / "reports" / f"quality_report{suffix}.html"
    out.mkdir(parents=True, exist_ok=True)
    threshold = args.threshold

    print("1/6. Рисую проверочные снимки…")
    main_scenes = make_test_scenes(PLAIN_KINDS, args.per_kind, first_seed=5_000_000)
    snow_scenes = make_test_scenes(list(SNOW_KINDS), max(args.per_kind // 2, 6), first_seed=5_500_000)

    print("2/6. Нейросеть отвечает…")
    predictions = {}
    started = time.time()
    for scene in main_scenes + snow_scenes:
        predictions[scene["seed"]] = net.analyze(scene["rgb"])["mask"]
    seconds_per_image = (time.time() - started) / len(predictions)
    net_predict = lambda s: predictions[s["seed"]]              # noqa: E731
    main = measure(main_scenes, net_predict, threshold)
    snow = measure(snow_scenes, net_predict, threshold)

    print("3/6. Простое правило для сравнения…")
    level = tune_brightness_rule(make_test_scenes(PLAIN_KINDS, 8, first_seed=6_000_000))
    rule_predict = lambda s: brightness_rule(s["rgb"], level)   # noqa: E731
    baseline_main = measure(main_scenes, rule_predict, threshold)
    baseline_snow = measure(snow_scenes, rule_predict, threshold)

    print("4/6. Проверка устойчивости…")
    stress = {}
    subset = main_scenes[::3]
    for how in ["темнее на 30 %", "светлее на 30 %", "сильное сжатие JPEG", "вдвое меньше размер"]:
        distorted = [distort(s, how) for s in subset]
        m = measure(distorted, lambda s: net.analyze(s["rgb"])["mask"], threshold)
        stress[how] = {k: m[k] for k in ("mae", "bias", "decision_accuracy")}

    print("5/6. Настоящие снимки…")
    real_check = None
    real_image = ROOT / "data" / "real_check" / "sentinel2_scene.png"
    real_mask = ROOT / "data" / "real_check" / "sentinel2_scene_reference_mask.png"
    if real_image.exists() and real_mask.exists():
        image = load_image(real_image)
        h, w = image["rgb"].shape[:2]
        reference = np.asarray(Image.open(real_mask).convert("L").resize((w, h), Image.NEAREST)) > 127
        result = net.analyze(image["rgb"])
        tiles = []
        for row in range(3):
            for col in range(2):
                ys, xs = slice(row * h // 3, (row + 1) * h // 3), slice(col * w // 2, (col + 1) * w // 2)
                tiles.append({"name": f"ряд {row + 1}, {'левая' if col == 0 else 'правая'} половина",
                              "reference": 100.0 * reference[ys, xs].mean(), "network": 100.0 * result["mask"][ys, xs].mean()})
        real_check = {"reference": 100.0 * reference.mean(), "network": result["cloud_percent"],
                      "pixel_agreement": float((result["mask"] == reference).mean()), "tiles": tiles}
        gap = np.full((h, 6, 3), 255, dtype=np.uint8)
        figure = np.concatenate([to_uint8(image["rgb"]), gap, mask_picture(reference), gap,
                                 np.asarray(make_overlay(image["rgb"], result["mask"], result["valid"]))], axis=1)
        Image.fromarray(figure).resize((figure.shape[1] * 2, figure.shape[0] * 2), Image.LANCZOS).save(out / "real_check.jpg", quality=88)

    user_masks = None
    real_dir, masks_dir = ROOT / "data" / "real", ROOT / "data" / "masks"
    if real_dir.exists() and masks_dir.exists():
        user_scenes = []
        for path in sorted(p for p in real_dir.iterdir() if is_image_file(p)):
            mask_file = masks_dir / (path.stem + ".png")
            if mask_file.exists():
                image = load_image(path)
                h, w = image["rgb"].shape[:2]
                truth = np.asarray(Image.open(mask_file).convert("L").resize((w, h), Image.NEAREST)) > 127
                analysis = net.analyze(image["rgb"], image["alpha"])
                valid = analysis["valid"]
                percent = 100.0 * (truth & valid).sum() / max(int(valid.sum()), 1)
                user_scenes.append({"rgb": image["rgb"], "mask": truth & valid, "valid": valid, "percent": percent,
                                    "kind": "ваши снимки", "seed": path.name, "_pred": analysis["mask"]})
        if user_scenes:
            user_masks = measure(user_scenes, lambda s: s["_pred"], threshold)
            user_masks.pop("rows")

    print("6/6. Собираю отчёт…")
    rng = np.random.default_rng(0)
    order = np.argsort([-abs(r["pred"] - r["true"]) for r in main["rows"]])
    save_strip(out / "worst.jpg", [(main_scenes[i], predictions[main_scenes[i]["seed"]]) for i in order[:6]])
    pick = rng.choice(len(main_scenes), 6, replace=False)
    save_strip(out / "typical.jpg", [(main_scenes[i], predictions[main_scenes[i]["seed"]]) for i in pick])
    pick = np.linspace(0, len(snow_scenes) - 1, 6).astype(int)
    save_strip(out / "snow.jpg", [(snow_scenes[i], predictions[snow_scenes[i]["seed"]]) for i in pick])

    kinds = by_kind(main)
    kinds.update(by_kind(snow))
    scatter = scatter_svg(main["rows"] + snow["rows"], threshold)
    strip = lambda m: {k: v for k, v in m.items() if k != "rows"}    # noqa: E731
    data = {
        "created_at": datetime.now().isoformat(timespec="seconds"), "threshold": threshold, "model": net.info,
        "main": strip(main), "snow": strip(snow), "baseline_level": level,
        "baseline_main": strip(baseline_main), "baseline_snow": strip(baseline_snow),
        "by_kind": kinds, "stress": stress, "real_check": real_check, "user_masks": user_masks,
        "seconds_per_image": seconds_per_image,
    }
    (out / "metrics.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(report_path, data, scatter, out.name)

    print()
    print(f"ОБЫЧНЫЕ СНИМКИ ({main['count']} шт., сеть их не видела)")
    print(f"  средняя ошибка процента облачности: {main['mae']:.1f} п.п. (простое правило: {baseline_main['mae']:.1f})")
    print(f"  верных решений «сохранить/удалить»: {main['decision_accuracy'] * 100:.1f} % (простое правило: {baseline_main['decision_accuracy'] * 100:.1f} %)")
    print(f"  хороших снимков удалено по ошибке:  {main['wrongly_deleted']} из {main['should_keep']}")
    print(f"  облачных снимков оставлено:         {main['wrongly_kept']} из {main['should_delete']}")
    print(f"СНЕГ И ЛЁД ({snow['count']} шт.) — известное слабое место")
    print(f"  средняя ошибка: {snow['mae']:.1f} п.п., смещение {snow['bias']:+.1f}; хороших удалено: {snow['wrongly_deleted']} из {snow['should_keep']}")
    if real_check:
        print(f"НАСТОЯЩИЙ СНИМОК Sentinel-2: эталон {real_check['reference']:.1f} %, сеть {real_check['network']:.1f} %")
    if user_masks:
        print(f"ВАШИ СНИМКИ ({user_masks['count']} шт.): средняя ошибка {user_masks['mae']:.1f} п.п.")
    print(f"\nПодробный отчёт с картинками: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
