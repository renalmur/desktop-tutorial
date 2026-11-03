"""Создаёт учебные снимки для пробы.  Запуск:  python tools/make_samples.py

Снимки появятся в data/incoming, а правильные ответы — в data/demo_answers.csv.
Сначала попробуйте угадать процент облачности на глаз, потом сравните с сетью и с ответом.
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cloudai.synth import make_scene, save_scene  # noqa: E402

# (тип местности, доля облаков, вид облаков)
DEMO = [
    ("побережье", 0.05, "кучевые"), ("лес и поля", 0.20, "мелкие кучевые"), ("океан", 0.35, "слоистые"),
    ("город и поля", 0.50, "смесь"), ("пустыня", 0.62, "кучевые"), ("побережье", 0.72, "смесь"),
    ("лес и поля", 0.79, "слоистые"), ("океан", 0.88, "смесь"), ("город и поля", 0.96, "слоистые"),
    ("лес и поля", 0.0, "слоистые"), ("пустыня", 0.30, "мелкие кучевые"), ("побережье", 1.0, "слоистые"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Учебные снимки для пробы")
    parser.add_argument("--out", default=str(ROOT / "data" / "incoming"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (kind, coverage, style) in enumerate(DEMO, start=1):
        scene = make_scene(args.seed * 100 + i, size=args.size, coverage=coverage, kind=kind,
                           nodata=(i == 4), cloud_style=style)
        name = f"snimok_{i:02d}.jpg"
        save_scene(scene, str(out / name))
        rows.append([name, kind, f"{scene['percent']:.1f}"])
    real = ROOT / "data" / "real_check" / "sentinel2_scene.png"
    if real.exists():                                   # настоящий снимок Sentinel-2 — для сравнения
        shutil.copy(real, out / "sentinel2_nastoyashchiy.png")
    with (ROOT / "data" / "demo_answers.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["файл", "местность", "истинная облачность, %"])
        writer.writerows(rows)
    print(f"Готово: {len(rows)} учебных снимков в {out}")
    print("Правильные ответы: data/demo_answers.csv")


if __name__ == "__main__":
    main()
