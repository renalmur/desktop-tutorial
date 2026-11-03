"""Демонстрационный набор для показа кейса.

Что делает:
    1) кладёт в data/incoming три настоящих снимка СПУТНИКС и пять учебных снимков
       генератора — от ясного до почти полностью закрытого облаками;
    2) по команде обрабатывает папку: считает процент облачности, удаляет снимки
       выше 75 % и показывает сообщение оператору.

Запуск:
    python tools/make_demo.py             подготовить и сразу обработать
    python tools/make_demo.py --prepare   только подготовить (например, для скриншота интерфейса)
    python tools/make_demo.py --dry-run   показать проценты, файлы не трогать

Настоящие снимки берутся из data/real_extra (открытые данные СПУТНИКС).
Учебные снимки рисует генератор cloudai/synth.py, поэтому их истинная облачность известна.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cloudai.imageio import is_image_file, load_image              # noqa: E402
from cloudai.net import CloudNet                                  # noqa: E402
from cloudai.pipeline import (DEFAULT_THRESHOLD, format_report,    # noqa: E402
                              process_folder, save_report)
from cloudai.synth import make_scene, save_scene                  # noqa: E402

# Три настоящих снимка: ясный, средней облачности и самый облачный из набора (64 %).
REAL_DEMO = [
    "2025-02-26_16-16-54_SXC3-2323_1.jpg",
    "2025-09-24_12-13-19_SXC3-232_1.jpg",
    "2026-06-03_14-09-03_SXC3-2331_1.jpg",
]

# Учебные снимки: (тип местности, истинная доля облаков, вид облаков).
SYNTH_DEMO = [
    ("океан", 0.05, "кучевые"),
    ("лес и поля", 0.45, "мелкие кучевые"),
    ("побережье", 0.70, "смесь"),
    ("пустыня", 0.82, "кучевые"),
    ("город и поля", 0.96, "слоистые"),
]

INCOMING = ROOT / "data" / "incoming"
SAVED = ROOT / "data" / "saved"
TRASH = ROOT / "data" / "trash"
PREVIEWS = ROOT / "data" / "previews" / "demo"
REPORTS = ROOT / "reports"


def clear_incoming() -> int:
    """Убирает из «входящих» старые снимки, чтобы демонстрация была одинаковой."""
    removed = 0
    for path in INCOMING.iterdir():
        if is_image_file(path):
            path.unlink()
            removed += 1
    return removed


def clear_old_outputs() -> None:
    """Убирает прошлые результаты именно демонстрационного набора (чужие файлы не трогаем)."""
    names = {Path(name).stem for name in REAL_DEMO}
    names.update(f"demo_{i:02d}" for i in range(1, len(SYNTH_DEMO) + 1))
    for folder in (SAVED, TRASH):
        if not folder.exists():
            continue
        for path in folder.iterdir():
            if path.is_file() and any(path.stem == n or path.stem.startswith(n + "_") for n in names):
                path.unlink()
    if PREVIEWS.exists():
        shutil.rmtree(PREVIEWS)


def prepare() -> list[dict]:
    INCOMING.mkdir(parents=True, exist_ok=True)
    removed = clear_incoming()
    if removed:
        print(f"Убрано старых снимков из data/incoming: {removed}")
    clear_old_outputs()

    rows, missing = [], []
    for name in REAL_DEMO:
        source = ROOT / "data" / "real_extra" / name
        if not source.exists():
            missing.append(name)
            continue
        shutil.copy2(source, INCOMING / name)
        rows.append({"file": name, "kind": "настоящий снимок СПУТНИКС", "true": None})

    for i, (kind, coverage, style) in enumerate(SYNTH_DEMO, start=1):
        scene = make_scene(seed=777_000 + i, size=640, coverage=coverage, kind=kind, cloud_style=style)
        name = f"demo_{i:02d}.jpg"
        save_scene(scene, str(INCOMING / name), quality=92)
        rows.append({"file": name, "kind": f"учебный: {kind}", "true": scene["percent"]})

    if missing:
        print("Не найдены настоящие снимки (пропущены): " + ", ".join(missing))
        print("Скачайте их: python tools/fetch_sputnix.py --count 40")
    (INCOMING / "README_DEMO.txt").write_text(
        "ДЕМОНСТРАЦИОННЫЙ НАБОР\n\n"
        "3 настоящих снимка СПУТНИКС (data/real_extra) и 5 учебных снимков генератора.\n"
        "Учебные снимки: от ясного до почти полностью закрытого облаками.\n"
        "Ожидаемый результат при пороге 75 %: снимки demo_04 и demo_05 удаляются,\n"
        "остальные сохраняются. Истинная облачность учебных снимков напечатана\n"
        "командой python tools/make_demo.py.\n\n"
        "Запуск: python tools/make_demo.py   (или кнопка «Обработать папку» в интерфейсе)\n",
        encoding="utf-8")
    return rows


def describe(rows: list[dict]) -> None:
    print("\nДЕМОНСТРАЦИОННЫЙ НАБОР")
    print("  файл                                  что это                        истинная облачность")
    for row in rows:
        true = "—" if row["true"] is None else f"{row['true']:.1f} %"
        print(f"  {row['file']:37s} {row['kind']:30s} {true}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Демонстрационный набор для кейса")
    parser.add_argument("--prepare", action="store_true", help="только подготовить файлы, не обрабатывать")
    parser.add_argument("--dry-run", action="store_true", help="посчитать проценты, ничего не удалять")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows = prepare()
    describe(rows)

    model = CloudNet.load()
    if args.prepare:
        print(f"\nГотово: {len(rows)} снимков лежат в {INCOMING}")
        print("Теперь запустите python app.py и нажмите «Обработать папку».")
        return 0

    if args.dry_run:
        print(f"\nПРОБНЫЙ ЗАПУСК: порог {args.threshold:g} %, файлы не изменяются")
        for path in sorted(p for p in INCOMING.iterdir() if is_image_file(p)):
            percent = model.analyze(load_image(path)["rgb"])["cloud_percent"]
            verdict = "был бы УДАЛЁН" if percent > args.threshold else "был бы сохранён"
            print(f"  {path.name:37s} {percent:6.1f} %   {verdict}")
        return 0

    print(f"\nОбрабатываю папку {INCOMING} (порог {args.threshold:g} %)…")
    report = process_folder(INCOMING, SAVED, model, threshold=args.threshold,
                            delete_mode="permanent", previews_dir=PREVIEWS,
                            progress=lambda done, total, name: print(f"  [{done}/{total}] {name}"))
    paths = save_report(report, REPORTS, run_id="demo")
    print()
    print(format_report(report))
    print(f"\nОтчёт сохранён: {paths['txt']}")
    print(f"Миниатюры для показа: {PREVIEWS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
