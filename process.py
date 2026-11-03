"""Обработка снимков из командной строки.

Примеры:
    python process.py                          обработать data/incoming, облачные удалить
    python process.py --trash                  облачные не стирать, а сложить в data/trash
    python process.py --threshold 60           отсеивать при облачности выше 60 %
    python process.py --incoming D:\\snimki     взять снимки из другой папки
    python process.py --dry-run                только посчитать, ничего не перемещать и не удалять
"""
import argparse
import sys
from pathlib import Path

from cloudai.imageio import ImageReadError, is_image_file
from cloudai.net import CloudNet
from cloudai.pipeline import (DEFAULT_THRESHOLD, analyze_file, format_report, percent_text, process_folder,
                              save_report, should_delete)

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Отсев облачных спутниковых снимков")
    parser.add_argument("--incoming", default=str(ROOT / "data" / "incoming"), help="папка с новыми снимками")
    parser.add_argument("--saved", default=str(ROOT / "data" / "saved"), help="куда переносить хорошие снимки")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="порог облачности, %% (по умолчанию 75)")
    parser.add_argument("--trash", action="store_true", help="не стирать облачные снимки, а переносить в data/trash")
    parser.add_argument("--dry-run", action="store_true", help="только показать проценты, файлы не трогать")
    parser.add_argument("--model", default=None, help="файл модели (по умолчанию models/cloud_net.npz)")
    args = parser.parse_args()

    # В консоли Windows русские буквы иногда превращаются в «кракозябры» — просим UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        model = CloudNet.load(args.model) if args.model else CloudNet.load()
    except FileNotFoundError as exc:
        print(exc)
        return 1

    incoming = Path(args.incoming)
    if not incoming.is_dir():
        print(f"Нет папки {incoming}. Создайте её и положите туда снимки.")
        return 1

    if args.dry_run:
        files = sorted(p for p in incoming.iterdir() if is_image_file(p))
        print(f"Пробный запуск: файлы не изменяются. Порог {args.threshold:g} %\n")
        for path in files:
            try:
                percent = round(analyze_file(path, model)["cloud_percent"], 1)
                verdict = "был бы УДАЛЁН" if should_delete(percent, args.threshold) else "был бы сохранён"
                print(f"  {path.name:40s} облачность {percent_text(percent):>7s}   {verdict}")
            except ImageReadError as exc:
                print(f"  {path.name:40s} не прочитан: {exc}")
        if not files:
            print("  в папке нет снимков")
        return 0

    def progress(done: int, total: int, name: str) -> None:
        print(f"  [{done}/{total}] {name}", flush=True)

    print(f"Обрабатываю папку {incoming} …")
    report = process_folder(incoming, args.saved, model, threshold=args.threshold,
                            delete_mode="trash" if args.trash else "permanent",
                            trash_dir=ROOT / "data" / "trash", progress=progress)
    paths = save_report(report, ROOT / "reports")
    print()
    print(format_report(report))
    print(f"\nОтчёт сохранён: {paths['txt']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
