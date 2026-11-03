"""Скачивание открытых снимков СПУТНИКС с публичного диска.

Публичная ссылка: https://disk.sputnix.ru/index.php/s/HbkKc34batcjkSX
Данные разрешено использовать в некоммерческих, образовательных и научных целях
со ссылкой на оригинальный проект СПУТНИКС (см. data/real_extra/README_cloud.md).

Скрипт берёт по несколько снимков на каждый месяц, чтобы в наборе были разные
сезоны, местности и погода, а не серия почти одинаковых кадров с одного витка.

    python tools/fetch_sputnix.py --count 40

Файлы кладутся в data/real_extra. Уже скачанные пропускаются, поэтому скрипт
можно запускать повторно.
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

from PIL import Image

BASE = "https://disk.sputnix.ru/public.php/webdav"
TOKEN = "HbkKc34batcjkSX"          # логин публичной ссылки, пароль пустой
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "real_extra"

NAME = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_(?P<sat>SXC3-[0-9A-Za-z]+)_")


def request(path: str) -> urllib.request.Request:
    path = path.removeprefix("/public.php/webdav")      # в PROPFIND путь приходит целиком
    url = BASE + urllib.parse.quote(path)
    req = urllib.request.Request(url)
    req.add_header("Authorization", "Basic " + __import__("base64").b64encode(f"{TOKEN}:".encode()).decode())
    return req


def list_folder(folder: str) -> list[str]:
    """Возвращает пути к JPG в папке публичного диска (с повторами и запасным списком)."""
    cache = OUT / ("_list_root.txt" if folder == "/" else "_list_extra.txt")
    last_error = None
    for attempt in range(4):
        try:
            req = request(folder)
            req.method = "PROPFIND"
            req.add_header("Depth", "1")
            req.add_header("User-Agent", "CloudControl/1.0 (educational project)")
            with urllib.request.urlopen(req, timeout=60) as response:
                xml = response.read().decode("utf-8", "replace")
            paths = re.findall(r"<d:href>(.*?)</d:href>", xml)
            result = [urllib.parse.unquote(p) for p in paths if p.lower().endswith(".jpg")]
            cache.write_text("\n".join(result), encoding="utf-8")
            return result
        except Exception as exc:
            last_error = exc
            time.sleep(2 + attempt * 3)
    if cache.exists():
        print(f"  диск недоступен ({last_error}), беру сохранённый список {cache.name}")
        return [line.strip() for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise last_error


def pick(images: list[str], per_month: int) -> list[str]:
    """По несколько снимков на месяц, по одному с каждой сцены (дата + спутник)."""
    by_month: dict[str, dict[tuple[str, str], list[str]]] = defaultdict(lambda: defaultdict(list))
    for path in images:
        match = NAME.search(path)
        if not match:
            continue
        scene = (match["date"], match["sat"])
        by_month[match["date"][:7]][scene].append(path)

    chosen = []
    for month in sorted(by_month):
        scenes = sorted(by_month[month].items())
        # равномерно по месяцу, а не первые подряд
        if len(scenes) <= per_month:
            picks = scenes
        else:
            step = (len(scenes) - 1) / (per_month - 1) if per_month > 1 else 0
            picks = [scenes[round(i * step)] for i in range(per_month)]
        for (date, sat), files in picks:
            middle = files[len(files) // 2]          # средний кадр серии
            chosen.append(middle)
    return chosen


def download(path: str) -> Path | None:
    target = OUT / Path(path).name
    if target.exists() and target.stat().st_size > 100_000:
        try:
            with Image.open(target) as img:      # недокачанный файл лучше скачать заново
                img.load()
            print(f"  уже есть: {target.name}")
            return target
        except Exception:
            target.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(request(path), timeout=120) as response, target.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        with Image.open(target) as img:               # проверяем, что файл дошёл целиком
            img.load()
        print(f"  скачан: {target.name} ({target.stat().st_size / 1e6:.1f} МБ)")
        return target
    except Exception as exc:                       # сеть бывает нестабильной
        print(f"  ОШИБКА {Path(path).name}: {exc}")
        target.unlink(missing_ok=True)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=40, help="сколько снимков скачать (по умолчанию 40)")
    parser.add_argument("--per-month", type=int, default=0, help="снимков на месяц; 0 — посчитать из --count")
    parser.add_argument("--random", type=int, default=0, help="взять N случайных сцен вместо равномерных по месяцам")
    parser.add_argument("--seed", type=int, default=7, help="зерно случайной выборки")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print("1/3. Читаю список файлов на публичном диске…")
    images = list_folder("/") + list_folder("/pictures_extra")
    images = sorted(set(images))
    print(f"     найдено снимков: {len(images)}")

    if args.random:
        # Одна сцена = дата + спутник, иначе в выборку попадут почти одинаковые кадры одного витка.
        scenes: dict[tuple[str, str], list[str]] = defaultdict(list)
        for path in images:
            match = NAME.search(path)
            if match:
                scenes[(match["date"], match["sat"])].append(path)
        have = {p.name for p in OUT.glob("*.jpg")}
        keys = [k for k, files in scenes.items() if not any(Path(f).name in have for f in files)]
        rng = __import__("random").Random(args.seed)
        rng.shuffle(keys)
        chosen = [sorted(scenes[k])[len(scenes[k]) // 2] for k in keys[: args.random]]
        print(f"2/3. Случайно выбрано сцен: {len(chosen)}")
    else:
        months = sorted({NAME.search(p)["date"][:7] for p in images if NAME.search(p)})
        per_month = args.per_month or max(1, round(args.count / max(1, len(months))))
        print(f"2/3. Отбираю по {per_month} снимку на месяц ({len(months)} месяцев)…")
        chosen = pick(images, per_month)[: args.count]
        print(f"     выбрано: {len(chosen)}")

    print("3/3. Скачиваю…")
    manifest = []
    for i, path in enumerate(chosen, 1):
        print(f"[{i}/{len(chosen)}] {Path(path).name}")
        saved = download(path)
        if saved is None:
            continue
        match = NAME.search(path)
        manifest.append({
            "file": saved.name,
            "date": match["date"] if match else "",
            "satellite": match["sat"] if match else "",
            "source": "https://disk.sputnix.ru/index.php/s/HbkKc34batcjkSX",
            "license": "Открытые данные СПУТНИКС; некоммерческое/образовательное использование со ссылкой на проект",
            "path": path,
        })
        if path.startswith("/public.php") and "pictures_extra" not in path:
            kml = path[:-4] + ".kml"
            try:
                with urllib.request.urlopen(request(kml), timeout=30) as response:
                    (OUT / (saved.stem + ".kml")).write_bytes(response.read())
            except Exception:
                pass
    manifest_path = OUT / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    merged = {row["file"]: row for row in old}
    merged.update({row["file"]: row for row in manifest})
    manifest_path.write_text(json.dumps(sorted(merged.values(), key=lambda r: r["file"]),
                                        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово: {len(manifest)} новых снимков, всего в папке {len(merged)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
