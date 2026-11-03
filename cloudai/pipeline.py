"""Обработка папки со снимками — то, что требует кейс.

Для каждого снимка из папки «входящие»:
    1) нейросеть считает процент облачности;
    2) если облачность ВЫШЕ порога (по умолчанию 75 %) — снимок удаляется;
    3) иначе снимок переносится в папку «сохранённые»;
    4) оператор получает сообщение: имя сохранённого файла, процент облачности,
       имя удалённого файла.

Правило безопасности: если файл не удалось прочитать, программа его НЕ трогает.
Удалять можно только то, в чём мы уверены.
"""
import csv
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from cloudai.imageio import ImageReadError, is_image_file, load_image
from cloudai.net import CloudNet

DEFAULT_THRESHOLD = 75.0

SAVED, DELETED, ERROR = "saved", "deleted", "error"


def should_delete(cloud_percent: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Решение по одному снимку. В кейсе сказано «выше 75 %», значит ровно 75 % — ещё оставляем."""
    return cloud_percent > threshold


@dataclass
class ImageResult:
    name: str                       # имя файла во «входящих»
    status: str                     # saved / deleted / error
    cloud_percent: float | None = None
    saved_as: str | None = None     # под каким именем лежит в «сохранённых»
    moved_to_trash: bool = False    # True — не стёрт, а перенесён в корзину
    error: str | None = None
    width: int | None = None
    height: int | None = None
    preview: str | None = None      # уменьшенная копия снимка (для интерфейса)
    overlay: str | None = None      # она же с закрашенными облаками


@dataclass
class Report:
    started_at: str
    threshold: float
    delete_mode: str
    incoming_dir: str
    saved_dir: str
    results: list[ImageResult] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def saved(self) -> list[ImageResult]:
        return [r for r in self.results if r.status == SAVED]

    @property
    def deleted(self) -> list[ImageResult]:
        return [r for r in self.results if r.status == DELETED]

    @property
    def errors(self) -> list[ImageResult]:
        return [r for r in self.results if r.status == ERROR]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["counts"] = {"total": len(self.results), "saved": len(self.saved),
                          "deleted": len(self.deleted), "errors": len(self.errors)}
        data["message"] = format_report(self)
        return data


def _free_name(folder: Path, name: str) -> Path:
    """Если файл с таким именем уже есть, добавляем _1, _2… — чужой снимок не затираем."""
    target = folder / name
    stem, suffix = Path(name).stem, Path(name).suffix
    n = 1
    while target.exists():
        target = folder / f"{stem}_{n}{suffix}"
        n += 1
    return target


def make_overlay(rgb: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> Image.Image:
    """Картинка для человека: облака закрашены малиновым, поля без данных — в полоску."""
    base = (np.clip(rgb, 0, 1) * 255).astype(np.float32)
    color = np.array([255, 45, 150], dtype=np.float32)
    base[mask] = base[mask] * 0.45 + color * 0.55
    if not valid.all():
        yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
        stripes = ((xx + yy) // 6) % 2 == 0
        base[~valid & stripes] = (70, 70, 80)
        base[~valid & ~stripes] = (40, 40, 48)
    return Image.fromarray(base.astype(np.uint8))


def analyze_file(path: Path, model: CloudNet) -> dict:
    """Один снимок: читаем, считаем облачность. Возвращает результат сети и уменьшенный снимок."""
    image = load_image(path)
    result = model.analyze(image["rgb"], image["alpha"])
    result.update(rgb=image["rgb"], width=image["width"], height=image["height"])
    return result


def process_folder(incoming_dir: str | Path, saved_dir: str | Path, model: CloudNet,
                   threshold: float = DEFAULT_THRESHOLD, delete_mode: str = "permanent",
                   trash_dir: str | Path | None = None, previews_dir: str | Path | None = None,
                   progress=None) -> Report:
    """Обрабатывает все снимки из incoming_dir.

    delete_mode:
        "permanent" — облачные снимки стираются с диска (как требует кейс);
        "trash"     — переносятся в папку trash_dir, чтобы можно было проверить и вернуть.
    previews_dir — куда складывать уменьшенные картинки для интерфейса (можно не задавать).
    progress     — функция, которую вызывают после каждого снимка: progress(номер, всего, имя).
    """
    incoming_dir, saved_dir = Path(incoming_dir), Path(saved_dir)
    if delete_mode not in ("permanent", "trash"):
        raise ValueError("delete_mode должен быть 'permanent' или 'trash'")
    if not 0 <= threshold <= 100:
        raise ValueError("порог облачности должен быть от 0 до 100")
    if delete_mode == "trash":
        if trash_dir is None:
            raise ValueError("для режима 'trash' укажите папку корзины trash_dir")
        trash_dir = Path(trash_dir)
        trash_dir.mkdir(parents=True, exist_ok=True)
    if incoming_dir.resolve() == saved_dir.resolve():
        raise ValueError("папки «входящие» и «сохранённые» должны быть разными")
    saved_dir.mkdir(parents=True, exist_ok=True)
    if previews_dir is not None:
        previews_dir = Path(previews_dir)
        previews_dir.mkdir(parents=True, exist_ok=True)

    report = Report(started_at=datetime.now().isoformat(timespec="seconds"), threshold=threshold,
                    delete_mode=delete_mode, incoming_dir=str(incoming_dir), saved_dir=str(saved_dir))
    started = time.time()
    files = sorted(p for p in incoming_dir.iterdir() if is_image_file(p)) if incoming_dir.exists() else []

    for index, path in enumerate(files, start=1):
        item = ImageResult(name=path.name, status=ERROR)
        try:
            analysis = analyze_file(path, model)
        except ImageReadError as exc:
            item.error = f"{exc}; файл оставлен на месте"
            report.results.append(item)
            if progress:
                progress(index, len(files), path.name)
            continue

        item.cloud_percent = round(analysis["cloud_percent"], 1)
        item.width, item.height = analysis["width"], analysis["height"]

        if previews_dir is not None:
            stem = f"{index:04d}"
            Image.fromarray((np.clip(analysis["rgb"], 0, 1) * 255).astype(np.uint8)).save(
                previews_dir / f"{stem}_photo.jpg", quality=85)
            make_overlay(analysis["rgb"], analysis["mask"], analysis["valid"]).save(
                previews_dir / f"{stem}_clouds.jpg", quality=85)
            item.preview, item.overlay = f"{stem}_photo.jpg", f"{stem}_clouds.jpg"

        try:
            # Решение принимаем по тому же округлённому числу, которое видит оператор в отчёте.
            if should_delete(item.cloud_percent, threshold):
                if delete_mode == "trash":
                    shutil.move(str(path), str(_free_name(trash_dir, path.name)))
                    item.moved_to_trash = True
                else:
                    os.remove(path)
                item.status = DELETED
            else:
                target = _free_name(saved_dir, path.name)
                shutil.move(str(path), str(target))
                item.saved_as = target.name
                item.status = SAVED
        except OSError as exc:
            item.status = ERROR
            item.error = f"не удалось переместить или удалить файл ({exc.strerror or exc}); файл оставлен на месте"
        report.results.append(item)
        if progress:
            progress(index, len(files), path.name)

    report.seconds = round(time.time() - started, 2)
    return report


def percent_text(value: float) -> str:
    """12.5 -> «12,5 %» (в русских текстах дробная часть отделяется запятой)."""
    return f"{value:.1f}".replace(".", ",") + " %"


def format_report(report: Report) -> str:
    """Сообщение оператору — обычный текст, который можно прочитать и распечатать."""
    when = report.started_at.replace("T", " ")
    lines = [
        "РЕЗУЛЬТАТ ОБРАБОТКИ СНИМКОВ",
        f"Дата и время: {when}",
        f"Порог отсева: облачность выше {report.threshold:g} %",
        f"Обработано: {len(report.results)}   сохранено: {len(report.saved)}   "
        f"удалено: {len(report.deleted)}   не прочитано: {len(report.errors)}",
        "",
        "СОХРАНЁННЫЕ ФАЙЛЫ",
    ]
    if report.saved:
        for n, r in enumerate(report.saved, start=1):
            renamed = f" (было: {r.name})" if r.saved_as != r.name else ""
            lines.append(f"  {n}. {r.saved_as}{renamed} — облачность {percent_text(r.cloud_percent)}")
    else:
        lines.append("  нет")
    lines += ["", "УДАЛЁННЫЕ ФАЙЛЫ"]
    if report.deleted:
        for n, r in enumerate(report.deleted, start=1):
            where = " (перенесён в корзину)" if r.moved_to_trash else ""
            lines.append(f"  {n}. {r.name} — облачность {percent_text(r.cloud_percent)}{where}")
    else:
        lines.append("  нет")
    if report.errors:
        lines += ["", "НЕ ОБРАБОТАНЫ (файлы не тронуты)"]
        for n, r in enumerate(report.errors, start=1):
            lines.append(f"  {n}. {r.name} — {r.error}")
    return "\n".join(lines)


def save_report(report: Report, reports_dir: str | Path, run_id: str | None = None) -> dict:
    """Сохраняет отчёт в трёх видах: текст для оператора, таблица CSV и JSON для программ."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    txt = reports_dir / f"report_{run_id}.txt"
    csv_path = reports_dir / f"report_{run_id}.csv"
    json_path = reports_dir / f"report_{run_id}.json"
    txt.write_text(format_report(report) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:       # utf-8-sig — чтобы Excel понял русские буквы
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["файл", "результат", "облачность, %", "сохранён как", "примечание"])
        names = {SAVED: "сохранён", DELETED: "удалён", ERROR: "не обработан"}
        for r in report.results:
            percent = "" if r.cloud_percent is None else f"{r.cloud_percent:.1f}".replace(".", ",")
            note = r.error or ("в корзине" if r.moved_to_trash else "")
            writer.writerow([r.name, names[r.status], percent, r.saved_as or "", note])
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"txt": txt, "csv": csv_path, "json": json_path, "run_id": run_id}
