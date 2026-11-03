"""Веб-интерфейс оператора.

Запуск:   python app.py        (или run_windows.bat / run_linux_mac.sh)
Страница: http://127.0.0.1:8000   (другой порт: переменная окружения CLOUD_PORT)

Сервер работает только на этом компьютере (адрес 127.0.0.1): снимки никуда не отправляются.
"""
import base64
import io
import os
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from cloudai import __version__
from cloudai.imageio import SUPPORTED_EXTENSIONS, ImageReadError, is_image_file, load_image
from cloudai.net import CloudNet
from cloudai.pipeline import DEFAULT_THRESHOLD, make_overlay, process_folder, save_report
from cloudai.training import LABEL_CLEAR, LABEL_CLOUD, label_path_for, train_and_save

ROOT = Path(__file__).resolve().parent


class Folders:
    """Все рабочие папки в одном месте (в тестах подменяются на временные)."""

    def __init__(self, data: Path, reports: Path, models: Path):
        self.incoming = data / "incoming"
        self.saved = data / "saved"
        self.trash = data / "trash"
        self.previews = data / "previews"
        self.real = data / "real"
        self.labels = data / "labels"
        self.masks = data / "masks"
        self.reports = reports
        self.model = models / "cloud_net.npz"
        self.base_model = models / "cloud_net_base.npz"
        for folder in (self.incoming, self.saved, self.trash, self.previews, self.real, self.labels, reports):
            folder.mkdir(parents=True, exist_ok=True)


folders = Folders(ROOT / "data", ROOT / "reports", ROOT / "models")
state = {"model": None}
train_lock = threading.Lock()
process_lock = threading.Lock()

app = FastAPI(title="Облачный контроль", version=__version__)


def get_model() -> CloudNet:
    if state["model"] is None:
        try:
            state["model"] = CloudNet.load(folders.model)
        except FileNotFoundError as exc:
            raise HTTPException(503, str(exc)) from exc
    return state["model"]


def safe_name(name: str) -> str:
    """Оставляет от присланного имени только само имя файла — без папок и опасных символов."""
    name = Path(name.replace("\\", "/")).name
    name = re.sub(r"[^\w.\- ()\[\]]", "_", name, flags=re.UNICODE).strip(" .")
    if not name or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"«{name or 'файл'}» — не снимок. Подходят: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    return name


def free_path(folder: Path, name: str) -> Path:
    target, n = folder / name, 1
    while target.exists():
        target = folder / f"{Path(name).stem}_{n}{Path(name).suffix}"
        n += 1
    return target


def save_uploads(files: list[UploadFile], folder: Path) -> list[str]:
    saved = []
    for upload in files:
        target = free_path(folder, safe_name(upload.filename or ""))
        with target.open("wb") as out:
            shutil.copyfileobj(upload.file, out)
        saved.append(target.name)
    return saved


def cleanup_previews(keep: int = 10) -> None:
    """Миниатюры прошлых запусков не копим: оставляем десять последних."""
    runs = sorted((p for p in folders.previews.iterdir() if p.is_dir()), key=lambda p: p.name)
    for old in runs[:-keep]:
        shutil.rmtree(old, ignore_errors=True)


def run_processing(threshold: float, delete_mode: str) -> dict:
    if not 0 <= threshold <= 100:
        raise HTTPException(400, "Порог облачности — число от 0 до 100")
    if delete_mode not in ("permanent", "trash"):
        raise HTTPException(400, "Режим удаления: permanent или trash")
    model = get_model()
    with process_lock:                                   # два запуска одновременно перепутали бы файлы
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        report = process_folder(folders.incoming, folders.saved, model, threshold=threshold,
                                delete_mode=delete_mode, trash_dir=folders.trash,
                                previews_dir=folders.previews / run_id)
        save_report(report, folders.reports, run_id)
        cleanup_previews()
    data = report.to_dict()
    data["run_id"] = run_id
    return data


# ------------------------------------------------------------------ страница и состояние

@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/status")
def status() -> dict:
    info = {}
    try:
        info = get_model().info
    except HTTPException:
        pass
    waiting = sorted(p.name for p in folders.incoming.iterdir() if is_image_file(p))
    return {
        "version": __version__,
        "model_ready": bool(info) or state["model"] is not None,
        "model": info,
        "default_threshold": DEFAULT_THRESHOLD,
        "incoming_waiting": waiting,
        "folders": {"incoming": str(folders.incoming), "saved": str(folders.saved), "trash": str(folders.trash),
                    "real": str(folders.real)},
        "formats": sorted(SUPPORTED_EXTENSIONS),
    }


# ------------------------------------------------------------------ обработка снимков

@app.post("/api/process")
def process_uploaded(files: list[UploadFile] = File(...), threshold: float = Form(DEFAULT_THRESHOLD),
                     delete_mode: str = Form("permanent")) -> dict:
    """Оператор выбрал файлы на странице: копируем их во «входящие» и обрабатываем."""
    get_model()                                           # если модели нет — скажем об этом до копирования файлов
    for upload in files:
        safe_name(upload.filename or "")                  # сначала проверяем все имена, потом сохраняем
    save_uploads(files, folders.incoming)
    return run_processing(threshold, delete_mode)


class FolderRequest(BaseModel):
    threshold: float = DEFAULT_THRESHOLD
    delete_mode: str = "permanent"


@app.post("/api/process-folder")
def process_incoming_folder(body: FolderRequest) -> dict:
    """Оператор сам положил снимки в папку data/incoming."""
    return run_processing(body.threshold, body.delete_mode)


@app.get("/api/report/{run_id}")
def download_report(run_id: str, format: str = "txt") -> FileResponse:
    if not re.fullmatch(r"[0-9_]{8,32}", run_id) or format not in ("txt", "csv", "json"):
        raise HTTPException(400, "Неверный номер отчёта")
    path = folders.reports / f"report_{run_id}.{format}"
    if not path.exists():
        raise HTTPException(404, "Отчёт не найден")
    return FileResponse(path, filename=path.name)


@app.get("/previews/{run_id}/{name}")
def preview(run_id: str, name: str) -> FileResponse:
    if not re.fullmatch(r"[0-9_]{8,32}", run_id) or not re.fullmatch(r"\d{4}_(photo|clouds)\.jpg", name):
        raise HTTPException(400, "Неверное имя миниатюры")
    path = folders.previews / run_id / name
    if not path.exists():
        raise HTTPException(404, "Миниатюра не найдена")
    return FileResponse(path)


# ------------------------------------------------------------------ разметка и дообучение

def real_image_path(name: str) -> Path:
    path = folders.real / safe_name(name)
    if not path.exists():
        raise HTTPException(404, f"Нет снимка {name} в папке data/real")
    return path


def png_response(image: Image.Image) -> Response:
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


def load_real(name: str) -> dict:
    try:
        return load_image(real_image_path(name))
    except ImageReadError as exc:
        raise HTTPException(422, f"Снимок не читается: {exc}") from exc


@app.get("/api/label/images")
def label_images() -> dict:
    items = []
    for path in sorted(p for p in folders.real.iterdir() if is_image_file(p)):
        items.append({"name": path.name, "labeled": label_path_for(path, folders.labels).exists()})
    return {"images": items}


@app.post("/api/label/upload")
def label_upload(files: list[UploadFile] = File(...)) -> dict:
    for upload in files:
        safe_name(upload.filename or "")
    return {"saved": save_uploads(files, folders.real)}


@app.get("/api/label/image/{name}")
def label_image(name: str) -> Response:
    """Снимок в том размере, в котором его видит сеть: один пиксель холста = один пиксель разметки."""
    image = load_real(name)
    return png_response(Image.fromarray((np.clip(image["rgb"], 0, 1) * 255).astype(np.uint8)))


@app.get("/api/label/prediction/{name}")
def label_prediction(name: str) -> dict:
    """Что сеть думает о снимке сейчас: процент и прозрачная картинка с закрашенными облаками."""
    image = load_real(name)
    result = get_model().analyze(image["rgb"], image["alpha"])
    rgba = np.zeros(result["mask"].shape + (4,), dtype=np.uint8)
    rgba[result["mask"]] = (255, 45, 150, 150)
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, "PNG")
    return {"cloud_percent": round(result["cloud_percent"], 1),
            "overlay": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}


@app.get("/api/label/mask/{name}")
def label_mask(name: str) -> Response:
    """Сохранённые мазки разметки: красный — облако, синий — не облако."""
    path = label_path_for(real_image_path(name), folders.labels)
    if not path.exists():
        raise HTTPException(404, "Разметки ещё нет")
    labels = np.asarray(Image.open(path).convert("L"))
    rgba = np.zeros(labels.shape + (4,), dtype=np.uint8)
    rgba[labels == LABEL_CLOUD] = (255, 0, 0, 255)
    rgba[labels == LABEL_CLEAR] = (0, 0, 255, 255)
    return png_response(Image.fromarray(rgba))


class LabelSave(BaseModel):
    name: str
    image: str = Field(description="PNG с холста: красные мазки — облако, синие — не облако")


@app.post("/api/label/save")
def label_save(body: LabelSave) -> dict:
    image = load_real(body.name)
    height, width = image["rgb"].shape[:2]
    try:
        raw = base64.b64decode(body.image.split(",", 1)[-1], validate=True)
        drawn = np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA").resize((width, height), Image.NEAREST))
    except Exception as exc:
        raise HTTPException(400, "Не удалось прочитать рисунок разметки") from exc
    opaque = drawn[..., 3] > 127
    labels = np.zeros((height, width), dtype=np.uint8)
    labels[opaque & (drawn[..., 2] > 127) & (drawn[..., 0] < 128)] = LABEL_CLEAR
    labels[opaque & (drawn[..., 0] > 127) & (drawn[..., 2] < 128)] = LABEL_CLOUD
    path = label_path_for(real_image_path(body.name), folders.labels)
    cloud, clear = int((labels == LABEL_CLOUD).sum()), int((labels == LABEL_CLEAR).sum())
    if cloud + clear == 0:
        path.unlink(missing_ok=True)                      # всё стёрли ластиком — разметки больше нет
    else:
        Image.fromarray(labels).save(path)
    return {"cloud_pixels": cloud, "clear_pixels": clear}


@app.post("/api/train")
def train() -> dict:
    """Дообучение: учебные снимки + всё, что размечено в data/labels и data/masks."""
    if not train_lock.acquire(blocking=False):
        raise HTTPException(409, "Обучение уже идёт. Дождитесь его окончания.")
    try:
        if folders.model.exists() and not folders.base_model.exists():
            shutil.copy(folders.model, folders.base_model)          # базовую модель бережём
            if folders.model.with_suffix(".json").exists():
                shutil.copy(folders.model.with_suffix(".json"), folders.base_model.with_suffix(".json"))
        log: list[str] = []
        info = train_and_save(model_path=folders.model, real_dir=folders.real, labels_dir=folders.labels,
                              masks_dir=folders.masks, log=log.append)
        state["model"] = CloudNet.load(folders.model)
        return {"info": info, "log": log}
    finally:
        train_lock.release()


@app.post("/api/model/reset")
def model_reset() -> dict:
    """Вернуть базовую модель, если дообучение сделало хуже."""
    if not folders.base_model.exists():
        raise HTTPException(404, "Базовая модель не сохранена: дообучение ещё не запускали")
    with train_lock:
        shutil.copy(folders.base_model, folders.model)
        if folders.base_model.with_suffix(".json").exists():
            shutil.copy(folders.base_model.with_suffix(".json"), folders.model.with_suffix(".json"))
        state["model"] = CloudNet.load(folders.model)
    return {"info": state["model"].info}


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


if __name__ == "__main__":
    import webbrowser

    import uvicorn

    # Порт по умолчанию — 8000. Если он занят, запустите так:  set CLOUD_PORT=8010  (Windows)
    #                                                        или  CLOUD_PORT=8010 python app.py  (macOS/Linux)
    port = int(os.environ.get("CLOUD_PORT", "8000"))
    address = f"http://127.0.0.1:{port}"
    print(f"Интерфейс оператора: {address}   (остановить — Ctrl+C)")
    threading.Timer(1.2, lambda: webbrowser.open(address)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
