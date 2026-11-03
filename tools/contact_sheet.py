"""Контактный лист: миниатюры снимков с прогнозом модели — чтобы быстро просмотреть набор.

    python tools/contact_sheet.py data/real_extra

Создаёт reports/contact/sheet_01.jpg, sheet_02.jpg, … и рядом sheet_01.json
с соответствием «номер на листе → имя файла → прогноз».
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cloudai.imageio import is_image_file, load_image       # noqa: E402
from cloudai.net import CloudNet                            # noqa: E402

TILE_W, TILE_H = 480, 360
CAPTION = 34
COLS, ROWS = 4, 3


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", default="data/real_extra")
    parser.add_argument("--model", default=None)
    parser.add_argument("--per-sheet", type=int, default=COLS * ROWS)
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = ROOT / folder
    model = CloudNet.load(args.model or ROOT / "models" / "cloud_net.npz")
    files = sorted(p for p in folder.iterdir() if is_image_file(p))
    if not files:
        print(f"Нет снимков в {folder}")
        return 1

    out_dir = ROOT / "reports" / "contact"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("sheet_*"):
        old.unlink()

    f_big, f_small = font(22), font(16)
    index_rows = []
    for start in range(0, len(files), args.per_sheet):
        chunk = files[start:start + args.per_sheet]
        sheet = Image.new("RGB", (COLS * TILE_W, ROWS * (TILE_H + CAPTION)), (24, 26, 32))
        draw = ImageDraw.Draw(sheet)
        rows = []
        for i, path in enumerate(chunk):
            number = start + i + 1
            image = load_image(path)
            percent = model.analyze(image["rgb"], image["alpha"])["cloud_percent"]
            photo = Image.fromarray((image["rgb"] * 255).astype("uint8")).resize((TILE_W, TILE_H), Image.LANCZOS)
            col, row = i % COLS, i // COLS
            x, y = col * TILE_W, row * (TILE_H + CAPTION)
            sheet.paste(photo, (x, y))
            draw.rectangle([x, y + TILE_H, x + TILE_W, y + TILE_H + CAPTION], fill=(24, 26, 32))
            draw.text((x + 8, y + TILE_H + 6), f"{number:02d}  {percent:5.1f}%", fill=(255, 210, 90), font=f_big)
            short = path.stem.split("_SXC3")[0]
            draw.text((x + 130, y + TILE_H + 9), short, fill=(190, 195, 205), font=f_small)
            rows.append({"number": number, "file": path.name, "model_percent": round(percent, 1)})
        sheet_path = out_dir / f"sheet_{(start // args.per_sheet) + 1:02d}.jpg"
        sheet.save(sheet_path, quality=88)
        (sheet_path.with_suffix(".json")).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        index_rows += rows
        print(f"{sheet_path.relative_to(ROOT)}  ({len(chunk)} снимков)")
    (out_dir / "index.json").write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
