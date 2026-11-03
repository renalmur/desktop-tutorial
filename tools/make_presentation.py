"""Сборка презентации кейса «Спутникс» в формате .pptx.

Запуск:
    python tools/make_presentation.py
    python tools/make_presentation.py --out "Моя_презентация.pptx"

Скрипт сам берёт цифры из отчётов:
    reports/quality/metrics.json        — качество на учебных снимках
    data/real_eval/expert_estimates.json — качество на настоящих снимках СПУТНИКС
    models/cloud_net.json               — как обучалась модель
    reports/report_demo.txt             — сообщение оператору из демонстрации
    reports/presentation/*.png          — скриншоты и картинки (если уже сделаны)

Если какого-то файла нет, подставляются значения по умолчанию — презентация всё равно соберётся.
"""
import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "reports" / "presentation"
QUALITY = ROOT / "reports" / "quality"

# ------------------------------------------------------------------ оформление
# Новый стиль: светлая «инженерная бумага», тёмно-синие чернила,
# сигнальный оранжевый и технический синий. Прямые углы, тонкие линии, сетка.
INK = RGBColor(0x0E, 0x1B, 0x2A)
TEAL = RGBColor(0x1B, 0x6C, 0xA8)        # технический синий
MAGENTA = RGBColor(0xE4, 0x57, 0x2E)     # сигнальный оранжевый
ORANGE = RGBColor(0xC9, 0x7B, 0x1E)      # тёплый янтарный
LIGHT = RGBColor(0xEF, 0xE9, 0xDD)
CARD = RGBColor(0xFF, 0xFF, 0xFF)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY = RGBColor(0x5B, 0x67, 0x70)
LINE = RGBColor(0xD8, 0xD3, 0xC8)
PAPER = RGBColor(0xF7, 0xF6, 0xF2)

PAPER_RGB = (0xF7, 0xF6, 0xF2)
INK_RGB = (0x0E, 0x1B, 0x2A)
ACCENT_RGB = (0xE4, 0x57, 0x2E)
BLUE_RGB = (0x1B, 0x6C, 0xA8)
GRID_RGB = (0xEC, 0xE6, 0xDA)
GRID_STRONG_RGB = (0xDF, 0xD7, 0xC8)

FONT = "Segoe UI"
W, H = 13.333, 7.5


def num(value, digits=1):
    if value is None:
        return "—"
    text = f"{value:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace(".", ",")


# ------------------------------------------------------------------ данные
def load_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def synthetic_metrics():
    data = load_json(QUALITY / "metrics.json", {})
    main = data.get("main", {})
    base = data.get("baseline_main", {})
    snow = data.get("snow", {})
    return {
        "mae": main.get("mae", 1.6), "dec": main.get("decision_accuracy", 0.99),
        "false_del": main.get("wrongly_deleted", 1), "keep": main.get("should_keep", 225),
        "missed": main.get("wrongly_kept", 2), "delete": main.get("should_delete", 75),
        "base_mae": base.get("mae", 7.4), "base_dec": base.get("decision_accuracy", 0.95),
        "snow_mae": snow.get("mae", 15.1), "count": main.get("count", 300),
        "real_check": data.get("real_check"),
    }


def real_metrics():
    info = load_json(ROOT / "models" / "cloud_net.json", {})
    train = set(info.get("real_images", []))
    csv_path = ROOT / "reports" / "real_quality_report.csv"
    used = []
    if csv_path.exists():
        import csv
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                if row.get("исключён") == "да" or row.get("файл") in train:
                    continue
                try:
                    used.append((float(row["человек, %"]), float(row["модель, %"])))
                except (KeyError, TypeError, ValueError):
                    continue
    if not used:
        estimates = load_json(ROOT / "data" / "real_eval" / "expert_estimates.json", [])
        used = [(e["expert_percent"], e["model_percent_384"])
                for e in estimates if not e.get("exclude") and e["file"] not in train]
    if not used:
        return {"n": 0, "mae": 9.2, "bias": -6.0, "points": [], "total": 0, "over75": 0}
    errors = [model - expert for expert, model in used]
    return {
        "n": len(used),
        "mae": sum(abs(e) for e in errors) / len(errors),
        "bias": sum(errors) / len(errors),
        "points": used,
        "over75": sum(1 for expert, _ in used if expert > 75),
    }


def model_info():
    return load_json(ROOT / "models" / "cloud_net.json", {})


def demo_message():
    path = ROOT / "reports" / "report_demo.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ("РЕЗУЛЬТАТ ОБРАБОТКИ СНИМКОВ\n"
            "Порог отсева: облачность выше 75 %\n\n"
            "СОХРАНЁННЫЕ ФАЙЛЫ\n"
            "  1. real_01.jpg — облачность 0,1 %\n"
            "  2. demo_02.jpg — облачность 42,0 %\n\n"
            "УДАЛЁННЫЕ ФАЙЛЫ\n"
            "  1. demo_05.jpg — облачность 95,6 %")


# ------------------------------------------------------------------ помощники
def add_text(slide, x, y, w, h, content, size=18, bold=False, color=INK,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font=FONT, spacing=1.0):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = Inches(0.02)
    frame.margin_top = frame.margin_bottom = 0
    lines = content.split("\n")
    for i, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.line_spacing = spacing
        run = paragraph.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = font
    return box


def add_bullets(slide, x, y, w, h, items, size=18, color=INK, spacing=1.15):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    first = True
    for item in items:
        if isinstance(item, tuple):
            text, level = item
        else:
            text, level = item, 0
        paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
        first = False
        paragraph.level = level
        paragraph.line_spacing = spacing
        paragraph.space_after = Pt(6)
        run = paragraph.add_run()
        run.text = ("•  " if level == 0 else "–  ") + text
        run.font.size = Pt(size if level == 0 else size - 2)
        run.font.color.rgb = color if level == 0 else GRAY
        run.font.name = FONT
    return box


def add_card(slide, x, y, w, h, fill=CARD, line=LINE, radius=False):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                   Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(0.75)
    shape.shadow.inherit = False
    return shape


def add_kpi(slide, x, y, w, value, label, accent=TEAL, height=1.35):
    add_card(slide, x, y, w, height)
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.06))
    rule.fill.solid()
    rule.fill.fore_color.rgb = accent
    rule.line.fill.background()
    add_text(slide, x + 0.2, y + 0.18, w - 0.35, 0.6, value, size=28, bold=True, color=INK)
    add_text(slide, x + 0.2, y + 0.82, w - 0.35, 0.45, label, size=12, color=GRAY)


def add_footer(slide, index, total):
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(H - 0.5),
                                  Inches(W - 1.1), Inches(0.012))
    line.fill.solid()
    line.fill.fore_color.rgb = LINE
    line.line.fill.background()
    add_text(slide, 0.55, H - 0.42, 8, 0.3, "Облачный контроль · кейс «Спутникс» · 14+", size=9, color=GRAY)
    add_text(slide, W - 1.25, H - 0.42, 0.7, 0.3, f"{index}/{total}", size=9, color=GRAY, align=PP_ALIGN.RIGHT)


def add_title(slide, title, subtitle=None, accent=MAGENTA):
    add_text(slide, 0.55, 0.30, 8.5, 0.25, "ОБЛАЧНЫЙ КОНТРОЛЬ · КЕЙС «СПУТНИКС»",
             size=9, bold=True, color=accent)
    add_text(slide, 0.55, 0.52, W - 1.1, 0.7, title, size=30, bold=True, color=INK)
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.57), Inches(1.24), Inches(1.1), Inches(0.04))
    rule.fill.solid()
    rule.fill.fore_color.rgb = accent
    rule.line.fill.background()
    if subtitle:
        add_text(slide, 0.57, 1.34, W - 1.1, 0.35, subtitle, size=13, color=GRAY)


def add_picture_fit(slide, path, x, y, w, h, border=True):
    path = Path(path)
    if not path.exists():
        add_card(slide, x, y, w, h, fill=LIGHT)
        add_text(slide, x + 0.2, y + h / 2 - 0.2, w - 0.4, 0.5,
                 "картинка появится после запуска демонстрации", size=13, color=GRAY, align=PP_ALIGN.CENTER)
        return None
    with Image.open(path) as image:
        iw, ih = image.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    left, top = x + (w - pw) / 2, y + (h - ph) / 2
    picture = slide.shapes.add_picture(str(path), Inches(left), Inches(top), Inches(pw), Inches(ph))
    if border:
        try:
            picture.line.color.rgb = LINE
            picture.line.width = Pt(0.75)
        except (AttributeError, ValueError):
            pass
    return picture


def add_table(slide, x, y, w, h, header, rows, widths=None, size=12, header_size=12):
    shape = slide.shapes.add_table(len(rows) + 1, len(header), Inches(x), Inches(y), Inches(w), Inches(h))
    table = shape.table
    table.first_row = True
    table.horz_banding = False
    if widths:
        for i, width in enumerate(widths):
            table.columns[i].width = Inches(width)
    for col, text in enumerate(header):
        cell = table.cell(0, col)
        cell.text = str(text)
        cell.fill.solid()
        cell.fill.fore_color.rgb = INK
        for paragraph in cell.text_frame.paragraphs:
            paragraph.alignment = PP_ALIGN.LEFT if col == 0 else PP_ALIGN.CENTER
            for run in paragraph.runs:
                run.font.size = Pt(header_size)
                run.font.bold = True
                run.font.color.rgb = WHITE
                run.font.name = FONT
    for r, row in enumerate(rows, start=1):
        for col, text in enumerate(row):
            cell = table.cell(r, col)
            cell.text = str(text)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if r % 2 else PAPER
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.LEFT if col == 0 else PP_ALIGN.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(size)
                    run.font.color.rgb = INK
                    run.font.name = FONT
    return table


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def new_slide(prs, background=True):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    if background:
        bg = ASSETS / "bg_grid.jpg"
        if bg.exists():
            slide.shapes.add_picture(str(bg), 0, 0, Inches(W), Inches(H))
    return slide


def add_arrow(slide, x, y, size=0.34, color=TEAL):
    arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(size), Inches(size * 0.5))
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = color
    arrow.line.fill.background()


# ------------------------------------------------------------------ слайды
def slide_title(prs):
    slide = new_slide(prs, background=False)
    bg = ASSETS / "bg_title.jpg"
    if bg.exists():
        slide.shapes.add_picture(str(bg), 0, 0, Inches(W), Inches(H))
    else:
        back = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(W), Inches(H))
        back.fill.solid()
        back.fill.fore_color.rgb = PAPER
        back.line.fill.background()

    add_text(slide, 0.65, 0.72, 7.0, 0.3, "КЕЙС «КОСМИЧЕСКИЕ ТЕХНОЛОГИИ» · СПУТНИКС",
             size=10, bold=True, color=MAGENTA)
    add_text(slide, 0.65, 1.22, 7.2, 1.7, "Облачный\nконтроль", size=54, bold=True, color=INK, spacing=1.0)
    add_text(slide, 0.65, 3.02, 6.9, 1.0,
             "Искусственный интеллект распознаёт облачность\nна спутниковом снимке и отсеивает облачные кадры",
             size=18, color=GRAY, spacing=1.2)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.68), Inches(4.12), Inches(1.5), Inches(0.05))
    line.fill.solid()
    line.fill.fore_color.rgb = MAGENTA
    line.line.fill.background()
    add_text(slide, 0.65, 4.38, 7.2, 1.3,
             "Команда: впишите имена участников\nВозраст: 14+\nНаправление: космические технологии",
             size=13, color=GRAY, spacing=1.3)
    add_text(slide, 0.65, 6.72, 7.0, 0.3, "Sk Challenge 2026 Kids · 14+", size=10, color=GRAY)

    add_text(slide, 8.35, 1.95, 4.3, 0.3, "ПОРОГ ОТСЕВА", size=10, bold=True,
             color=RGBColor(0xF0, 0xA6, 0x8C))
    add_text(slide, 8.35, 2.3, 4.4, 1.4, "75 %", size=86, bold=True, color=WHITE)
    add_text(slide, 8.38, 3.78, 4.2, 1.1, "облачности выше —\nснимок удаляется",
             size=15, color=RGBColor(0xE6, 0xE9, 0xEE), spacing=1.2)
    add_text(slide, 8.38, 5.55, 4.2, 0.8, "14 → 32 → 16 → 1\n1025 весов · numpy",
             size=11, color=RGBColor(0x9F, 0xB4, 0xC8), font="Consolas", spacing=1.2)
    add_notes(slide, "Представьте команду и одним предложением — что делает решение. "
                     "Не читайте слайд, скажите главное: «Мы считаем облачность и удаляем кадры, где облаков больше 75 %».")
    return slide


def slide_problem(prs, index, total):
    slide = new_slide(prs)
    add_title(slide, "Проблема", "Спутник снимает тысячи кадров, но Земли на многих не видно")
    add_bullets(slide, 0.65, 1.85, 7.3, 4.4, [
        "Облака закрывают Землю — снимок бесполезен для карт, полей и мониторинга.",
        "Проверять каждый кадр вручную долго и дорого.",
        "Нужно автоматически отсеивать заведомо облачные кадры и оставлять пригодные.",
        ("Важно не удалить хороший снимок: вернуть его уже нельзя.", 1),
    ], size=19)
    add_card(slide, 8.35, 1.9, 4.35, 2.05, fill=INK, line=None)
    add_text(slide, 8.65, 2.05, 3.8, 0.5, "Порог кейса", size=13, color=RGBColor(0xF0, 0xA6, 0x8C))
    add_text(slide, 8.65, 2.45, 3.8, 1.0, "75 %", size=56, bold=True, color=WHITE)
    add_text(slide, 8.65, 3.45, 3.7, 0.4, "облачности — и кадр удаляется", size=13, color=RGBColor(0xD8, 0xDE, 0xE6))
    add_card(slide, 8.35, 4.25, 4.35, 1.75)
    add_text(slide, 8.6, 4.45, 3.9, 1.4,
             "Две ошибки стоят по-разному:\nпропуск — лишний файл на диске;\nложное удаление — потерянный снимок.",
             size=13.5, color=INK, spacing=1.2)
    add_footer(slide, index, total)
    add_notes(slide, "Подчеркните: задача не просто «найти облака», а принять безопасное решение по всему снимку.")
    return slide


def slide_requirements(prs, index, total):
    slide = new_slide(prs)
    add_title(slide, "Требования кейса", "Три обязательных пункта — все реализованы")
    cards = [
        ("1", "Процент облачности", "Для каждого снимка — доля площади, закрытой облаками.", TEAL),
        ("2", "Удаление выше 75 %", "Снимок с облачностью больше 75 % удаляется с диска.", MAGENTA),
        ("3", "Сообщение оператору", "Имя сохранённого файла, процент облачности, имя удалённого файла.", ORANGE),
    ]
    for i, (number, title, text, color) in enumerate(cards):
        x = 0.65 + i * 4.05
        add_card(slide, x, 1.9, 3.8, 3.1)
        badge = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.28), Inches(2.12), Inches(0.62), Inches(0.62))
        badge.fill.solid()
        badge.fill.fore_color.rgb = color
        badge.line.fill.background()
        add_text(slide, x + 0.28, 2.2, 0.62, 0.5, number, size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text(slide, x + 0.28, 2.95, 3.25, 0.5, title, size=17, bold=True, color=INK)
        add_text(slide, x + 0.28, 3.5, 3.25, 1.3, text, size=13.5, color=GRAY, spacing=1.2)
    add_card(slide, 0.65, 5.3, 12.0, 0.95, fill=RGBColor(0xF1, 0xE9, 0xDD))
    add_text(slide, 0.9, 5.48, 11.5, 0.6,
             "Результат решения кейса — презентация .pptx с описанием этапов и демонстрацией работы алгоритма.",
             size=15, bold=True, color=TEAL)
    add_footer(slide, index, total)
    add_notes(slide, "Пройдите по трём пунктам и покажите, что каждый закрыт в интерфейсе и в отчёте.")
    return slide


def slide_pipeline(prs, index, total):
    slide = new_slide(prs)
    add_title(slide, "Как устроено решение", "От пикселя до сообщения оператору")
    steps = [("Снимок", "JPG, PNG,\nTIFF, 8/16 бит"), ("14 признаков", "цвет, яркость,\nокрестности"),
             ("Нейросеть", "14→32→16→1\n1025 весов"), ("Маска", "вероятность\nоблака > 0,5"),
             ("Процент", "облачные / все\nпиксели с данными"), ("Решение", "> 75 % —\nудалить")]
    x = 0.62
    for i, (title, text) in enumerate(steps):
        add_card(slide, x, 2.05, 1.78, 1.75)
        add_text(slide, x + 0.12, 2.22, 1.55, 0.45, title, size=15, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
        add_text(slide, x + 0.1, 2.78, 1.58, 0.9, text, size=11.5, color=GRAY, align=PP_ALIGN.CENTER, spacing=1.1)
        if i < len(steps) - 1:
            add_arrow(slide, x + 1.83, 2.72)
        x += 2.07

    add_card(slide, 0.65, 4.35, 12.0, 1.55, fill=LIGHT)
    add_text(slide, 0.95, 4.55, 11.4, 0.45, "Сообщение оператору", size=16, bold=True, color=INK)
    add_text(slide, 0.95, 5.0, 11.4, 0.7,
             "СОХРАНЁННЫЕ ФАЙЛЫ: имя файла — облачность 10,5 %     ·     "
             "УДАЛЁННЫЕ ФАЙЛЫ: имя файла — облачность 88,2 %",
             size=13.5, color=GRAY)
    add_text(slide, 0.65, 6.1, 12.0, 0.4,
             "Отчёт сохраняется в трёх видах: текст, CSV для Excel и JSON для программ.",
             size=12.5, color=GRAY)
    add_footer(slide, index, total)
    add_notes(slide, "Проведите пальцем по схеме слева направо. Скажите, что вся логика открыта и помещается в несколько файлов.")
    return slide


def slide_data(prs, index, total, info):
    slide = new_slide(prs)
    add_title(slide, "Данные для обучения", "Синтетика даёт точные маски, реальные снимки — настоящую погоду")
    add_card(slide, 0.65, 1.8, 5.85, 2.35)
    add_text(slide, 0.9, 1.98, 5.3, 0.45, "Учебные снимки генератора", size=16, bold=True, color=TEAL)
    add_bullets(slide, 0.9, 2.5, 5.25, 1.5, [
        "600 сцен: океан, лес, пустыня, город, снег",
        "Точная маска каждого облачного пикселя",
        "240 000 учебных пикселей",
    ], size=13)
    add_card(slide, 6.8, 1.8, 5.85, 2.35)
    add_text(slide, 7.05, 1.98, 5.3, 0.45, "Настоящие снимки СПУТНИКС", size=16, bold=True, color=MAGENTA)
    add_bullets(slide, 7.05, 2.5, 5.25, 1.5, [
        "88 снимков скачано с открытого диска",
        f"{len(info.get('real_images', []))} снимков размечено вручную",
        f"{info.get('real_pixels', 61004):,} размеченных пикселей".replace(",", " "),
    ], size=13)
    add_picture_fit(slide, ASSETS / "real_samples.jpg", 0.65, 4.4, 7.3, 2.55)
    add_card(slide, 8.2, 4.4, 4.45, 2.55, fill=LIGHT)
    add_text(slide, 8.45, 4.6, 4.0, 0.45, "Почему так", size=15, bold=True, color=INK)
    add_text(slide, 8.45, 5.05, 4.0, 1.75,
             "Размечать сотни настоящих снимков вручную долго. Генератор даёт точные ответы бесплатно, "
             "а настоящие снимки показывают, где модель ошибается.",
             size=12.5, color=GRAY, spacing=1.2)
    add_footer(slide, index, total)
    add_notes(slide, "Назовите точные числа: 600 синтетических сцен, 88 реальных снимков, 11 размечено, 61 004 пикселя. "
                     "Если спросят, почему не только реальные — из-за стоимости ручной разметки.")
    return slide


def slide_features(prs, index, total):
    slide = new_slide(prs)
    add_title(slide, "Что видит нейросеть", "14 чисел на каждый пиксель вместо трёх цветов")
    add_bullets(slide, 0.65, 1.8, 5.2, 4.5, [
        "Цвет: R, G, B — три канала.",
        "Яркость и самый тёмный канал.",
        "Насыщенность: облако серое, лес и море цветные.",
        "Синий минус красный: вода и дымка синее.",
        "Яркость вокруг в окнах 5×5, 15×15, 41×41.",
        "Пестрота вокруг: гладкое облако или рябая земля.",
        "Минимум вокруг 7×7: рядом тёмное — значит, пятно маленькое.",
    ], size=16)
    header = ["Признак", "Облако", "Лес", "Море"]
    rows = [["яркость", "0,92", "0,53", "0,29"],
            ["самый тёмный канал", "0,87", "0,39", "0,16"],
            ["насыщенность", "0,06", "0,24", "0,37"],
            ["яркость вокруг 15", "0,89", "0,44", "0,22"],
            ["вероятность облака", "1,00", "0,003", "0,000"]]
    add_table(slide, 6.2, 2.0, 6.5, 2.6, header, rows, widths=[3.0, 1.15, 1.15, 1.2], size=13)
    add_card(slide, 6.2, 4.95, 6.5, 1.55, fill=LIGHT)
    add_text(slide, 6.45, 5.12, 6.0, 1.2,
             "Одной яркости мало: светлыми бывают песок, снег, крыши и блики. "
             "Поэтому добавлены признаки окрестностей.",
             size=13, color=GRAY, spacing=1.2)
    add_footer(slide, index, total)
    add_notes(slide, "Покажите таблицу: сеть получает не «картинку», а 14 понятных чисел. Это делает решение объяснимым.")
    return slide


def slide_network(prs, index, total):
    slide = new_slide(prs)
    add_title(slide, "Нейросеть", "Маленькая, быстрая и понятная")
    layers = [("14", "признаков", 1.0, TEAL), ("32", "нейрона", 4.15, TEAL),
              ("16", "нейронов", 7.3, TEAL), ("1", "вероятность", 10.45, MAGENTA)]
    for value, label, x, color in layers:
        circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(2.2), Inches(1.75), Inches(1.75))
        circle.fill.solid()
        circle.fill.fore_color.rgb = color
        circle.line.fill.background()
        add_text(slide, x, 2.55, 1.75, 0.6, value, size=30, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text(slide, x - 0.2, 4.05, 2.15, 0.4, label, size=13, color=GRAY, align=PP_ALIGN.CENTER)
    for x in (2.95, 6.1, 9.25):
        add_arrow(slide, x, 2.85, size=0.55)
    add_card(slide, 0.65, 4.75, 6.0, 1.7, fill=INK, line=None)
    add_text(slide, 0.9, 4.92, 5.5, 1.35,
             "h1 = ReLU(x · W1 + b1)\nh2 = ReLU(h1 · W2 + b2)\np  = sigmoid(h2 · W3 + b3)",
             size=14, color=WHITE, font="Consolas", spacing=1.25)
    add_bullets(slide, 7.0, 4.7, 5.6, 1.9, [
        "1025 весов — всё, что сеть «знает»",
        "Работает на обычном компьютере без видеокарты",
        "Считает снимок за доли секунды",
    ], size=15)
    add_footer(slide, index, total)
    add_notes(slide, "Скажите, что это настоящая нейросеть, просто маленькая: три слоя, 1025 весов. "
                     "Её можно объяснить по шагам — в этом сила решения.")
    return slide


def slide_training(prs, index, total, info):
    slide = new_slide(prs)
    add_title(slide, "Обучение", "Веса подбираются по примерам, а не придумываются вручную")
    steps = [("1", "Показать пример", "пиксель и правильный ответ"),
             ("2", "Получить ошибку", "сеть отвечает и ошибается"),
             ("3", "Поправить веса", "чуть-чуть, в сторону меньшей ошибки"),
             ("4", "Повторить", "сотни тысяч пикселей, десятки кругов")]
    for i, (number, title, text) in enumerate(steps):
        x = 0.65 + i * 3.08
        add_card(slide, x, 1.85, 2.85, 1.65)
        add_text(slide, x + 0.2, 2.0, 0.5, 0.45, number, size=22, bold=True, color=MAGENTA)
        add_text(slide, x + 0.2, 2.5, 2.45, 0.4, title, size=14, bold=True, color=INK)
        add_text(slide, x + 0.2, 2.95, 2.45, 0.45, text, size=11.5, color=GRAY)
    acc = info.get("pixel_accuracy_holdout", 0.982)
    add_kpi(slide, 0.65, 3.95, 3.8, "98,2 %", "верных пикселей на отложенных данных", TEAL)
    add_kpi(slide, 4.75, 3.95, 3.8, "240 000", "учебных пикселей из генератора", TEAL)
    add_kpi(slide, 8.85, 3.95, 3.8, "61 004", "размеченных реальных пикселя", MAGENTA)
    if acc != 0.982:
        add_text(slide, 0.65, 5.45, 12, 0.3, f"Точность текущей модели: {num(acc * 100)} %", size=11, color=GRAY)
    add_card(slide, 0.65, 5.55, 12.0, 1.0, fill=LIGHT)
    add_text(slide, 0.9, 5.72, 11.5, 0.65,
             "Каждый восьмой пиксель сеть при обучении не видит — на нём и проверяют, чему она научилась.",
             size=14, color=INK)
    add_footer(slide, index, total)
    add_notes(slide, "Не углубляйтесь в математику: достаточно идеи «показали пример — поправили веса». "
                     "Цифра 98,2 % — только на учебных пикселях, на реальных снимках она ниже.")
    return slide


def slide_decision(prs, index, total):
    slide = new_slide(prs)
    add_title(slide, "От вероятности к решению", "Пять шагов, которые превращают карту в приказ «удалить»")
    steps = [("1", "Карта вероятностей", "для каждого пикселя — число от 0 до 1"),
             ("2", "Сглаживание 3×3", "одинокий пиксель среди леса — ошибка"),
             ("3", "Порог 0,5", "вероятность выше 0,5 — облако"),
             ("4", "Чёрные поля", "края без данных не считаются"),
             ("5", "Процент и решение", "облачные / все пиксели с данными × 100")]
    for i, (number, title, text) in enumerate(steps):
        y = 1.95 + i * 0.86
        badge = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.7), Inches(y), Inches(0.5), Inches(0.5))
        badge.fill.solid()
        badge.fill.fore_color.rgb = TEAL if i < 4 else MAGENTA
        badge.line.fill.background()
        add_text(slide, 0.7, y + 0.04, 0.5, 0.4, number, size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text(slide, 1.4, y - 0.02, 4.1, 0.4, title, size=15, bold=True, color=INK)
        add_text(slide, 1.4, y + 0.32, 5.2, 0.35, text, size=11.5, color=GRAY)
    add_card(slide, 7.15, 1.9, 5.5, 3.7, fill=INK, line=None)
    add_text(slide, 7.45, 2.1, 4.9, 0.4, "Правило кейса", size=13, color=RGBColor(0xF0, 0xA6, 0x8C))
    add_text(slide, 7.45, 2.5, 4.9, 0.9, "облачность > 75 %", size=26, bold=True, color=WHITE)
    add_text(slide, 7.45, 3.35, 4.8, 1.9,
             "→ снимок удаляется.\nРовно 75 % — ещё сохраняется:\nв задании сказано «выше 75 %».\n\n"
             "Битый файл не удаляется:\nсначала читаем, потом решаем.",
             size=13, color=RGBColor(0xE6, 0xE9, 0xEE), spacing=1.25)
    add_card(slide, 7.15, 5.8, 5.5, 0.75, fill=RGBColor(0xF7, 0xE9, 0xE0), line=None)
    add_text(slide, 7.4, 5.96, 5.0, 0.5, "Есть режимы «корзина» и «пробный запуск»", size=12.5, color=MAGENTA)
    add_footer(slide, index, total)
    add_notes(slide, "Главная фраза: «Сначала читаем файл, потом удаляем. Битый файл не трогаем».")
    return slide


def slide_demo(prs, index, total):
    slide = new_slide(prs)
    add_title(slide, "Демонстрация работы", "Интерфейс оператора: загрузка, проценты, маски и сообщение")
    screenshot = ASSETS / "ui_report.png"
    add_picture_fit(slide, screenshot, 0.55, 1.75, 8.05, 5.25)
    add_card(slide, 8.85, 1.75, 3.95, 5.25, fill=RGBColor(0x14, 0x21, 0x3D), line=None)
    add_text(slide, 9.05, 1.92, 3.55, 0.4, "Сообщение оператору", size=13, bold=True, color=RGBColor(0xF0, 0xA6, 0x8C))
    message = demo_message()
    lines = message.splitlines()
    if len(lines) > 20:
        lines = lines[:20] + ["…"]
    add_text(slide, 9.05, 2.35, 3.55, 4.3, "\n".join(lines), size=9.5, color=WHITE, font="Consolas", spacing=1.12)
    add_footer(slide, index, total)
    add_notes(slide, "Покажите: слева снимок, справа маска облаков, внизу сообщение. "
                     "Обязательно назовите три вещи: имя сохранённого файла, процент, имя удалённого файла.")
    return slide


def slide_examples(prs, index, total):
    slide = new_slide(prs)
    add_title(slide, "Примеры работы", "В каждой колонке: снимок, правильная маска и ответ сети")
    add_picture_fit(slide, QUALITY / "typical.jpg", 0.55, 1.75, 7.4, 2.95)
    add_text(slide, 0.55, 4.8, 7.4, 0.4, "Обычные снимки: облака найдены, лишнего почти нет",
             size=13, bold=True, color=TEAL)
    add_picture_fit(slide, QUALITY / "worst.jpg", 8.2, 1.75, 4.55, 2.95)
    add_text(slide, 8.2, 4.8, 4.55, 0.4, "Самые большие ошибки: снег и дымка",
             size=13, bold=True, color=MAGENTA)
    add_card(slide, 0.55, 5.35, 7.4, 1.45, fill=LIGHT)
    add_text(slide, 0.8, 5.5, 7.0, 0.4, "Что видно", size=14, bold=True, color=INK)
    add_text(slide, 0.8, 5.9, 7.0, 0.8,
             "Малиновым закрашены облака. Маска повторяет форму облаков, а не всё светлое пятно. "
             "Ошибки — на снегу, льду и тонкой дымке.",
             size=12, color=GRAY, spacing=1.15)
    add_card(slide, 8.2, 5.35, 4.55, 1.45, fill=RGBColor(0xF1, 0xE9, 0xDD))
    add_text(slide, 8.45, 5.5, 4.1, 0.4, "Цифра", size=14, bold=True, color=TEAL)
    add_text(slide, 8.45, 5.9, 4.1, 0.8, "Средняя ошибка на 300 новых учебных снимках — 1,7 п.п.",
             size=13, color=INK, spacing=1.15)
    add_footer(slide, index, total)
    add_notes(slide, "Покажите, что сеть находит форму облаков, а не просто яркие пятна. "
                     "Справа честно показаны худшие случаи — снег и дымка.")
    return slide


def slide_quality(prs, index, total, metrics):
    slide = new_slide(prs)
    add_title(slide, "Качество на учебных снимках", f"{metrics['count']} снимков, которых сеть не видела")
    add_kpi(slide, 0.65, 1.8, 3.8, f"{num(metrics['mae'])} п.п.", "средняя ошибка процента облачности", TEAL)
    add_kpi(slide, 4.75, 1.8, 3.8, f"{num(metrics['dec'] * 100)} %", "верных решений «сохранить / удалить»", TEAL)
    add_kpi(slide, 8.85, 1.8, 3.8, f"{metrics['false_del']} из {metrics['keep']}", "хороших снимков удалено по ошибке", MAGENTA)

    chart_data = CategoryChartData()
    chart_data.categories = ["Нейросеть", "Простое правило"]
    chart_data.add_series("Средняя ошибка, п.п.", (metrics["mae"], metrics["base_mae"]))
    chart = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.65), Inches(3.55),
                                   Inches(5.7), Inches(2.9), chart_data).chart
    chart.font.name = FONT
    chart.font.size = Pt(10)
    chart.font.color.rgb = GRAY
    chart.has_legend = False
    chart.has_title = True
    chart.chart_title.text_frame.text = "Средняя ошибка процента облачности"
    for paragraph in chart.chart_title.text_frame.paragraphs:
        for run in paragraph.runs:
            run.font.size = Pt(12)
            run.font.bold = True
            run.font.color.rgb = INK
            run.font.name = FONT
    plot = chart.plots[0]
    plot.has_data_labels = True
    plot.data_labels.font.size = Pt(11)
    plot.data_labels.number_format = "0.0"
    plot.data_labels.number_format_is_linked = False
    chart.value_axis.minimum_scale = 0
    chart.value_axis.maximum_scale = 8
    chart.value_axis.major_unit = 2
    series = plot.series[0]
    series.format.fill.solid()
    series.format.fill.fore_color.rgb = TEAL

    chart_data2 = CategoryChartData()
    chart_data2.categories = ["Нейросеть", "Простое правило"]
    chart_data2.add_series("Верных решений, %", (metrics["dec"] * 100, metrics["base_dec"] * 100))
    chart2 = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(6.6), Inches(3.55),
                                    Inches(6.0), Inches(2.9), chart_data2).chart
    chart2.font.name = FONT
    chart2.font.size = Pt(10)
    chart2.font.color.rgb = GRAY
    chart2.has_legend = False
    chart2.has_title = True
    chart2.chart_title.text_frame.text = "Верных решений, %"
    for paragraph in chart2.chart_title.text_frame.paragraphs:
        for run in paragraph.runs:
            run.font.size = Pt(12)
            run.font.bold = True
            run.font.color.rgb = INK
            run.font.name = FONT
    plot2 = chart2.plots[0]
    plot2.has_data_labels = True
    plot2.data_labels.font.size = Pt(11)
    plot2.data_labels.number_format = "0"
    plot2.data_labels.number_format_is_linked = False
    chart2.value_axis.minimum_scale = 0
    chart2.value_axis.maximum_scale = 100
    chart2.value_axis.major_unit = 20
    plot2.series[0].format.fill.solid()
    plot2.series[0].format.fill.fore_color.rgb = MAGENTA
    add_footer(slide, index, total)
    add_notes(slide, "Сравнение честное: простое правило «яркий = облако» подобрано так, чтобы работать как можно лучше. "
                     "Нейросеть ошибается в 4–5 раз меньше.")
    return slide


def slide_real(prs, index, total, real):
    slide = new_slide(prs)
    add_title(slide, "Качество на настоящих снимках", "Проверка на снимках СПУТНИКС, которых не было в обучении")
    add_kpi(slide, 0.65, 1.8, 3.8, f"{num(real['mae'])} п.п.", f"средняя ошибка на {real['n']} снимках", ORANGE)
    bias_text = ("+" if real["bias"] > 0 else "") + num(real["bias"])
    add_kpi(slide, 4.75, 1.8, 3.8, f"{bias_text} п.п.", "смещение: модель занижает облачность", ORANGE)
    add_kpi(slide, 8.85, 1.8, 3.8, f"{real['over75']}", "снимков > 75 % в наборе оценки", MAGENTA)

    chart_data = XyChartData()
    series = chart_data.add_series("Снимки СПУТНИКС")
    for x, y in real["points"]:
        series.add_data_point(x, y)
    series2 = chart_data.add_series("Идеальное совпадение")
    series2.add_data_point(0, 0)
    series2.add_data_point(100, 100)
    chart = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, Inches(0.65), Inches(3.45),
                                   Inches(7.3), Inches(3.35), chart_data).chart
    chart.font.name = FONT
    chart.font.size = Pt(10)
    chart.font.color.rgb = GRAY
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(10)
    chart.has_title = True
    chart.chart_title.text_frame.text = "Оценка человека → ответ модели, %"
    for paragraph in chart.chart_title.text_frame.paragraphs:
        for run in paragraph.runs:
            run.font.size = Pt(12)
            run.font.bold = True
            run.font.color.rgb = INK
            run.font.name = FONT
    plot = chart.plots[0]
    plot.series[0].marker.style = XL_MARKER_STYLE.CIRCLE
    plot.series[0].marker.size = 6
    plot.series[0].marker.format.fill.solid()
    plot.series[0].marker.format.fill.fore_color.rgb = TEAL
    plot.series[0].format.line.fill.background()
    plot.series[1].marker.style = XL_MARKER_STYLE.NONE
    plot.series[1].format.line.color.rgb = MAGENTA
    plot.series[1].format.line.width = Pt(1.25)
    for axis in (chart.category_axis, chart.value_axis):
        axis.minimum_scale = 0
        axis.maximum_scale = 100
        axis.major_unit = 25

    add_card(slide, 8.2, 3.45, 4.5, 3.35, fill=RGBColor(0xFF, 0xFA, 0xF0), line=RGBColor(0xE8, 0xD3, 0xA8))
    add_text(slide, 8.45, 3.62, 4.0, 0.4, "Честный вывод", size=15, bold=True, color=RGBColor(0x8A, 0x52, 0x00))
    add_text(slide, 8.45, 4.05, 4.0, 2.6,
             "Модель занижает облачность в среднем на 6 п.п. "
             "Это безопаснее ложного удаления, но облачные кадры она может пропустить.\n\n"
             "Главное ограничение набора: в нём нет ни одного снимка выше 75 %, "
             "поэтому ветка «удалить» проверена на учебных снимках.",
             size=12, color=RGBColor(0x6B, 0x4A, 0x12), spacing=1.18)
    add_footer(slide, index, total)
    add_notes(slide, "Не прячьте ограничение: скажите, что реальный набор оказался почти безоблачным. "
                     "Это честно и показывает, что команда понимает разницу между «сохранить» и «удалить».")
    return slide


def slide_limits(prs, index, total):
    slide = new_slide(prs)
    add_title(slide, "Где модель ошибается", "Снег, дымка и плохая экспозиция — известные ограничения")
    limits = [("Снег и лёд", "Белый снег похож на белое облако. Нужны инфракрасные каналы.", MAGENTA),
              ("Тонкая дымка", "Полупрозрачные облака почти не отличаются от земли.", ORANGE),
              ("Тёмный / пересвет", "Меняется яркость — меняется процент облачности.", TEAL)]
    for i, (title, text, color) in enumerate(limits):
        y = 1.85 + i * 1.15
        add_card(slide, 0.65, y, 6.0, 0.98)
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.65), Inches(y), Inches(0.08), Inches(0.98))
        bar.fill.solid()
        bar.fill.fore_color.rgb = color
        bar.line.fill.background()
        add_text(slide, 0.9, y + 0.1, 5.5, 0.35, title, size=14, bold=True, color=INK)
        add_text(slide, 0.9, y + 0.45, 5.5, 0.45, text, size=11.5, color=GRAY)
    add_picture_fit(slide, QUALITY / "snow.jpg", 6.95, 1.8, 5.75, 2.9)
    add_card(slide, 6.95, 4.9, 5.75, 1.9, fill=LIGHT)
    add_text(slide, 7.2, 5.05, 5.3, 0.4, "Что мы сделали", size=14, bold=True, color=INK)
    add_text(slide, 7.2, 5.45, 5.3, 1.25,
             "Проверили обучение на снегу: ошибка падает, но сеть начинает пропускать настоящие белые облака. "
             "Вывод: обучение не убирает неоднозначность, а переносит ошибку.",
             size=11.5, color=GRAY, spacing=1.15)
    add_footer(slide, index, total)
    add_notes(slide, "Жюри ценит честность. Скажите: «В цветном снимке нет инфракрасных каналов, поэтому снег и облако "
                     "почти неразличимы. Мы это измерили и описали».")
    return slide


def slide_experiments(prs, index, total):
    slide = new_slide(prs)
    add_title(slide, "Опыты и выводы", "Что мы проверили, кроме базового обучения")
    header = ["Опыт", "Что меняли", "Результат"]
    rows = [["Снег в обучении", "0 % → 25 % снежных сцен", "Ошибка на снегу 14,4 → 12,1 п.п., но белые облака стали пропускаться"],
            ["Размер сети", "(32, 16) → (4,) и (64, 32)", "Маленькая сеть хуже, большая почти не выигрывает"],
            ["Меньше данных", "600 → 30 сцен", "Ошибка заметно растёт: данных важно много"],
            ["Порог отсева", "50 / 75 / 90 %", "Порог выбирается под задачу, а не «на глаз»"]]
    add_table(slide, 0.65, 1.9, 12.0, 2.8, header, rows, widths=[2.4, 4.2, 5.4], size=11.5)
    add_card(slide, 0.65, 5.0, 5.85, 1.75, fill=RGBColor(0xF1, 0xE9, 0xDD))
    add_text(slide, 0.9, 5.15, 5.3, 0.4, "Главный вывод", size=14, bold=True, color=TEAL)
    add_text(slide, 0.9, 5.55, 5.3, 1.1,
             "Качество ограничено данными, а не размером сети. Больше размеченных облачных снимков — "
             "главный путь к улучшению.",
             size=12, color=INK, spacing=1.15)
    add_card(slide, 6.8, 5.0, 5.85, 1.75, fill=LIGHT)
    add_text(slide, 7.05, 5.15, 5.3, 0.4, "Что дальше", size=14, bold=True, color=INK)
    add_text(slide, 7.05, 5.55, 5.3, 1.1,
             "Добавить 10–15 реальных облачных снимков, построить полные маски, "
             "попробовать инфракрасный канал и свёрточную сеть.",
             size=12, color=GRAY, spacing=1.15)
    add_footer(slide, index, total)
    add_notes(slide, "Расскажите один опыт подробно — например, про снег. Это показывает, что команда умеет "
                     "не только обучать, но и делать выводы.")
    return slide


def slide_final(prs, index, total):
    slide = new_slide(prs)
    add_title(slide, "Итоги", "Все требования кейса закрыты, ограничения описаны честно")
    checks = [("Процент облачности для каждого снимка", True),
              ("Удаление снимков выше 75 %", True),
              ("Сообщение: файл, процент, удалённый файл", True),
              ("Обучение на снимках СПУТНИКС", True),
              ("Презентация и демонстрация", True)]
    for i, (text, done) in enumerate(checks):
        y = 1.85 + i * 0.72
        mark = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.7), Inches(y), Inches(0.42), Inches(0.42))
        mark.fill.solid()
        mark.fill.fore_color.rgb = TEAL if done else ORANGE
        mark.line.fill.background()
        add_text(slide, 0.7, y + 0.03, 0.42, 0.35, "✓" if done else "!", size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text(slide, 1.3, y - 0.02, 6.5, 0.45, text, size=15, color=INK)
    add_card(slide, 8.15, 1.8, 4.55, 4.1, fill=INK, line=None)
    add_text(slide, 8.45, 2.0, 4.0, 0.4, "Что можно улучшить", size=14, bold=True, color=RGBColor(0xF0, 0xA6, 0x8C))
    add_text(slide, 8.45, 2.45, 4.0, 3.2,
             "• Больше реальных облачных снимков\n"
             "• Полные маски для попиксельной проверки\n"
             "• Калибровка занижения облачности\n"
             "• Инфракрасные каналы\n"
             "• Свёрточная нейросеть",
             size=13, color=RGBColor(0xE6, 0xE9, 0xEE), spacing=1.35)
    add_text(slide, 0.7, 5.6, 7.0, 0.5, "Спасибо за внимание!", size=24, bold=True, color=MAGENTA)
    add_text(slide, 0.7, 6.1, 7.0, 0.4, "Готовы ответить на вопросы.", size=13, color=GRAY)
    add_footer(slide, index, total)
    add_notes(slide, "Закончите тремя фразами: что сделали, какие цифры получили, что улучшим. Не тяните с финалом.")
    return slide


# ------------------------------------------------------------------ картинки
def make_montage(paths, out, width=1500, gap=16):
    """Собирает несколько снимков в одну полосу для слайда."""
    images = []
    for path in paths:
        if Path(path).exists():
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
    if not images:
        return False
    tile_w = (width - gap * (len(images) - 1)) // len(images)
    tiles = []
    for image in images:
        height = round(image.height * tile_w / image.width)
        tiles.append(image.resize((tile_w, height), Image.LANCZOS))
    height = max(t.height for t in tiles)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, (height - tile.height) // 2))
        x += tile_w + gap
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=90)
    return True


def prepare_assets():
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_backgrounds()
    make_montage([ROOT / "data" / "real_extra" / name for name in (
        "2025-02-26_16-16-54_SXC3-2323_1.jpg",
        "2025-09-24_12-13-19_SXC3-232_1.jpg",
        "2026-06-03_14-09-03_SXC3-2331_1.jpg",
    )], ASSETS / "real_samples.jpg")


def make_backgrounds():
    """Фон слайдов: светлая «инженерная бумага» с сеткой и отдельный титульный фон."""
    wp, hp = 1920, 1080

    grid = Image.new("RGB", (wp, hp), PAPER_RGB)
    draw = ImageDraw.Draw(grid)
    for x in range(0, wp, 48):
        color = GRID_STRONG_RGB if x % 240 == 0 else GRID_RGB
        draw.line([(x, 0), (x, hp)], fill=color, width=1)
    for y in range(0, hp, 48):
        color = GRID_STRONG_RGB if y % 240 == 0 else GRID_RGB
        draw.line([(0, y), (wp, y)], fill=color, width=1)
    grid.save(ASSETS / "bg_grid.jpg", quality=88)

    title = Image.new("RGB", (wp, hp), PAPER_RGB)
    draw = ImageDraw.Draw(title)
    panel_left = int(wp * 0.60)
    for x in range(0, panel_left, 48):
        color = GRID_STRONG_RGB if x % 240 == 0 else GRID_RGB
        draw.line([(x, 0), (x, hp)], fill=color, width=1)
    for y in range(0, hp, 48):
        color = GRID_STRONG_RGB if y % 240 == 0 else GRID_RGB
        draw.line([(0, y), (panel_left, y)], fill=color, width=1)
    draw.rectangle([panel_left, 0, wp, hp], fill=INK_RGB)
    draw.rectangle([panel_left - 4, 0, panel_left, hp], fill=ACCENT_RGB)

    cx, cy = int(wp * 0.80), int(hp * 0.45)
    for radius in (110, 210, 330, 470):
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                     outline=(45, 66, 88), width=2)
    draw.line([(panel_left + 50, hp - 90), (wp - 30, 120)], fill=(66, 94, 122), width=2)
    draw.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=ACCENT_RGB)
    for angle, radius, color in ((0, 110, BLUE_RGB), (140, 210, ACCENT_RGB), (220, 330, (190, 202, 214))):
        x = cx + int(radius * math.cos(math.radians(angle)))
        y = cy + int(radius * math.sin(math.radians(angle)))
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color)
    title.save(ASSETS / "bg_title.jpg", quality=90)


# ------------------------------------------------------------------ сборка
def main() -> int:
    parser = argparse.ArgumentParser(description="Сборка презентации .pptx")
    parser.add_argument("--out", default=str(ROOT / "Презентация_Облачный_контроль.pptx"))
    args = parser.parse_args()

    prepare_assets()
    metrics = synthetic_metrics()
    real = real_metrics()
    info = model_info()

    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)

    builders = [
        lambda i, n: slide_title(prs),
        lambda i, n: slide_problem(prs, i, n),
        lambda i, n: slide_requirements(prs, i, n),
        lambda i, n: slide_pipeline(prs, i, n),
        lambda i, n: slide_data(prs, i, n, info),
        lambda i, n: slide_features(prs, i, n),
        lambda i, n: slide_network(prs, i, n),
        lambda i, n: slide_training(prs, i, n, info),
        lambda i, n: slide_decision(prs, i, n),
        lambda i, n: slide_demo(prs, i, n),
        lambda i, n: slide_examples(prs, i, n),
        lambda i, n: slide_quality(prs, i, n, metrics),
        lambda i, n: slide_real(prs, i, n, real),
        lambda i, n: slide_limits(prs, i, n),
        lambda i, n: slide_experiments(prs, i, n),
        lambda i, n: slide_final(prs, i, n),
    ]
    total = len(builders)
    for index, build in enumerate(builders, start=1):
        build(index, total)

    # Плоский стиль: убираем тени у всех фигур (PowerPoint добавляет их по умолчанию).
    for slide in prs.slides:
        for shape in slide.shapes:
            try:
                shape.shadow.inherit = False
            except (AttributeError, ValueError, NotImplementedError):
                pass

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    print(f"Готово: {out}")
    print(f"Слайдов: {len(prs.slides)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
