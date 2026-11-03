"""Проверка качества на НАСТОЯЩИХ снимках СПУТНИКС.

В отличие от evaluate.py (синтетика), здесь модель сравнивается с оценками
человека по реальным снимкам: процент облачности, ошибка в процентных пунктах
и — главное для кейса — правильность решения «сохранить / удалить» при 75 %.

    python tools/evaluate_real.py                     # текущая модель
    python tools/evaluate_real.py --model models/cloud_net_base.npz --tag base

Отчёт: reports/real_quality_report.html (картинки и таблица), рядом CSV.
"""
import argparse
import base64
import csv
import html
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cloudai.imageio import ImageReadError, load_image       # noqa: E402
from cloudai.net import CloudNet                             # noqa: E402

THRESHOLD = 75.0


def thumbnail(image: dict, width: int = 320) -> str:
    rgb = image["rgb"]
    height = max(1, round(rgb.shape[0] * width / rgb.shape[1]))
    pic = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((width, height), Image.LANCZOS)
    buf = io.BytesIO()
    pic.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def measure(rows: list[dict]) -> dict:
    if not rows:
        return {}
    errors = np.array([r["model"] - r["expert_percent"] for r in rows], dtype=float)
    false_delete = sum(1 for r in rows if r["model"] > THRESHOLD and r["expert_percent"] <= THRESHOLD)
    missed = sum(1 for r in rows if r["model"] <= THRESHOLD and r["expert_percent"] > THRESHOLD)
    correct = len(rows) - false_delete - missed
    return {
        "n": len(rows),
        "mae": float(np.abs(errors).mean()),
        "bias": float(errors.mean()),
        "rmse": float(np.sqrt((errors ** 2).mean())),
        "decision_accuracy": 100.0 * correct / len(rows),
        "false_delete": false_delete,
        "missed": missed,
        "correct": correct,
        "should_delete": sum(1 for r in rows if r["expert_percent"] > THRESHOLD),
        "should_keep": sum(1 for r in rows if r["expert_percent"] <= THRESHOLD),
    }


def scatter_svg(rows: list[dict], size: int = 460) -> str:
    pad = 46
    plot = size - 2 * pad
    points = []
    for r in rows:
        x = pad + plot * r["expert_percent"] / 100
        y = pad + plot * (1 - r["model"] / 100)
        color = "#d64578" if r["expert_percent"] > THRESHOLD or r["model"] > THRESHOLD else "#2f7fd1"
        points.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" opacity=".8"/>')
    grid = []
    for value in range(0, 101, 25):
        x = pad + plot * value / 100
        y = pad + plot * (1 - value / 100)
        grid.append(f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{pad + plot}" stroke="#e3e7ee"/>')
        grid.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{pad + plot}" y2="{y:.1f}" stroke="#e3e7ee"/>')
        grid.append(f'<text x="{x:.1f}" y="{pad + plot + 20}" font-size="12" fill="#7b8698" text-anchor="middle">{value}</text>')
        grid.append(f'<text x="{pad - 10}" y="{y + 4:.1f}" font-size="12" fill="#7b8698" text-anchor="end">{value}</text>')
    return f'''<svg viewBox="0 0 {size} {size}" width="100%" style="max-width:{size}px">
  <rect x="{pad}" y="{pad}" width="{plot}" height="{plot}" fill="#fbfcfe" stroke="#cfd6e2"/>
  {''.join(grid)}
  <line x1="{pad}" y1="{pad + plot}" x2="{pad + plot}" y2="{pad}" stroke="#9aa6b8" stroke-dasharray="5 4"/>
  {''.join(points)}
  <text x="{size / 2}" y="{size - 8}" font-size="13" fill="#5b6575" text-anchor="middle">оценка человека, % облачности</text>
  <text x="14" y="{size / 2}" font-size="13" fill="#5b6575" text-anchor="middle" transform="rotate(-90 14 {size / 2})">модель, %</text>
</svg>'''


def metric_card(title: str, m: dict, accent: str = "") -> str:
    if not m:
        return ""
    cls = f"card {accent}".strip()
    return f'''<div class="{cls}">
  <h3>{html.escape(title)}</h3>
  <p class="big">{m['mae']:.1f} <span>п.п. MAE</span></p>
  <p>снимков: {m['n']} · смещение: {m['bias']:+.1f} п.п. · RMSE: {m['rmse']:.1f}</p>
  <p>верных решений: <b>{m['decision_accuracy']:.0f} %</b> ({m['correct']} из {m['n']})</p>
</div>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--images", default="data/real_extra")
    parser.add_argument("--estimates", default="data/real_eval/expert_estimates.json")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else ROOT / "models" / "cloud_net.npz"
    images_dir = ROOT / args.images
    estimates_path = ROOT / args.estimates
    if not estimates_path.exists():
        print(f"Нет файла оценок {estimates_path}. Сначала создайте его (см. tools/make_expert_estimates.py).")
        return 1
    estimates = json.loads(estimates_path.read_text(encoding="utf-8"))
    model = CloudNet.load(model_path)
    train_images = set(model.info.get("real_images", []))       # эти снимки сеть видела при обучении
    print(f"Модель: {model_path.name} ({model.info.get('trained_at', '—')})")
    print(f"Снимков в наборе: {len(estimates)}")
    if train_images:
        print(f"Обучающих снимков в наборе: {sum(1 for e in estimates if e['file'] in train_images)} "
              f"(исключаются из метрик, чтобы не было утечки)")

    rows = []
    for i, item in enumerate(estimates, 1):
        path = images_dir / item["file"]
        if not path.exists():
            print(f"  пропуск (нет файла): {item['file']}")
            continue
        try:
            image = load_image(path)
        except ImageReadError as exc:
            print(f"  пропуск (не читается): {item['file']} — {exc}")
            continue
        percent = model.analyze(image["rgb"], image["alpha"])["cloud_percent"]
        row = {
            **item,
            "model": round(percent, 1),
            "error": round(percent - item["expert_percent"], 1),
            "train": item["file"] in train_images,
            "thumb": thumbnail(image),
        }
        rows.append(row)
        print(f"  [{i:02d}/{len(estimates)}] {item['file']}: модель {percent:5.1f} % · человек {item['expert_percent']:5.1f} % · ошибка {row['error']:+5.1f}")

    used = [r for r in rows if not r.get("exclude") and not r.get("train")]
    high = [r for r in used if r.get("confidence", "high") == "high"]
    low = [r for r in used if r.get("confidence") == "low"]
    all_m, high_m, low_m = measure(used), measure(high), measure(low)

    print("\nИТОГО")
    for title, m in (("все снимки", all_m), ("уверенные оценки", high_m), ("сложные (снег/дымка)", low_m)):
        if m:
            print(f"  {title:22s}: MAE {m['mae']:.1f} п.п., смещение {m['bias']:+.1f}, "
                  f"решения {m['decision_accuracy']:.0f} %, зря удалено {m['false_delete']}, пропущено {m['missed']}")
    if all_m and all_m["should_delete"] == 0:
        print("\n  ВНИМАНИЕ: в наборе нет снимков с облачностью выше 75 %.")
        print("  Значит, ветка «удалить облачный снимок» на реальных данных НЕ проверена:")
        print("  все решения здесь — только «сохранить». Добавьте облачные снимки (10–15 шт.).")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    table = []
    for r in sorted(rows, key=lambda x: (x.get("exclude", False) or x.get("train", False),
                                         x.get("confidence", "high"), -abs(x["error"]))):
        verdict = "удалить" if r["model"] > THRESHOLD else "сохранить"
        truth = "удалить" if r["expert_percent"] > THRESHOLD else "сохранить"
        bad = "" if verdict == truth else " bad"
        note = (" · обучающий снимок (не в метриках)" if r.get("train") else
                " · исключён" if r.get("exclude") else
                " · низкая уверенность" if r.get("confidence") == "low" else "")
        table.append(f'''<tr class="{ 'muted' if (r.get('exclude') or r.get('train')) else '' }">
  <td><img src="{r['thumb']}" alt=""></td>
  <td class="name">{html.escape(r['file'])}<div class="note">оценка человека: {r['confidence']}{note}</div></td>
  <td class="num">{r['expert_percent']:.0f}</td>
  <td class="num">{r['model']:.1f}</td>
  <td class="num{ ' err' if abs(r['error']) > 15 else '' }">{r['error']:+.1f}</td>
  <td><span class="pill{bad}">{verdict}</span></td>
</tr>''')

    warning = ""
    if all_m and all_m["should_delete"] == 0:
        warning = '''<div class="panel warn-panel">
  <b>Ветка удаления на реальных данных не проверена.</b>
  В наборе нет ни одного снимка с облачностью выше 75 %, поэтому все решения здесь — «сохранить».
  Чтобы проверить главное требование кейса, добавьте 10–15 реальных облачных снимков (или отметьте их в экспертных оценках).
</div>'''

    report = f'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Качество на реальных снимках СПУТНИКС</title>
<style>
  :root {{ --ink:#1d2433; --muted:#6b7688; --line:#e3e7ee; --accent:#2f7fd1; --bad:#c0392b; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:32px; font:15px/1.5 "Segoe UI", Arial, sans-serif; color:var(--ink); background:#f4f6fa; }}
  .wrap {{ max-width:1180px; margin:0 auto; }}
  h1 {{ font-size:28px; margin:0 0 4px; }}
  .sub {{ color:var(--muted); margin:0 0 24px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px; margin:0 0 24px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:18px 20px; }}
  .card h3 {{ margin:0 0 8px; font-size:15px; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.04em; }}
  .card .big {{ font-size:30px; font-weight:700; margin:0 0 6px; }}
  .card .big span {{ font-size:14px; font-weight:400; color:var(--muted); }}
  .card p {{ margin:4px 0; color:var(--muted); font-size:13.5px; }}
  .card.good {{ border-left:5px solid #2e9e5b; }}
  .card.warn {{ border-left:5px solid #d39b1c; }}
  .warn-panel {{ border-left:5px solid #d39b1c; background:#fffaf0; }}
  .panel {{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:20px; margin-bottom:24px; }}
  .grid {{ display:grid; grid-template-columns:minmax(280px,460px) 1fr; gap:28px; align-items:center; }}
  .legend {{ font-size:13.5px; color:var(--muted); }}
  .legend b {{ color:var(--ink); }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:14px; overflow:hidden; }}
  th {{ text-align:left; font-size:12.5px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em;
        padding:12px 14px; border-bottom:1px solid var(--line); background:#fafbfd; }}
  td {{ padding:10px 14px; border-bottom:1px solid var(--line); vertical-align:middle; }}
  td img {{ width:150px; border-radius:8px; display:block; }}
  td.name {{ font-size:13.5px; max-width:340px; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.num.err {{ color:var(--bad); font-weight:600; }}
  .note {{ color:var(--muted); font-size:12px; margin-top:3px; }}
  tr.muted {{ opacity:.5; }}
  .pill {{ padding:3px 10px; border-radius:999px; background:#e8f4ec; color:#1c7c43; font-size:12.5px; white-space:nowrap; }}
  .pill.bad {{ background:#fbe9e7; color:var(--bad); }}
  .foot {{ color:var(--muted); font-size:13px; margin-top:16px; }}
  @media (max-width:800px) {{ .grid {{ grid-template-columns:1fr; }} td img {{ width:100px; }} }}
</style></head><body><div class="wrap">
<h1>Качество на реальных снимках СПУТНИКС</h1>
<p class="sub">Модель: {html.escape(model_path.name)} · обучена {html.escape(str(model.info.get('trained_at', '—')))} · отчёт {stamp}</p>

<div class="cards">
  {metric_card("Все снимки", all_m, "good")}
  {metric_card("Уверенные оценки", high_m)}
  {metric_card("Сложные: снег, дымка, край Земли", low_m, "warn")}
</div>

{warning}

<div class="panel">
  <div class="grid">
    <div>{scatter_svg(used)}</div>
    <div class="legend">
      <p><b>Что означают цифры.</b> MAE — средняя ошибка процента облачности в процентных пунктах.
      Смещение показывает, завышает модель облачность (+) или занижает (−).
      «Верных решений» — сколько раз вывод «сохранить/удалить» при пороге 75 % совпал с оценкой человека.</p>
      <p>Точки у диагонали — модель совпала с человеком. Точки выше диагонали — модель завысила облачность,
      ниже — занизила. Красным отмечены снимки, где модель или человек перешли порог 75 %.</p>
      <p><b>Честно об оценке.</b> «Человек» здесь — визуальная оценка по снимку, а не попиксельная маска.
      У снега, льда и дымки граница «облако / не облако» размыта, поэтому такие кадры помечены как сложные
      и показаны отдельно. Снимки с бликом Солнца и ровным полем исключены из метрик, а снимки,
      на которых сеть обучалась, исключены из метрик, чтобы не завышать качество.</p>
    </div>
  </div>
</div>

<div class="panel">
  <table>
    <thead><tr><th>Снимок</th><th>Файл</th><th>Человек, %</th><th>Модель, %</th><th>Ошибка</th><th>Решение модели</th></tr></thead>
    <tbody>{''.join(table)}</tbody>
  </table>
  <p class="foot">Источник снимков: открытые данные СПУТНИКС (disk.sputnix.ru), некоммерческое образовательное использование.</p>
</div>
</div></body></html>'''

    out = ROOT / "reports" / (f"real_quality_report_{args.tag}.html" if args.tag else "real_quality_report.html")
    out.write_text(report, encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["файл", "человек, %", "модель, %", "ошибка, п.п.", "уверенность", "исключён"])
        for r in rows:
            writer.writerow([r["file"], r["expert_percent"], r["model"], r["error"],
                             r.get("confidence", "high"),
                             "да" if (r.get("exclude") or r.get("train")) else "нет"])
    print(f"\nОтчёт: {out.relative_to(ROOT)}")
    print(f"Таблица: {csv_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
