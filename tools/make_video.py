"""Запись короткого демонстрационного видео и сжатие полной записи.

Что делает:
    1) готовит демонстрационный набор (8 снимков, 2 из них удаляются);
    2) записывает интерфейс в 1280×720 с подписями;
    3) сжимает готовое видео в H.264 для GitHub;
    4) если в корне лежит полная запись 20261103-*.mp4 — сжимает и её.

Запуск (сервер уже работает на http://127.0.0.1:8000):
    python tools/make_video.py

Результат:
    video/demo.mp4        — короткая демонстрация с удалением
    video/full_demo.mp4   — сжатая полная запись (если была)
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "video"
FULL_SOURCE = ROOT / "20261103-1422-11.3759514.mp4"

TITLE = """
<div style="position:fixed;inset:0;z-index:99999;background:#111;color:#fff;
            display:flex;flex-direction:column;justify-content:center;padding:90px;
            font-family:Arial,Helvetica,sans-serif">
  <div style="color:#FFD400;font-size:22px;font-weight:700;letter-spacing:.08em">
    КЕЙС «СПУТНИКС» · 14+
  </div>
  <div style="font-size:74px;font-weight:900;line-height:1.05;margin-top:26px">
    ОБЛАЧНЫЙ<br>КОНТРОЛЬ
  </div>
  <div style="font-size:28px;color:#CFCFCF;margin-top:30px;max-width:900px">
    ИИ считает процент облачности<br>и удаляет снимки, где облаков больше 75 %
  </div>
</div>
"""

END = """
<div style="position:fixed;inset:0;z-index:99999;background:#111;color:#fff;
            display:flex;flex-direction:column;justify-content:center;align-items:center;
            font-family:Arial,Helvetica,sans-serif;text-align:center">
  <div style="color:#FFD400;font-size:26px;font-weight:700;letter-spacing:.08em">
    РЕЗУЛЬТАТ КЕЙСА
  </div>
  <div style="font-size:52px;font-weight:900;margin-top:20px">
    8 обработано · 6 сохранено · 2 удалено
  </div>
  <div style="font-size:24px;color:#CFCFCF;margin-top:20px">
    Имена файлов и проценты — в сообщении оператору
  </div>
</div>
"""

CAPTION_JS = """
() => {
  let el = document.getElementById('demo-caption');
  if (!el) {
    el = document.createElement('div');
    el.id = 'demo-caption';
    el.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:99998;' +
      'background:#111;color:#FFD400;font:700 20px Arial,Helvetica,sans-serif;' +
      'padding:14px 26px;border-top:4px solid #FFD400;';
    document.body.appendChild(el);
  }
  window.__setCaption = (text) => { el.textContent = text; };
}
"""


def set_caption(page, text):
    page.evaluate("(t) => window.__setCaption(t)", text)


def overlay(page, html):
    page.evaluate("(h) => { document.body.style.margin = '0'; document.body.innerHTML = h; }", html)


def record(url: str, work: Path) -> Path:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            record_video_dir=str(work),
            record_video_size={"width": 1280, "height": 720},
        )
        page = context.new_page()
        page.goto(url, wait_until="networkidle")

        overlay(page, TITLE)
        page.wait_for_timeout(3000)

        page.goto(url, wait_until="networkidle")
        page.evaluate(CAPTION_JS)
        page.locator("#run-folder").wait_for(state="visible", timeout=15000)

        set_caption(page, "В папке 8 снимков: 3 настоящих спутниковых и 5 учебных.")
        page.wait_for_timeout(2500)

        set_caption(page, "Нажимаем «Обработать папку».")
        page.wait_for_timeout(1200)
        page.locator("#run-folder").click()
        page.locator("#report").wait_for(state="visible", timeout=180000)
        set_caption(page, "Обработано 8: 6 сохранено, 2 удалено. Это заняло несколько секунд.")
        page.wait_for_timeout(2800)

        page.locator("#message").scroll_into_view_if_needed()
        page.evaluate("""() => {
            const el = document.getElementById('message');
            const bottom = el.getBoundingClientRect().bottom + window.scrollY;
            window.scrollTo(0, Math.max(0, bottom - window.innerHeight + 120));
        }""")
        set_caption(page, "В сообщении — имена сохранённых файлов, проценты и имена удалённых.")
        page.wait_for_timeout(3200)

        page.locator("#rows").scroll_into_view_if_needed()
        set_caption(page, "Слева снимок, справа маска: малиновым закрашены облака.")
        page.wait_for_timeout(3200)

        page.locator("#rows li").last.scroll_into_view_if_needed()
        set_caption(page, "demo_04 и demo_05 удалены: 81,7 % и 97,6 % облачности.")
        page.wait_for_timeout(3000)

        page.get_by_role("tab", name="Разметка и дообучение").click()
        page.wait_for_timeout(800)
        set_caption(page, "Вкладка разметки: ошибки сети можно исправить кистью и дообучить.")
        page.wait_for_timeout(3000)

        overlay(page, END)
        page.wait_for_timeout(3200)

        context.close()
        browser.close()

    videos = sorted(work.glob("*.webm"), key=lambda p: p.stat().st_mtime)
    if not videos:
        raise RuntimeError("Playwright не сохранил видео")
    return videos[-1]


def encode(source: Path, target: Path, height: int = 720, crf: int = 24, audio: bool = False) -> None:
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ff, "-y", "-i", str(source), "-vf", f"scale=-2:{height},fps=30",
               "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    command += ["-c:a", "aac", "-b:a", "96k"] if audio else ["-an"]
    command.append(str(target))
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Запись демонстрационного видео")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--skip-record", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_demo.py"), "--prepare"], check=True)

    if not args.skip_record:
        with tempfile.TemporaryDirectory() as tmp:
            raw = record(args.url, Path(tmp))
            print(f"Записано: {raw.name} ({raw.stat().st_size / 1e6:.1f} МБ)")
            encode(raw, OUT / "demo.mp4", height=720, crf=24, audio=False)
            print(f"Сжато: {OUT / 'demo.mp4'} ({ (OUT / 'demo.mp4').stat().st_size / 1e6:.1f} МБ)")

    if FULL_SOURCE.exists() and not (OUT / "full_demo.mp4").exists():
        encode(FULL_SOURCE, OUT / "full_demo.mp4", height=720, crf=28, audio=True)
        print(f"Полная запись сжата: {OUT / 'full_demo.mp4'} "
              f"({ (OUT / 'full_demo.mp4').stat().st_size / 1e6:.1f} МБ)")

    # Возвращаем демонстрационный набор в исходное состояние.
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_demo.py"), "--prepare"], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
