"""Скриншоты интерфейса для презентации.

Запуск (сервер уже работает на http://127.0.0.1:8000):
    python tools/make_screenshots.py

Скрипт сам готовит демонстрационный набор, открывает страницу в headless-браузере,
нажимает «Обработать папку» и сохраняет:
    reports/presentation/ui_report.png    — вся страница с результатами
    reports/presentation/ui_message.png   — только сообщение оператору
    reports/presentation/ui_label.png     — вкладка разметки (если есть снимки)

Нужен установленный Microsoft Edge (на Windows он есть по умолчанию).
Playwright — только для скриншотов:  pip install playwright
"""
import argparse
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "presentation"


def main() -> int:
    parser = argparse.ArgumentParser(description="Скриншоты интерфейса")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--no-prepare", action="store_true", help="не пересоздавать демонстрационный набор")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if not args.no_prepare:
        subprocess.run([sys.executable, str(ROOT / "tools" / "make_demo.py"), "--prepare"], check=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
        page.goto(args.url, wait_until="networkidle")
        page.locator("#run-folder").wait_for(state="visible", timeout=20000)
        page.locator("#run-folder").click()
        page.locator("#report").wait_for(state="visible", timeout=120000)
        page.wait_for_timeout(600)
        page.screenshot(path=str(OUT / "ui_report.png"), full_page=True)
        page.locator("#message").screenshot(path=str(OUT / "ui_message.png"))

        # Вкладка разметки: показываем снимок и предсказание сети.
        page.get_by_role("tab", name="Разметка и дообучение").click()
        page.wait_for_timeout(800)
        buttons = page.locator("#image-list li button")
        if buttons.count():
            buttons.first.click()
            page.locator("#label-work").wait_for(state="visible", timeout=30000)
            page.locator("#show-prediction").check()
            page.wait_for_timeout(1200)
            page.screenshot(path=str(OUT / "ui_label.png"), full_page=True)

        browser.close()

    print("Готово:")
    for name in ("ui_report.png", "ui_message.png", "ui_label.png"):
        path = OUT / name
        if path.exists():
            print(f"  {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
