@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem Один клик перед показом: вернуть демонстрационные снимки и запустить интерфейс.
if not exist ".venv\Scripts\python.exe" (
  echo Сначала запустите run_windows.bat — он создаст окружение.
  pause
  exit /b 1
)

call ".venv\Scripts\activate.bat"
python tools\make_demo.py --prepare
python app.py
pause
