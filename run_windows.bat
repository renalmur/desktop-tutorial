@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem Если папка .venv осталась от старой версии Python или библиотеки не встали,
rem окружение пересоздаётся: иначе python.exe внутри .venv просто не запустится.
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import numpy, scipy, PIL, fastapi, uvicorn" >nul 2>&1
  if errorlevel 1 (
    echo Окружение .venv повреждено или неполное — пересоздаю его...
    rmdir /s /q ".venv"
  )
)

if not exist ".venv" (
  echo Первый запуск: создаю окружение и ставлю библиотеки. Это займёт несколько минут...
  python -m venv .venv || py -3 -m venv .venv
  call ".venv\Scripts\activate.bat"
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call ".venv\Scripts\activate.bat"
)

python app.py
pause
