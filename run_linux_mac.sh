#!/bin/sh
# Запуск интерфейса на Linux и macOS:  sh run_linux_mac.sh
cd "$(dirname "$0")" || exit 1

# Если окружение осталось от другой версии Python или библиотеки не встали — пересоздаём.
if [ -x .venv/bin/python ] && ! .venv/bin/python -c "import numpy, scipy, PIL, fastapi, uvicorn" >/dev/null 2>&1; then
  echo "Окружение .venv повреждено или неполное — пересоздаю его..."
  rm -rf .venv
fi

if [ ! -d .venv ]; then
  echo "Первый запуск: создаю окружение и ставлю библиотеки. Это займёт несколько минут..."
  python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
else
  . .venv/bin/activate
fi
python app.py
