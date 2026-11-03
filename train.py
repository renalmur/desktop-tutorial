"""Обучение нейросети.  Запуск:  python train.py

Скрипт рисует учебные снимки, добавляет ваши размеченные настоящие снимки
(если они есть в data/labels или data/masks), подбирает веса сети и сохраняет
их в models/cloud_net.npz. После обучения запустите проверку: python evaluate.py
"""
import argparse
from pathlib import Path

from cloudai.training import SYNTH_CACHE, train_and_save

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение нейросети для поиска облаков")
    parser.add_argument("--scenes", type=int, default=600, help="сколько учебных снимков нарисовать (по умолчанию 600)")
    parser.add_argument("--seed", type=int, default=0, help="число, от которого зависит случайность")
    parser.add_argument("--snow-share", type=float, default=0.0,
                        help="доля снежных снимков в обучении, от 0 до 1 (по умолчанию 0 — см. GUIDE_RU.md, опыт 3)")
    parser.add_argument("--fresh", action="store_true", help="нарисовать учебные снимки заново, не брать сохранённые")
    parser.add_argument("--out", default=None, help="куда сохранить модель (по умолчанию models/cloud_net.npz)")
    args = parser.parse_args()
    if args.fresh and SYNTH_CACHE.exists():
        SYNTH_CACHE.unlink()
    extra = {"model_path": Path(args.out)} if args.out else {}
    train_and_save(n_scenes=args.scenes, seed=args.seed, snow_share=args.snow_share, **extra)
