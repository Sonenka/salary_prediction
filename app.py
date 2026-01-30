import argparse
import logging
from pathlib import Path
from typing import List

from core.io import load_npy, load_model
from core.validator import validate_x


MODEL_PATH = Path("resources/ridge_model.joblib")


def setup_logging(verbose: bool) -> None:
    """
    Настроить систему логирования приложения.

    При включённом verbose-режиме устанавливает уровень INFO,
    иначе используется уровень WARNING.

    Аргументы:
        verbose (bool): Флаг подробного логирования.
    """
    level = logging.INFO if verbose else logging.WARNING

    logging.getLogger().setLevel(level)

    if not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)


def parse_args() -> argparse.Namespace:
    """
    Распарсить аргументы командной строки.

    Возвращает:
        argparse.Namespace: Объект с аргументами:
            - x_path (Path): путь к файлу с признаками (.npy)
            - verbose (bool): флаг подробного логирования
    """
    parser = argparse.ArgumentParser(
        description="Predict salaries using Ridge regression model"
    )
    parser.add_argument(
        "x_path",
        type=Path,
        help="Path to .npy file with input features",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> List[float]:
    """
    Точка входа в приложение.

    Загружает данные, валидирует входные признаки,
    выполняет предсказание зарплат и возвращает результат.

    Возвращает:
        List[float]: Список предсказанных зарплат.
    """
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("Application started")

    logger.info("Loading input data")
    X = load_npy(args.x_path)
    validate_x(X)

    logger.info("Loading model")
    model = load_model(MODEL_PATH)

    logger.info("Running prediction")
    predictions = model.predict(X)

    salaries = predictions.astype(float).tolist()
    logger.info("Prediction finished successfully")

    return salaries


if __name__ == "__main__":
    main()