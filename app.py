import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from core.io import load_npy, load_model
from core.validator import validate_x


MODEL_PATH = Path("resources/ridge_model.joblib")

Y_PATH = Path("y_data.npy")  # для логирования, только при verbose


def setup_logging(verbose: bool) -> None:
    """
    Настроить систему логирования приложения.

    Аргументы:
        verbose (bool): Флаг подробного логирования. Если True, устанавливается уровень INFO,
                        иначе уровень WARNING.
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

    Аргументы:
        Нет

    Возвращает:
        argparse.Namespace: Объект с аргументами командной строки:
            - x_path (Path): путь к файлу с признаками (.npy)
            - verbose (bool): флаг подробного логирования

    Исключения:
        SystemExit: Если аргументы командной строки некорректны.
    """
    parser = argparse.ArgumentParser(
        description="Predict salaries using Ridge regression model"
    )
    parser.add_argument("x_path", type=Path, help="Path to .npy file with input features")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    return parser.parse_args()


def main() -> List[float]:
    """
    Выполнить предсказание зарплат на основе модели Ridge-регрессии.

    Загружает данные из .npy файла, валидирует их, выполняет предсказание зарплат,
    а при включённом verbose выводит сравнение с исходными значениями из y_data.npy.

    Аргументы:
        Нет, использует sys.argv для парсинга входных аргументов.

    Возвращает:
        List[float]: Список предсказанных зарплат в рублях.

    Исключения:
        FileNotFoundError: Если не найден файл признаков или модель.
        TypeError: Если загруженные данные X не являются numpy-массивом.
        ValueError: Если X не двумерный или пустой.
    """
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("Application started")

    X = load_npy(args.x_path)
    validate_x(X)
    logger.info("Loaded data shape: %s", X.shape)

    model = load_model(MODEL_PATH)

    predictions = model.predict(X)
    salaries = predictions.astype(float)

    if args.verbose and Y_PATH.exists():
        y_true = load_npy(Y_PATH).ravel()
        diffs = salaries - y_true
        logger.info("First 10 predictions vs true values:")
        for i in range(min(10, len(salaries))):
            logger.info(
                "pred: %.2f | true: %.2f | diff: %.2f",
                salaries[i],
                y_true[i],
                diffs[i],
            )
        logger.info(
            "Salary diff stats | min: %.2f | max: %.2f | mean: %.2f | std: %.2f",
            float(np.min(diffs)),
            float(np.max(diffs)),
            float(np.mean(diffs)),
            float(np.std(diffs)),
        )

    return salaries


if __name__ == "__main__":
    main()