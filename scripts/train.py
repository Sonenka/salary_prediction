import logging
from pathlib import Path

from core.trainer import RidgeTrainer


def setup_logging(verbose: bool) -> None:
    """
    Настроить систему логирования приложения.

    Аргументы:
        verbose (bool): Флаг подробного логирования. Если True, устанавливается уровень INFO,
                        иначе WARNING.
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


def main(verbose: bool = False) -> None:
    """
    Точка входа для обучения Ridge-регрессии.

    Выполняет:
    - инициализацию трейнера,
    - обучение модели с подбором гиперпараметров,
    - сохранение модели,
    - логирование метрик обучения.

    Аргументы:
        verbose (bool): Если True, включается подробное логирование.

    Исключения:
        FileNotFoundError: Если отсутствуют файлы с данными.
        ValueError: Если данные некорректны (несоответствие размеров X и y и т.д.)
        TypeError: Если X или y имеют неверный тип.
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    logger.info("Training started")

    trainer = RidgeTrainer(
        x_path=Path("x_data.npy"),
        y_path=Path("y_data.npy"),
        model_path=Path("resources/ridge_model.joblib"),
    )

    metrics = trainer.run()

    logger.info("Training completed")
    for k, v in metrics.items():
        logger.info("%-12s: %.4f", k, v)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Ridge regression model")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()
    main(verbose=args.verbose)