import logging
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)


def load_npy(path: Path) -> np.ndarray:
    """
    Загрузить данные из файла формата .npy.

    Аргументы:
        path (Path): Путь к файлу .npy.

    Возвращает:
        np.ndarray: Массив, загруженный из файла.

    Исключения:
        FileNotFoundError: Если файл по указанному пути не существует.
    """
    if not path.exists():
        logger.error("File not found: %s", path)
        raise FileNotFoundError(f"File not found: {path}")
    
    logger.info("Loading npy file: %s", path)
    return np.load(path, allow_pickle=True)


def save_model(model, path: Path) -> None:
    """
    Сохранить модель в файл с использованием joblib.
    При необходимости создать родительские директории.

    Аргументы:
        model: Обученная модель.
        path (Path): Путь для сохранения модели.
    """
    logger.info("Saving model to %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    """
    Загрузить сохранённую модель из файла.

    Аргументы:
        path (Path): Путь к файлу с сохранённой моделью.

    Возвращает:
        Объект модели, загруженный из файла.

    Исключения:
        FileNotFoundError: Если файл с моделью не найден.
    """
    if not path.exists():
        logger.error("Model not found: %s", path)
        raise FileNotFoundError(
            "Model weights not found. Train the model first."
        )
    
    logger.info("Loading model from %s", path)
    return joblib.load(path)
