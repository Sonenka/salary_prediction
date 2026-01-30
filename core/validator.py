import logging
import numpy as np


logger = logging.getLogger(__name__)


def validate_x(X: np.ndarray) -> None:
    """
    Проверить корректность входного массива признаков X.

    Аргументы:
        X (np.ndarray): Массив признаков.

    Исключения:
        TypeError: Если X не является numpy-массивом.
        ValueError:
            - Если X не является двумерным массивом
            - Если X пустой
    """
    logger.info("Validating input X")
    
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")

    if X.ndim != 2:
        raise ValueError(
            f"X must be 2D array, got shape {X.shape}"
        )

    if X.shape[0] == 0:
        raise ValueError("X is empty")
