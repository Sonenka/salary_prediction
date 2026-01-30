import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score

from core.io import load_npy, save_model
from core.model import build_pipeline
from core.validator import validate_x


logger = logging.getLogger(__name__)


@dataclass
class RidgeTrainer:
    """Класс для обучения Ridge-регрессии с подбором гиперпараметров."""

    x_path: Path
    y_path: Path
    model_path: Path
    cv_splits: int = 5
    random_state: int = 42
    n_jobs: int = -1

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Загрузить и валидировать входные данные X и целевую переменную y.

        Возвращает:
            tuple[np.ndarray, np.ndarray]: Кортеж из массива признаков X
            и массива целевых значений y.

        Исключения:
            FileNotFoundError: Если файлы X или y не найдены.
            TypeError: Если X не является numpy-массивом.
            ValueError:
                - Если X имеет неверную размерность
                - Если X пустой
                - Если длины X и y не совпадают
        """
        X = load_npy(self.x_path)
        y = load_npy(self.y_path).ravel()

        validate_x(X)

        if len(X) != len(y):
            raise ValueError("X and y have different lengths")

        return X, y

    def run(self) -> Dict[str, float]:
        """
        Запустить процесс обучения модели:
        - загрузить данные
        - подобрать оптимальный alpha для Ridge-регрессии
        - обучить лучшую модель
        - сохранить модель на диск
        - вычислить метрики качества

        Возвращает:
            Dict[str, Any]: Словарь с результатами обучения:
                - best_alpha (float): лучшее значение коэффициента регуляризации
                - rmse (float): корень из среднеквадратичной ошибки
                - r2 (float): коэффициент детерминации R²
        """
        logger.info("Starting training")

        X, y = self.load_data()
        logger.info("Data loaded: X=%s, y=%s", X.shape, y.shape)

        pipeline = build_pipeline()

        param_grid = {
            "regressor__alpha": np.arange(0.1, 15.0, 0.25)
        }

        cv = KFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        logger.info("Running GridSearchCV")
        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=cv,
            n_jobs=self.n_jobs,
        )

        grid.fit(X, y)

        logger.info(
            "Best alpha found: %.3f",
            grid.best_params_["regressor__alpha"],
        )

        best_model = grid.best_estimator_
        save_model(best_model, self.model_path)

        preds = best_model.predict(X)

        logger.info("Training finished successfully")

        return {
            "best_alpha": grid.best_params_["regressor__alpha"],
            "rmse": mean_squared_error(y, preds) ** 0.5,
            "r2": r2_score(y, preds),
        }
