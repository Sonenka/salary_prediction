from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(alpha: float = 1.0) -> Pipeline:
    """
    Создать sklearn Pipeline для Ridge-регрессии
    с предварительной стандартизацией признаков.

    Аргументы:
        alpha (float, optional): Коэффициент регуляризации Ridge.
            По умолчанию 1.0.

    Возвращает:
        Pipeline: Конвейер из StandardScaler и Ridge-регрессора.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=alpha)),
        ]
    )
