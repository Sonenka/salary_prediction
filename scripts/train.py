from pathlib import Path
from core.trainer import RidgeTrainer


def main() -> None:
    trainer = RidgeTrainer(
        x_path=Path("x_data.npy"),
        y_path=Path("y_data.npy"),
        model_path=Path("resources/ridge_model.joblib"),
    )

    metrics = trainer.run()

    print("\nTraining completed")
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.4f}")


if __name__ == "__main__":
    main()
