import csv
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from wind_power_predictor.config import SETTINGS
from wind_power_predictor.features.split import TrainingSplit
from wind_power_predictor.reporting.paths import PredictionFiles


class PredictionReport:
    def __init__(self, plant, output_path):
        self._files = PredictionFiles(plant, Path(output_path))

    def save(self, split: TrainingSplit, prediction, name, history):
        result_path = self._files.prediction_csv(name, history)
        title = result_path.stem
        # 모든 모델의 예측값과 평가 지표를 원래 발전량 단위로 비교합니다.
        predicted = split.original_target(prediction)
        actual = split.original_target(split.test_y)
        if len(predicted) != len(actual) or len(actual) != len(split.test_dates):
            raise ValueError("Predictions, actual values and timestamps must have the same length")

        mse = mean_squared_error(actual, predicted)
        scores = {
            "R2": r2_score(actual, predicted),
            "MSE": mse,
            "RMSE": mse ** 0.5,
            "MAE": mean_absolute_error(actual, predicted),
        }
        print(title)
        for label, value in scores.items():
            print(f"{label}: {value}")
        print()

        output_dir = result_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        # 지표와 그래프가 준비될 때까지 기존 예측 결과를 보존합니다.
        with TemporaryDirectory(dir=output_dir) as directory:
            staged = Path(directory)
            pd.DataFrame({
                "tgt_datetime": split.test_dates,
                "predict_value": predicted,
                "actual_value": actual,
            }).to_csv(staged / f"{title}.csv", index=False)

            fig, ax = plt.subplots(figsize=SETTINGS["reporting"]["figure_size"])
            try:
                ax.set_title(title, fontsize=16)
                ax.set_xlabel("Time", fontsize=16)
                ax.set_ylabel("Power generation [kW]", fontsize=16)
                ax.plot(actual, marker=",", label="True Value")
                ax.plot(predicted, ls="--", marker=",", label="Predict Value")
                ax.legend(loc="upper left")
                fig.savefig(staged / f"{title}.png")
            finally:
                plt.close(fig)

            self._publish(staged, output_dir, title, scores)

    def _publish(self, staged, output_dir, title, scores):
        summary = self._files.output_path / "model_performance.csv"
        # 지표 CSV를 교체하는 동안에도 같은 파일로 잠금을 유지합니다.
        with (summary.parent / ".model_performance.lock").open("a+b") as lock:
            if os.name == "nt":
                import msvcrt

                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

            rows = []
            if summary.exists():
                with summary.open(newline="") as file:
                    rows = [row for row in csv.reader(file) if not row or row[0] != title]

            staged_summary = staged / summary.name
            with staged_summary.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(rows)
                writer.writerow([title, *scores.values()])

            destinations = [output_dir / f"{title}.csv", output_dir / f"{title}.png", summary]
            previous = staged / "previous"
            previous.mkdir()
            for destination in destinations:
                if destination.exists():
                    shutil.copyfile(destination, previous / destination.name)

            published = []
            try:
                for destination in destinations:
                    (staged / destination.name).replace(destination)
                    published.append(destination)
            except OSError:
                # 파일 교체가 중간에 실패하면 이미 교체한 결과를 되돌립니다.
                for destination in reversed(published):
                    backup = previous / destination.name
                    if backup.exists():
                        backup.replace(destination)
                    else:
                        destination.unlink()
                raise

    def save_coefficients(self, columns, values, history):
        title = self._files.title("ElasticNet", history)
        self._save_table(pd.DataFrame(values, index=columns), "elasticnet", f"{title}_coefficient.csv")

    def save_importance(self, columns, values, name, history, folder):
        title = self._files.title(name, history)
        self._save_table(pd.DataFrame({"Importance": values}, index=columns), folder, f"{title}.csv")

    def _save_table(self, table, folder, filename):
        directory = self._files.output_path / folder
        directory.mkdir(parents=True, exist_ok=True)
        table.to_csv(directory / filename)
