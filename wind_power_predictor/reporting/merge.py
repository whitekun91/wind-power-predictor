from pathlib import Path

import numpy as np
import pandas as pd

from wind_power_predictor.config import DEFAULT_OUTPUT_PATH, SETTINGS
from wind_power_predictor.reporting.paths import PredictionFiles


class PredictionMerger:
    def __init__(self, plant, history, output_path=DEFAULT_OUTPUT_PATH):
        self.history = history
        self._files = PredictionFiles(plant, Path(output_path))

    def merge(self, models):
        # MLR의 시각 순서와 실측값을 전체 모델의 비교 기준으로 사용합니다.
        names = list(dict.fromkeys(("MLR", *models)))
        result = self._load("MLR")
        for name in names[1:]:
            prediction = self._load(name)
            if len(prediction) != len(result) or not result.index.isin(prediction.index).all():
                raise ValueError(f"{name} and MLR must contain the same prediction timestamps")
            # 입력 CSV의 행 순서가 달라도 같은 시각의 예측값을 붙입니다.
            prediction = prediction.reindex(result.index)
            if not np.allclose(prediction["actual_value"], result["actual_value"], **SETTINGS["reporting"]["merge_tolerance"]):
                raise ValueError(f"{name} and MLR must contain the same actual values")
            result[f"{name}_predict"] = prediction[f"{name}_predict"]

        for name in names:
            result[f"{name}_residual"] = result[f"{name}_predict"] - result["actual_value"]
        return result.reset_index()

    def save(self, models):
        destination = self._files.merged_csv(self.history)
        self.merge(models).to_csv(destination, index=False)
        return destination

    def _load(self, name):
        path = self._files.prediction_csv(name, self.history)
        frame = pd.read_csv(path, usecols=["tgt_datetime", "actual_value", "predict_value"])
        frame["tgt_datetime"] = pd.to_datetime(frame["tgt_datetime"])
        if frame["tgt_datetime"].isna().any() or frame["tgt_datetime"].duplicated().any():
            raise ValueError(f"{name} contains missing or duplicate prediction timestamps")
        return frame.set_index("tgt_datetime")[["actual_value", "predict_value"]].rename(
            columns={"predict_value": f"{name}_predict"},
        )
