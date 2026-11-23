from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from wind_power_predictor.config import SETTINGS


_SCALE_RANGE = tuple(SETTINGS["features"]["scale_range"])


@dataclass(frozen=True)
class TrainingSplit:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    test_dates: pd.Series
    feature_names: pd.Index
    target_scaler: MinMaxScaler | None = None

    def scaled(self):
        # 이미 정규화한 분할은 기존 스케일러를 보존해야 kW로 복원할 수 있습니다.
        if self.target_scaler is not None:
            return self

        # 평가 자료는 학습 자료에서 정한 최소값과 최대값으로 변환합니다.
        feature_scaler = MinMaxScaler(feature_range=_SCALE_RANGE)
        target_scaler = MinMaxScaler(feature_range=_SCALE_RANGE)
        return replace(
            self,
            train_x=feature_scaler.fit_transform(self.train_x),
            train_y=target_scaler.fit_transform(self.train_y),
            test_x=feature_scaler.transform(self.test_x),
            test_y=target_scaler.transform(self.test_y),
            target_scaler=target_scaler,
        )

    def original_target(self, values):
        values = np.asarray(values).reshape(-1, 1)
        if self.target_scaler is not None:
            values = self.target_scaler.inverse_transform(values)
        return values.ravel()
