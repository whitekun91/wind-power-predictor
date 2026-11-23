from numbers import Integral

import pandas as pd

from wind_power_predictor.config import SETTINGS
from wind_power_predictor.features.split import TrainingSplit
from wind_power_predictor.preprocessing.time import FORECAST_LEAD_HOURS, SLOT_HOURS


_FEATURES = SETTINGS["features"]
HISTORY_COUNTS = tuple(_FEATURES["history_counts"])
if not HISTORY_COUNTS or any(type(count) is not int or count < 0 for count in HISTORY_COUNTS):
    raise ValueError("features.history_counts must contain nonnegative integers")


class TrainingDataset:
    def __init__(self, frame, plant):
        if frame.columns.empty or frame.columns[-1] != plant:
            raise ValueError(f"Expected the last column of the training data to be {plant!r}")
        self.plant = plant
        if _FEATURES["drop_final_row"]:
            frame = frame.iloc[:-1]
        rows = frame.drop(columns=_FEATURES["excluded_weather"]).dropna()
        self._target = rows[plant]
        self._weather = pd.get_dummies(rows.drop(columns=plant), columns=_FEATURES["categorical_weather"])

    def features(self, history):
        if not isinstance(history, Integral) or history not in HISTORY_COUNTS:
            raise ValueError(f"history must be one of {HISTORY_COUNTS}")

        # 과거 발전량은 결측 제거 후 남은 관측 행을 기준으로 선택합니다.
        first_lag = _FEATURES["first_lag_rows"]
        if history and first_lag * SLOT_HOURS <= FORECAST_LEAD_HOURS:
            raise ValueError("The first historical observation must precede the forecast issue time")
        lags = [
            self._target.shift(lag).rename(f"y_t{lag * SLOT_HOURS - FORECAST_LEAD_HOURS}h")
            for lag in range(first_lag, first_lag + history)
        ]
        return pd.concat([self._weather, *lags, self._target], axis=1).dropna().reset_index(drop=True)

    def split(self, history):
        frame = self.features(history)
        date_columns = {"tgt_Y": "year", "tgt_M": "month", "tgt_D": "day"}
        # 평가 연도는 최종 평가에 사용하고 나머지 연도로 학습합니다.
        test_year = _FEATURES["test_year"]
        test_rows = frame["tgt_Y"] == test_year
        test_calendar = frame.loc[test_rows, list(date_columns)].rename(columns=date_columns)
        hours = (frame.loc[test_rows, "tgt_tz"] * SLOT_HOURS).astype(int)
        if not hours.between(0, 23).all():
            raise ValueError("Test time slots must resolve to hours between 0 and 23")
        dates = pd.to_datetime(test_calendar) + pd.to_timedelta(hours, unit="h")
        # 연월일은 평가 시각 복원에 쓰고 시간대 번호는 입력 특성에 남깁니다.
        features = frame.drop(columns=[*date_columns, self.plant])
        target = frame[self.plant]
        if not test_rows.any() or test_rows.all():
            raise ValueError(f"Training and {test_year} test data must both contain rows after preprocessing")

        return TrainingSplit(
            train_x=features.loc[~test_rows].to_numpy(),
            train_y=target.loc[~test_rows].to_numpy().reshape(-1, 1),
            test_x=features.loc[test_rows].to_numpy(),
            test_y=target.loc[test_rows].to_numpy().reshape(-1, 1),
            test_dates=dates.reset_index(drop=True),
            feature_names=features.columns,
        )
