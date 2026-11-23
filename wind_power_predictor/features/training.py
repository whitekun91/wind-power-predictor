from numbers import Integral

import pandas as pd

from wind_power_predictor.config import SETTINGS
from wind_power_predictor.features.split import TrainingSplit
from wind_power_predictor.preprocessing.time import FORECAST_LEAD_HOURS, SLOT_HOURS


_FEATURES = SETTINGS["features"]
HISTORY_COUNTS = tuple(_FEATURES["history_counts"])
if not HISTORY_COUNTS or any(type(count) is not int or count < 0 for count in HISTORY_COUNTS):
    raise ValueError("features.history_counts must contain nonnegative integers")


_DATE_COLUMNS = {"tgt_Y": "year", "tgt_M": "month", "tgt_D": "day"}


def _target_times(frame):
    # 연월일과 시간대 번호에서 예측 대상 시각을 복원합니다.
    calendar = frame[list(_DATE_COLUMNS)].rename(columns=_DATE_COLUMNS)
    return pd.to_datetime(calendar) + pd.to_timedelta(frame["tgt_tz"] * SLOT_HOURS, unit="h")


class TrainingDataset:
    def __init__(self, frame, plant):
        if frame.columns.empty or frame.columns[-1] != plant:
            raise ValueError(f"Expected the last column of the training data to be {plant!r}")
        self.plant = plant
        if _FEATURES["drop_final_row"]:
            frame = frame.iloc[:-1]
        # 과거 발전량은 기상 예보의 결측과 관계없이 관측 시각으로 찾습니다.
        observed = frame.dropna(subset=[plant])
        self._history = pd.Series(observed[plant].to_numpy(), index=_target_times(observed))
        if self._history.index.has_duplicates:
            raise ValueError("Training data must contain one row per target time")
        rows = frame.drop(columns=_FEATURES["excluded_weather"]).dropna()
        self._times = _target_times(rows)
        self._target = rows[plant]
        self._weather = pd.get_dummies(rows.drop(columns=plant), columns=_FEATURES["categorical_weather"])

    def features(self, history):
        if not isinstance(history, Integral) or history not in HISTORY_COUNTS:
            raise ValueError(f"history must be one of {HISTORY_COUNTS}")

        # 논문 Figure 3과 같이 과거 발전량은 대상 시각에서 거슬러 올라간 시각의 값을 씁니다.
        # 해당 시각의 관측이 없으면 다른 시각의 값으로 대신하지 않고 그 행을 제외합니다.
        first_lag = _FEATURES["first_lag_rows"]
        if history and first_lag * SLOT_HOURS <= FORECAST_LEAD_HOURS:
            raise ValueError("The first historical observation must precede the forecast issue time")
        lags = [
            pd.Series(
                self._history.reindex(self._times - pd.Timedelta(hours=lag * SLOT_HOURS)).to_numpy(),
                index=self._target.index, name=f"y_t{lag * SLOT_HOURS - FORECAST_LEAD_HOURS}h",
            )
            for lag in range(first_lag, first_lag + history)
        ]
        return pd.concat([self._weather, *lags, self._target], axis=1).dropna().reset_index(drop=True)

    def split(self, history):
        frame = self.features(history)
        date_columns = _DATE_COLUMNS
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
