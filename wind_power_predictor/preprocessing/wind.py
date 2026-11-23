import numpy as np
import pandas as pd

from wind_power_predictor.config import PLANTS, SETTINGS
from wind_power_predictor.preprocessing.time import SLOT_HOURS, target_time_columns


_WIND = SETTINGS["data"]["wind"]


def prepare_wind(readings, plant, *, fill_missing=False):
    if plant not in PLANTS:
        raise ValueError(f"Unknown wind plant: {plant}")
    # 원본 엑셀 자료에 적용하던 빈 값 처리 규칙입니다.
    if fill_missing:
        readings = readings.fillna(0)

    generation = readings.drop(columns="PCTimeStamp").max(axis=1).clip(lower=0)
    if plant in _WIND["convert_to_kw"]:
        generation = generation * _WIND["kw_multiplier"] / _WIND["kw_divisor"]

    # 원본 실험의 첫 집계 구간에 맞춰 앞쪽 관측값을 건너뜁니다.
    generation = generation.iloc[_WIND["skip_readings"]:]
    interval = _WIND["sampling_minutes"]
    if interval <= 0 or SLOT_HOURS * 60 % interval:
        raise ValueError("Wind sampling interval must evenly divide the target time slot")
    readings_per_slot = (SLOT_HOURS * 60) // interval
    averages = generation.groupby(np.arange(len(generation)) // readings_per_slot).mean()
    times = pd.date_range(_WIND["first_target_time"], periods=len(averages), freq=f"{SLOT_HOURS}h")
    return pd.DataFrame({
        "tgt_time": times,
        **target_time_columns(times),
        plant: averages.to_numpy(),
    })
