import pandas as pd

from wind_power_predictor.config import SETTINGS
from wind_power_predictor.preprocessing.time import (
    FORECAST_LEAD_HOURS, SLOT_HOURS, report_irregular_times, target_time_columns,
)


_WEATHER = SETTINGS["data"]["weather"]


def prepare_forecast(forecasts, variable, year):
    if "forecast" not in forecasts or forecasts.shape[1] < 4:
        raise ValueError("Weather CSV requires a forecast column and a fourth column with forecast values")
    # 원본 CSV의 네 번째 열에서 설정한 선행 시간의 예보값을 선택합니다.
    values = forecasts.loc[forecasts["forecast"].eq(FORECAST_LEAD_HOURS)].iloc[:, 3].reset_index(drop=True)
    # 발표 시각은 행 순서로 만들기 때문에 한 해의 발표 횟수와 다르면 알립니다.
    expected = len(pd.date_range(
        f"{year}-01-01 {_WEATHER['first_issue_time']}", f"{year + 1}-01-01", freq=f"{SLOT_HOURS}h", inclusive="left",
    ))
    if len(values) != expected:
        report_irregular_times(
            f"{variable} {year}: found {len(values)} forecasts but expected {expected}; "
            "issue times generated from row order may be shifted"
        )
    issued = pd.date_range(f"{year}-01-01 {_WEATHER['first_issue_time']}", periods=len(values), freq=f"{SLOT_HOURS}h")
    return pd.DataFrame({
        variable: values,
        "anc_time": issued,
        "tgt_beg": issued + pd.Timedelta(hours=FORECAST_LEAD_HOURS),
    })


def merge_weather_variables(forecasts):
    first, *others = forecasts
    merged = first.copy()
    # 일부 예보 항목이 누락된 시각도 남겨 후속 전처리에서 결측값을 처리합니다.
    for forecast in others:
        merged = merged.merge(forecast, on=["anc_time", "tgt_beg"], how="outer", validate="one_to_one")

    return merged.assign(**target_time_columns(merged["tgt_beg"].dt))
