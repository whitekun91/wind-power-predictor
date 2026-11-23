from pathlib import Path

import pandas as pd

from wind_power_predictor.config import SETTINGS
from wind_power_predictor.datasets.weather import WeatherDataset
from wind_power_predictor.datasets.wind import WindDataset
from wind_power_predictor.preprocessing.merge import WindWeatherMerge


def _existing(path, label):
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def prepare_dataset(raw_data, plant):
    root = Path(raw_data)
    data = SETTINGS["data"]
    if plant not in data["plants"]:
        raise ValueError(f"Unknown wind plant: {plant}")
    source = data["plants"][plant]
    # 원천 파일 문제는 어느 발전소의 어느 파일인지 알 수 있게 알립니다.
    workbook = _existing(root / source["wind_file"], f"{plant} wind workbook")
    try:
        readings = WindDataset.from_excel(workbook, source["wind_sheet"])
    except ValueError as error:
        raise ValueError(f"{plant}: cannot read sheet {source['wind_sheet']} of {workbook}: {error}") from error
    wind = readings.prepare(plant)

    weather = data["weather"]
    annual = []
    for year in weather["years"]:
        forecasts = []
        for variable in WindWeatherMerge.WEATHER_COLUMNS:
            path = _existing(
                root / weather["file_pattern"].format(site=source["weather_site"], year=year, variable=variable),
                f"{plant} {variable} {year} weather forecast",
            )
            try:
                forecasts.append(WeatherDataset(path).prepare(variable, year))
            except ValueError as error:
                raise ValueError(f"{path}: {error}") from error
        annual.append(WeatherDataset.merge_variables(forecasts))

    # 연도별 기상 자료를 이어 붙이고 병합 결과를 메모리에서 바로 반환합니다.
    return WindWeatherMerge(wind, pd.concat(annual, ignore_index=True)).merge(plant)
