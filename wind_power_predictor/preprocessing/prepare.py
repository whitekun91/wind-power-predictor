from pathlib import Path

import pandas as pd

from wind_power_predictor.config import SETTINGS
from wind_power_predictor.datasets.weather import WeatherDataset
from wind_power_predictor.datasets.wind import WindDataset
from wind_power_predictor.preprocessing.merge import WindWeatherMerge


def prepare_dataset(raw_data, plant):
    root = Path(raw_data)
    data = SETTINGS["data"]
    source = data["plants"][plant]
    wind = WindDataset.from_excel(root / source["wind_file"], source["wind_sheet"]).prepare(plant)

    weather = data["weather"]
    annual = []
    for year in weather["years"]:
        forecasts = (
            WeatherDataset(root / weather["file_pattern"].format(
                site=source["weather_site"], year=year, variable=variable,
            )).prepare(variable, year)
            for variable in WindWeatherMerge.WEATHER_COLUMNS
        )
        annual.append(WeatherDataset.merge_variables(forecasts))

    # 연도별 기상 자료를 이어 붙이고 병합 결과를 메모리에서 바로 반환합니다.
    return WindWeatherMerge(wind, pd.concat(annual, ignore_index=True)).merge(plant)
