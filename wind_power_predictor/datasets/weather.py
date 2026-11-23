import pandas as pd

from wind_power_predictor.preprocessing.weather import merge_weather_variables, prepare_forecast


class WeatherDataset:
    def __init__(self, path):
        self._forecasts = pd.read_csv(path)

    def prepare(self, variable, year):
        # 데이터 객체의 기존 호출 방식을 유지하면서 변환은 전처리 모듈에 맡깁니다.
        return prepare_forecast(self._forecasts, variable, year)

    merge_variables = staticmethod(merge_weather_variables)
