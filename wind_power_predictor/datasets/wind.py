import pandas as pd

from wind_power_predictor.preprocessing.wind import prepare_wind


class WindDataset:
    def __init__(self, readings, *, fill_missing=False):
        self._readings = readings
        self._fill_missing = fill_missing

    @classmethod
    def from_csv(cls, path):
        return cls(pd.read_csv(path))

    @classmethod
    def from_excel(cls, path, sheet=0):
        # 엑셀 원천 자료의 빈 값 처리 조건을 전처리에 전달합니다.
        return cls(pd.read_excel(path, sheet_name=sheet), fill_missing=True)

    def prepare(self, plant):
        return prepare_wind(self._readings, plant, fill_missing=self._fill_missing)
