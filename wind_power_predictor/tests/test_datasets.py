import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from wind_power_predictor.datasets.weather import WeatherDataset
from wind_power_predictor.datasets.wind import WindDataset
from wind_power_predictor.preprocessing.merge import WindWeatherMerge
from wind_power_predictor.preprocessing.prepare import prepare_dataset


class DatasetTest(unittest.TestCase):
    def test_raw_generation_and_weather_join(self):
        # 첫 28개 관측값을 제외하면 완전한 집계 구간과 부분 구간이 하나씩 남습니다.
        readings = pd.DataFrame({
            "PCTimeStamp": "2014-01-01 00:00:00",
            "generator_1": [0] * 28 + [-2, 18] * 9 + [36],
            "generator_2": -3,
        })
        with TemporaryDirectory() as directory:
            path = Path(directory)
            readings.to_csv(path / "wind.csv", index=False)
            wind = WindDataset.from_csv(path / "wind.csv").prepare("hk1")
            readings.to_excel(path / "wind.xlsx", index=False)
            wind_kw = WindDataset.from_excel(path / "wind.xlsx").prepare("hk2")

            pd.DataFrame({
                "day": [1, 1, 1], "hour": [200, 200, 500],
                "forecast": [4, 25, 4], "value": [10, 90, 20],
            }).to_csv(path / "weather.csv", index=False)
            forecast = WeatherDataset(path / "weather.csv")
            weather = WeatherDataset.merge_variables(
                forecast.prepare(name, 2014) for name in WindWeatherMerge.WEATHER_COLUMNS
            )

        self.assertEqual(wind["hk1"].tolist(), [9, 36])
        np.testing.assert_allclose(wind_kw["hk2"], [0.054, 0.216])
        self.assertEqual(len(WindDataset(readings.iloc[:-1]).prepare("hk1")), 1)
        merged = WindWeatherMerge(wind, weather.iloc[::-1]).merge("hk1")
        self.assertEqual(merged[["temp3h", "hk1"]].values.tolist(), [[10, 9], [20, 36]])
        self.assertEqual(merged.columns[-1], "hk1")

    def test_prepare_dataset_reads_the_configured_workbook_sheet_and_returns_a_frame(self):
        readings = pd.DataFrame({
            "PCTimeStamp": "2014-01-01",
            "generator": [0] * 28 + [1000] * 18 + [2000] * 18,
        })
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = root / "wind" / "붙임1_한경풍력 발전실적 정보.xlsx"
            workbook.parent.mkdir()
            with pd.ExcelWriter(workbook) as writer:
                readings.assign(generator=0).to_excel(writer, sheet_name="hk1", index=False)
                readings.to_excel(writer, sheet_name="hk2", index=False)

            # 원본 예보 폴더 구조를 사용하면서 병합 중간 파일이 생기는지 확인합니다.
            for year in (2014, 2015, 2016, 2017):
                folder = root / "weather" / "hankyeong" / str(year)
                folder.mkdir(parents=True)
                for variable in WindWeatherMerge.WEATHER_COLUMNS:
                    (folder / f"weather_forecast_{variable}.csv").write_text(
                        "day,hour,forecast,value\n1,200,4,10\n1,500,4,20\n", encoding="utf-8",
                    )
            source_paths = set(root.rglob("*"))
            merged = prepare_dataset(root, "hk2")
            self.assertEqual(set(root.rglob("*")), source_paths)

        self.assertEqual(merged[["temp3h", "hk2"]].values.tolist(), [[10, 6], [20, 12]])
