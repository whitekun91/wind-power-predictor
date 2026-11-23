import csv
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from wind_power_predictor.features.split import TrainingSplit
from wind_power_predictor.reporting.merge import PredictionMerger
from wind_power_predictor.reporting.report import PredictionReport


class PredictionReportTest(unittest.TestCase):
    def test_rerun_updates_one_score_row_to_match_the_latest_predictions(self):
        target = np.array([[100.0], [200.0]])
        split = TrainingSplit(
            train_x=target, train_y=target, test_x=target, test_y=target,
            test_dates=pd.Series(pd.date_range("2017-01-01", periods=2, freq="3h")),
            feature_names=pd.Index(["x"]),
        )
        title = "hk1 Generation_MLP H0"
        other = ["hk2 Generation_MLP H0", "1", "0", "0", "0"]
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)
            summary = output / "model_performance.csv"
            # 재실행할 때 기존 중복 지표를 정리하면서 다른 발전소의 지표를 보존해야 합니다.
            with summary.open("w", newline="") as file:
                csv.writer(file).writerows([other, [title, 1, 0, 0, 0], [title, 0.96, 100, 10, 10]])

            report = PredictionReport("hk1", output)
            report.save(split, [90, 210], "MLP", 0)
            report.save(split, [120, 180], "MLP", 0)
            with self.assertRaises(ValueError):
                report.save(split, [np.nan, 180], "MLP", 0)

            result_path = output / "hk1" / "h0" / f"{title}.csv"
            graph_path = result_path.with_suffix(".png")
            last_graph = graph_path.read_bytes()
            replace = Path.replace

            def deny_summary_replace(source, destination):
                if destination == summary:
                    raise PermissionError("Summary is open")
                return replace(source, destination)

            with patch.object(Path, "replace", deny_summary_replace), self.assertRaises(PermissionError):
                report.save(split, [80, 220], "MLP", 0)

            predictions = pd.read_csv(result_path)
            self.assertEqual(predictions["predict_value"].tolist(), [120, 180])
            self.assertEqual(graph_path.read_bytes(), last_graph)
            with summary.open(newline="") as file:
                rows = list(csv.reader(file))
            self.assertEqual(rows[0], other)
            self.assertEqual([row[0] for row in rows], [other[0], title])
            self.assertEqual(float(rows[1][2]), 400)


class PredictionMergerTest(unittest.TestCase):
    def test_merge_uses_timestamps_and_rejects_mismatched_runs(self):
        actual = pd.DataFrame({
            "tgt_datetime": ["2017-01-01 00:00", "2017-01-01 03:00"],
            "actual_value": [1, 2],
            "predict_value": [3, 4],
        })
        with TemporaryDirectory() as directory:
            output = Path(directory)
            data = output / "hk1" / "h0"
            data.mkdir(parents=True)
            actual.to_csv(data / "hk1 Generation_MLR H0.csv", index=False)
            result_path = data / "hk1 Generation_PLS H0.csv"
            other = actual.assign(predict_value=[10, 20])
            # 한 모델의 저장 순서를 뒤집어도 예측 시각으로 정확히 연결해야 합니다.
            other.iloc[::-1].to_csv(result_path, index=False)
            merger = PredictionMerger("hk1", 0, output)
            result = pd.read_csv(merger.save(["PLS"]))
            self.assertEqual(result["PLS_predict"].tolist(), [10, 20])
            self.assertEqual(result["PLS_residual"].tolist(), [9, 18])

            other.iloc[:1].to_csv(result_path, index=False)
            with self.assertRaisesRegex(ValueError, "same prediction timestamps"):
                merger.merge(["PLS"])

            other.iloc[[0, 0]].to_csv(result_path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate prediction timestamps"):
                merger.merge(["PLS"])

            other.assign(actual_value=[1, 3]).to_csv(result_path, index=False)
            with self.assertRaisesRegex(ValueError, "same actual values"):
                merger.merge(["PLS"])
