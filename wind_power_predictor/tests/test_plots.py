import unittest

import pandas as pd
from plotly.subplots import make_subplots

from wind_power_predictor.reporting.plots import PredictVisualization


class PredictionVisualizationTest(unittest.TestCase):
    def test_monthly_lines_filter_dates_and_residual_plots_keep_their_axes(self):
        # 두 달의 자료를 함께 넣어 선택한 달만 그래프에 표시되는지 확인합니다.
        dates = ["2017-03-01 06:00", "2017-06-01 06:00"]
        data = pd.DataFrame({
            "tgt_datetime": dates,
            "actual_value": [10, 20],
            **{f"model_{index}_predict": [9, 21] for index in range(11)},
            "model_0_residual": [-1, 1],
            "month": [3, 6],
        })
        plots = PredictVisualization(data)

        monthly = make_subplots(rows=2, cols=1)
        plots.h2h3_monthly_line_visualization(monthly, 3)
        self.assertEqual(len(monthly.data), 13)
        self.assertTrue(all(list(trace.x) == dates[:1] for trace in monthly.data))

        residuals = make_subplots(rows=2, cols=3)
        plots.h4_scatterplot_matrix_total_plants_prediction(residuals, "hk2", "model_0")
        self.assertEqual([trace.xaxis for trace in residuals.data], ["x2", "x2", "x5", "x5"])
        self.assertEqual(residuals.layout.yaxis5.title.text, "Residual value")
        self.assertEqual(set(residuals.data[-1].y), {0})
        with self.assertRaisesRegex(ValueError, "Unknown wind plant"):
            plots.h4_scatterplot_matrix_total_plants_prediction(residuals, "unknown", "model_0")

    def test_lines_use_prediction_names_and_timestamps_for_any_model_count(self):
        # 월과 일 열이 없고 잔차 열이 섞인 12개 모델 자료에서도 예측값만 모두 그려야 합니다.
        data = pd.DataFrame({
            "tgt_datetime": ["2017-03-05 06:00", "2017-03-06 06:00", "2017-06-05 06:00"],
            "actual_value": [10, 20, 30],
            "model_0_residual": [-1, 1, 0],
            **{f"model_{index}_predict": [9, 21, 30] for index in range(12)},
        })
        plots = PredictVisualization(data)

        yearly = make_subplots(rows=2, cols=1)
        plots.h2h3_yearly_line_visualization(yearly)
        names = [trace.name for trace in yearly.data]
        self.assertEqual(names.count("actual_value"), 2)
        self.assertEqual(sorted(set(names) - {"actual_value"}), sorted(f"model_{index}_predict" for index in range(12)))

        one_day = make_subplots(rows=4, cols=1)
        plots.h4_one_day_lineplot(one_day, [3], 5)
        self.assertEqual(len(one_day.data), 13)
        self.assertTrue(all(list(trace.x) == ["2017-03-05 06:00"] for trace in one_day.data))
