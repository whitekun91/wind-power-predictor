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
