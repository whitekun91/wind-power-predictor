import unittest

import numpy as np
import pandas as pd

from wind_power_predictor.experiment import PredictionExperiment


class PredictionExperimentTest(unittest.TestCase):
    def test_history_settings_share_original_rows_and_keep_the_same_test_dates(self):
        raw = pd.DataFrame({
            "tgt_Y": [2016] * 8 + [2017] * 4,
            "tgt_M": 1,
            "tgt_D": 1,
            "tgt_tz": list(range(8)) + list(range(4)),
            "raintype": 0,
            "skystatus": 1,
            "seawave": 0,
            "hk1": list(range(12)),
        })
        dataset = PredictionExperiment(raw, "hk1").dataset

        # 같은 객체에서 서로 다른 입력 개수를 연속으로 사용해 특성 생성의 독립성을 확인합니다.
        h5 = dataset.features(5)
        h0 = dataset.features(0)
        with self.assertRaises(ValueError):
            dataset.features(6)
        self.assertEqual(h5.loc[0, ["y_t2h", "y_t5h", "y_t8h", "y_t11h", "y_t14h"]].tolist(),
                         [4, 3, 2, 1, 0])
        self.assertEqual(h0["hk1"].tolist(), list(range(11)))

        split = dataset.split(5)
        scaled = split.scaled()
        pd.testing.assert_series_equal(scaled.test_dates, split.test_dates)
        pd.testing.assert_index_equal(scaled.feature_names, split.feature_names)
        np.testing.assert_allclose(scaled.original_target(scaled.test_y), split.test_y.ravel())
        scaled_again = scaled.scaled()
        np.testing.assert_allclose(scaled_again.original_target(scaled_again.test_y), split.test_y.ravel())
        self.assertGreater(scaled.test_y[0, 0], 1)
        self.assertEqual(split.test_dates.tolist(), list(pd.date_range("2017-01-01", periods=3, freq="3h")))
