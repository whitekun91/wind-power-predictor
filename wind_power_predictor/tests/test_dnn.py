import io
import unittest
from contextlib import redirect_stdout
from importlib.util import find_spec

import numpy as np


@unittest.skipUnless(find_spec("torch") is not None, "DNN tests require PyTorch")
class DnnModelTest(unittest.TestCase):
    def test_fits_and_predicts_one_value_per_row(self):
        import torch

        from wind_power_predictor.models.dnn import DnnModel

        features = np.linspace(0, 1, 20).reshape(-1, 1)
        target = 0.2 + 0.6 * features

        # 다른 테스트의 난수 상태를 보존한 채 가중치 초기화를 재현합니다.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = DnnModel(feature_count=1)
            before = model.predict(features)
            with redirect_stdout(io.StringIO()):
                model.fit(features, target)
            after = model.predict(features)

        self.assertEqual(after.shape, target.shape)
        self.assertTrue(np.isfinite(after).all())
        self.assertLess(np.mean((after - target) ** 2), np.mean((before - target) ** 2))
