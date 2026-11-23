from wind_power_predictor.config import DEFAULT_OUTPUT_PATH
from wind_power_predictor.config import PLANTS as _PLANTS
from wind_power_predictor.features.training import TrainingDataset
from wind_power_predictor.models.ml import scaled_regressors, tree_regressors, xgboost_regressor
from wind_power_predictor.reporting.report import PredictionReport


# 발전소별로 데이터를 한 번 준비하고 입력 개수에 따라 실험을 반복합니다.
class PredictionExperiment:
    PLANTS = _PLANTS

    def __init__(self, frame, plant, output_path=DEFAULT_OUTPUT_PATH):
        if plant not in self.PLANTS:
            raise ValueError(f"Unknown wind plant: {plant}")
        self.dataset = TrainingDataset(frame, plant)
        self.report = PredictionReport(plant, output_path)

    def train_ml(self, history, *, feature_importance=False):
        split = self.dataset.split(history)
        if feature_importance:
            fitted = xgboost_regressor().fit(split)
            self.report.save_importance(split.feature_names, fitted.feature_importances_, "Xgboost", history, "xgboost")
            self.report.save(split, fitted.predict(split.test_x), "Xgboost_", history)
            return

        # 정규화 모델과 트리 모델에 같은 분할을 사용해 평가 표본을 맞춥니다.
        self._fit_regressors(split.scaled(), scaled_regressors(), history)
        self._fit_regressors(split, tree_regressors(), history)

    def _fit_regressors(self, split, regressors, history):
        for regressor in regressors:
            fitted = regressor.fit(split)
            if regressor.name == "ElasticNet":
                self.report.save_coefficients(split.feature_names, fitted.coef_, history)
            elif regressor.name == "RF":
                self.report.save_importance(split.feature_names, fitted.feature_importances_, "RF", history, "randomforest")
            self.report.save(split, fitted.predict(split.test_x), regressor.name, history)

    def train_dl(self, history):
        # ML 실행 환경에서 PyTorch 설치를 요구하지 않도록 DNN 실행 시 가져옵니다.
        from wind_power_predictor.models.dnn import DnnModel

        split = self.dataset.split(history).scaled()
        model = DnnModel(split.train_x.shape[1])
        model.fit(split.train_x, split.train_y)
        self.report.save(split, model.predict(split.test_x), "MLP", history)
