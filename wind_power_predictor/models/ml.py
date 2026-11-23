from dataclasses import dataclass

from sklearn.base import RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

from wind_power_predictor.config import SETTINGS
from wind_power_predictor.features.split import TrainingSplit


_ML = SETTINGS["models"]["ml"]


@dataclass(frozen=True)
class Regressor:
    name: str
    estimator: RegressorMixin
    flatten_target: bool = True

    def fit(self, split: TrainingSplit):
        # 모델별로 요구하는 목표값 배열의 차원을 맞춥니다.
        target = split.train_y.ravel() if self.flatten_target else split.train_y
        return self.estimator.fit(split.train_x, target)


# 기상 변수의 단위와 값 범위 차이를 줄인 입력으로 학습할 모델입니다.
def scaled_regressors():
    return (
        Regressor("MLR", LinearRegression(**_ML["linear"]), flatten_target=False),
        Regressor("PLS", PLSRegression(), flatten_target=False),
        Regressor("ElasticNet", ElasticNetCV(**_ML["elastic_net"])),
        Regressor("SVR", LinearSVR(random_state=_ML["random_state"], **_ML["svr"])),
    )


# 결정 트리와 트리 기반 앙상블은 입력 특성의 원래 단위를 사용합니다.
def tree_regressors():
    from lightgbm import LGBMRegressor

    seed = _ML["random_state"]
    yield Regressor("DT", DecisionTreeRegressor(random_state=seed), flatten_target=False)
    yield Regressor("RF", RandomForestRegressor(random_state=seed, **_ML["random_forest"]))
    yield Regressor("Bag", BaggingRegressor(random_state=seed))
    yield Regressor("Ada", AdaBoostRegressor(random_state=seed))
    yield Regressor("GBR", GradientBoostingRegressor(random_state=seed))
    yield xgboost_regressor()
    yield Regressor("LGBM", LGBMRegressor(**_ML["lightgbm"]))


def xgboost_regressor():
    from xgboost import XGBRegressor

    return Regressor("XGB", XGBRegressor(**_ML["xgboost"]), flatten_target=False)
