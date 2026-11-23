import argparse
from functools import partial

from wind_power_predictor.config import DEFAULT_OUTPUT_PATH
from wind_power_predictor.experiment import PredictionExperiment
from wind_power_predictor.features.training import HISTORY_COUNTS
from wind_power_predictor.preprocessing.prepare import prepare_dataset


# 두 학습 명령의 데이터, 발전소, 입력 개수 옵션을 같은 규칙으로 처리합니다.
def run_training(description, *, ml=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("raw_data", help="Directory containing the original wind workbooks and weather CSVs.")
    parser.add_argument("plant", choices=PredictionExperiment.PLANTS)
    parser.add_argument("--history", type=int, choices=HISTORY_COUNTS, help="Default: run all configured values.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    if ml:
        parser.add_argument(
            "--feature-importance", action="store_true", help="Run only the XGBoost importance experiment.",
        )

    args = parser.parse_args()
    try:
        frame = prepare_dataset(args.raw_data, args.plant)
    except (FileNotFoundError, ValueError) as error:
        # 원천 자료 문제는 추적 정보 대신 원인만 보여 주고 종료합니다.
        parser.error(str(error))
    experiment = PredictionExperiment(frame, args.plant, args.output)
    train = partial(experiment.train_ml, feature_importance=args.feature_importance) if ml else experiment.train_dl
    # 입력 개수를 지정하지 않으면 실험 설정에 등록된 값을 모두 평가합니다.
    history_counts = HISTORY_COUNTS if args.history is None else (args.history,)
    for history in history_counts:
        train(history)
