from wind_power_predictor.scripts._train import run_training


# 공통 실행기에 ML 전용 변수 중요도 옵션을 활성화합니다.
def main():
    run_training("Train ML models from raw wind and weather data.", ml=True)


if __name__ == "__main__":
    main()
