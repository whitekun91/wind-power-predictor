from wind_power_predictor.scripts._train import run_training


# DNN 전용 명령도 공통 실행기의 데이터와 실험 옵션을 사용합니다.
def main():
    run_training("Train the PyTorch DNN from raw wind and weather data.")


if __name__ == "__main__":
    main()
