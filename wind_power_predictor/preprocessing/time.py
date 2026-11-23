import warnings

from wind_power_predictor.config import SETTINGS


# 풍력 발전량 집계와 기상 예보에 공통으로 적용하는 시간 간격입니다.
SLOT_HOURS = SETTINGS["data"]["slot_hours"]
FORECAST_LEAD_HOURS = SETTINGS["data"]["weather"]["lead_hours"]
if type(SLOT_HOURS) is not int or SLOT_HOURS <= 0:
    raise ValueError("data.slot_hours must be a positive integer")


def report_irregular_times(message):
    # 밀린 시각으로 학습하지 않도록 중단하고, 원본 조건을 재현할 때만 경고로 낮춥니다.
    if SETTINGS["data"]["strict_source_times"]:
        raise ValueError(f"{message}. Set data.strict_source_times to false to continue with a warning")
    warnings.warn(message, stacklevel=3)


def target_time_columns(times):
    # 날짜와 3시간 구간 번호로 병합에 사용할 키를 만듭니다.
    return {
        "tgt_Y": times.year,
        "tgt_M": times.month,
        "tgt_D": times.day,
        "tgt_tz": times.hour / SLOT_HOURS,
    }
