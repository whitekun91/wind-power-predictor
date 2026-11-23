from wind_power_predictor.config import SETTINGS


# 풍력 발전량 집계와 기상 예보에 공통으로 적용하는 시간 간격입니다.
SLOT_HOURS = SETTINGS["data"]["slot_hours"]
FORECAST_LEAD_HOURS = SETTINGS["data"]["weather"]["lead_hours"]
if type(SLOT_HOURS) is not int or SLOT_HOURS <= 0:
    raise ValueError("data.slot_hours must be a positive integer")


def target_time_columns(times):
    # 날짜와 3시간 구간 번호로 병합에 사용할 키를 만듭니다.
    return {
        "tgt_Y": times.year,
        "tgt_M": times.month,
        "tgt_D": times.day,
        "tgt_tz": times.hour / SLOT_HOURS,
    }
