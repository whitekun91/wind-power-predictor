# 발전량과 기상 예보를 예보 대상 시각 기준으로 연결합니다.
class WindWeatherMerge:
    TIME_KEYS = ["tgt_Y", "tgt_M", "tgt_D", "tgt_tz"]
    WEATHER_COLUMNS = [
        "temp3h", "humidity", "windspeed", "winddirection", "rain6h", "snow6h",
        "rainprobability", "raintype", "seawave", "skystatus",
    ]

    def __init__(self, wind, weather):
        self._wind = wind
        self._weather = weather

    def merge(self, plant):
        columns = [*self.TIME_KEYS, *self.WEATHER_COLUMNS]
        # 두 자료에 모두 있는 시각을 사용하고 같은 시각의 중복 병합을 막습니다.
        merged = self._wind[[*self.TIME_KEYS, plant]].merge(
            self._weather[columns], on=self.TIME_KEYS, how="inner", validate="one_to_one",
        )
        return merged[[*columns, plant]]
