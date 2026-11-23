from dataclasses import dataclass
from pathlib import Path


# 결과를 쓰는 코드와 읽는 코드가 같은 파일명 규칙을 사용합니다.
@dataclass(frozen=True)
class PredictionFiles:
    plant: str
    output_path: Path

    def title(self, model, history):
        return f"{self.plant} Generation_{model} H{history}"

    def prediction_csv(self, model, history):
        return self.output_path / self.plant / f"h{history}" / f"{self.title(model, history)}.csv"

    def merged_csv(self, history):
        return self.output_path / f"{self.plant} Generation_total_result_h{history}.csv"
