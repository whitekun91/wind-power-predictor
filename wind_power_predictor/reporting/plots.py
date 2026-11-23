import plotly.express as px
import plotly.graph_objects as go

from wind_power_predictor.config import SETTINGS


_PLOTS = SETTINGS["reporting"]["plots"]
_RESIDUAL_LOWER = _PLOTS["residual_lower"]


class PredictVisualization:
    _MODEL_COLORS = (
        "darkblue", "darkviolet", "lightcoral", "limegreen", "olive", "orange",
        "pink", "red", "sandybrown", "steelblue", "yellow",
    )

    def __init__(self, dataset):
        self.dataset = dataset

    def scatterplot_matrix_actual_and_prediction(self, fig, plant, his, model_name):
        # 모델을 행에, 과거 발전량 입력 개수를 열에 배치합니다.
        _, limits, _ = self._plant_layout(plant)
        reference = list(range(*limits))
        for row, name in enumerate(model_name, start=1):
            self._scatter(fig, name, "actual_value", f"{name}_predict", reference, (row, his + 1))

    def h2h3_scatterplot_matrix_prediction_and_residual(self, fig, plant, model_name):
        # 모델별로 첫 행에 실제값 대비 예측값을, 둘째 행에 잔차를 표시합니다.
        _, _, limit = self._plant_layout(plant)
        reference = list(range(_RESIDUAL_LOWER, limit))
        for column, name in enumerate(model_name, start=1):
            self._residual_pair(fig, name, reference, (1, column), (2, column))

    def h2h3_scatterplot_matrix_total_plants_prediction(self, fig, plant, model_name):
        plant_column, _, limit = self._plant_layout(plant)
        reference = list(range(_RESIDUAL_LOWER, limit))
        for row, name in enumerate(model_name, start=1):
            self._residual_pair(fig, name, reference, (row, plant_column * 2 - 1), (row, plant_column * 2))

    def h4_scatterplot_matrix_total_plants_prediction(self, fig, plant, model_name):
        column, _, limit = self._plant_layout(plant)
        reference = list(range(_RESIDUAL_LOWER, limit))
        self._residual_pair(fig, model_name, reference, (1, column), (2, column), font_size=15)

    def monthly_line_visualization(self, fig, his, month):
        data = self.dataset.loc[self.dataset["month"] == month]
        self._lines(fig, data, data.columns[1:13], ("olivedrab", *self._MODEL_COLORS), his + 1)

    def h2h3_yearly_line_visualization(self, fig):
        self._grouped_lines(fig, self.dataset)

    def h2h3_monthly_line_visualization(self, fig, month):
        # 연간 그래프와 같은 모델 구성을 사용하되 선택한 달만 표시합니다.
        self._grouped_lines(fig, self.dataset.loc[self.dataset["month"] == month])

    def h4_one_day_lineplot(self, fig, month_list, day):
        # 월을 일부만 선택해도 분기별 그래프의 행 위치는 유지합니다.
        quarter_rows = {3: 1, 6: 2, 9: 3}
        for month in month_list:
            self._day_lines(fig, month, day, quarter_rows.get(month, 4), self._MODEL_COLORS, legend=month == 3)

    def h4_one_day_lineplot_dec(self, fig, month, day):
        self._day_lines(fig, month, day, 1, px.colors.qualitative.Safe, legend=month == 12)

    def _plant_layout(self, plant):
        # 연구에서 사용한 발전소별 배치와 기준선 범위는 실험 설정에서 읽습니다.
        plants = _PLOTS["plants"]
        layout = plants.get(plant, plants[_PLOTS["default_plant"]])
        return layout["column"], layout["actual_range"], layout["residual_upper"]

    def _scatter(self, fig, name, x_column, y_column, reference, position, *, residual=False, font_size=10):
        row, column = position
        fig.add_trace(
            go.Scatter(
                x=self.dataset[x_column], y=self.dataset[y_column], mode="markers",
                name=name, marker={"color": "Grey", "opacity": 0.5},
            ),
            row=row, col=column,
        )
        # 잔차 그래프에는 0선을, 실제값 대비 예측값 그래프에는 일치선을 그립니다.
        fig.add_trace(
            go.Scatter(
                x=reference, y=[0] * len(reference) if residual else reference,
                mode="lines", name="lines", line={"color": "grey", "width": 1, "dash": "dash"},
            ),
            row=row, col=column,
        )
        self._axes(
            fig, position, "Predict value" if residual else "Actual value",
            "Residual value" if residual else "Predict value", font_size,
        )

    def _residual_pair(self, fig, name, reference, prediction_position, residual_position, *, font_size=10):
        prediction = f"{name}_predict"
        self._scatter(fig, name, "actual_value", prediction, reference, prediction_position, font_size=font_size)
        self._scatter(
            fig, name, prediction, f"{name}_residual", reference, residual_position,
            residual=True, font_size=font_size,
        )

    def _grouped_lines(self, fig, data):
        # 병합 CSV의 열 순서에 따라 단일 모델과 앙상블 모델을 두 행에 나눕니다.
        for row, columns, colors in (
            (1, data.columns[2:8], self._MODEL_COLORS[:6]),
            (2, data.columns[8:13], self._MODEL_COLORS[6:]),
        ):
            self._lines(fig, data, ["actual_value", *columns], ("olivedrab", *colors), row)

    def _day_lines(self, fig, month, day, row, model_colors, *, legend):
        data = self.dataset.loc[(self.dataset["month"] == month) & (self.dataset["day"] == day)]
        # 여러 날짜를 한 그림에 배치할 때 범례를 한 번만 표시합니다.
        showlegend = None if legend else False
        self._lines(fig, data, ["actual_value"], ["black"], row, width=3, showlegend=showlegend)
        self._lines(fig, data, data.columns[2:13], model_colors, row, showlegend=showlegend)
        self._axes(fig, (row, 1), "Time", "Power [kW]", 15)

    @staticmethod
    def _lines(fig, data, columns, colors, row, *, width=1, showlegend=None):
        for column, color in zip(columns, colors):
            fig.add_trace(
                go.Scatter(
                    x=data["tgt_datetime"], y=data[column], mode="lines+markers", name=column,
                    line={"color": color, "width": width}, showlegend=showlegend,
                ),
                row=row, col=1,
            )

    @staticmethod
    def _axes(fig, position, x_title, y_title, font_size):
        row, column = position
        tick_size = 9 if font_size == 10 else font_size
        style = {"row": row, "col": column, "title_font": {"size": font_size}, "tickfont": {"size": tick_size}}
        fig.update_xaxes(title_text=x_title, **style)
        fig.update_yaxes(title_text=y_title, **style)
