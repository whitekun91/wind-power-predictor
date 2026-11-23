import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wind_power_predictor.config import SETTINGS


_DNN = SETTINGS["models"]["dnn"]


class DnnModel(nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        self.hidden = nn.Sequential()
        input_size = feature_count
        hidden_units = _DNN["hidden_units"]
        # 기존 은닉층에서 ReLU 뒤의 LeakyReLU는 음수 입력을 받지 않았습니다.
        for index in range(_DNN["hidden_layers"]):
            # 기존 체크포인트의 은닉층 가중치 이름을 유지합니다.
            self.hidden.add_module(str(index * 3), nn.Linear(input_size, hidden_units))
            self.hidden.add_module(str(index * 3 + 1), nn.ReLU())
            input_size = hidden_units
        self.output = nn.Linear(input_size, 1)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, features):
        return self.output(self.hidden(features))

    def fit(self, train_x, train_y):
        features = torch.as_tensor(train_x, dtype=torch.float32)
        target = torch.as_tensor(train_y, dtype=torch.float32).reshape(-1, 1)
        # 입력 순서의 끝을 검증에 쓰고 학습 구간에서만 배치를 섞습니다.
        split = int(len(features) * (1 - _DNN["validation_fraction"]))
        if len(target) != len(features) or not 0 < split < len(features):
            raise ValueError("DNN training and validation require at least two paired feature and target rows")

        batches = DataLoader(TensorDataset(features[:split], target[:split]), batch_size=_DNN["batch_size"], shuffle=True)
        # 기본 설정은 기존 Keras Nadam의 모멘텀 감쇠 일정을 따릅니다.
        optimizer = torch.optim.NAdam(self.parameters(), **_DNN["nadam"])
        epochs = _DNN["epochs"]
        if epochs < 1:
            raise ValueError("models.dnn.epochs must be positive")
        for epoch in range(1, epochs + 1):
            self.train()
            train_loss = 0.0
            for batch_x, batch_y in batches:
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(batch_x, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_x)

            self.eval()
            with torch.no_grad():
                validation_loss = self._loss(features[split:], target[split:]).item()
            print(f"Epoch {epoch}/{epochs} - loss: {train_loss / split:.6f} - val_loss: {validation_loss:.6f}")

    @torch.inference_mode()
    def predict(self, test_x):
        self.eval()
        return self(torch.as_tensor(test_x, dtype=torch.float32)).numpy()

    def _loss(self, features, target):
        mse = nn.functional.mse_loss(self(features), target)
        # L1과 L2 규제는 기존 실험과 같이 은닉층의 가중치에만 적용합니다.
        penalty = sum(
            layer.weight.abs().sum() + layer.weight.square().sum()
            for layer in self.hidden
            if isinstance(layer, nn.Linear)
        )
        return mse + _DNN["l1_l2"] * penalty
