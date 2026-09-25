import torch
from torch import Tensor, nn


class BattleValueModel(nn.Module):
    def __init__(self, feature_count: int, hidden_dim: int = 128) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(feature_count, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

        final_layer = self.network[-2]

        if not isinstance(final_layer, nn.Linear):
            raise TypeError("Expected the penultimate layer to be nn.Linear")

        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)

    def predict_one(self, features: Tensor) -> float:
        if features.ndim != 1:
            raise ValueError("features must be 1-D")

        with torch.inference_mode():
            value = self(features.unsqueeze(0))

        return float(value[0].item())
