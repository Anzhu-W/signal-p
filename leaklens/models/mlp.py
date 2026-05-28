import torch.nn as nn
from leaklens.models.base import SCAModel


class MLPBaseline(SCAModel):
    n_classes = 256
    leakage_model = "identity"
    input_shape = (700,)

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(700, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, 256),
        )

    def forward(self, x):
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        return self.net(x)
