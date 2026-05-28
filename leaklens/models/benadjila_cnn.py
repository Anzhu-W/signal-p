"""ASCAD CNN_best — 5 conv blocks + 2 dense, identity-256 head. NO BatchNorm.

Architecture from ANSSI-FR/ASCAD/ASCAD_train_models.py.
Filters: 64 -> 128 -> 256 -> 512 -> 512. Kernel size 11. AvgPool(2) after each.
"""
import torch
import torch.nn as nn
from leaklens.models.base import SCAModel


class BenadjilaCNNBest(SCAModel):
    n_classes = 256
    leakage_model = "identity"
    input_shape = (700,)

    def __init__(self):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=11, padding=5),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2),
            )

        self.features = nn.Sequential(
            block(1,   64),
            block(64,  128),
            block(128, 256),
            block(256, 512),
            block(512, 512),
        )
        # 700 → 350 → 175 → 87 → 43 → 21 (floor at each stage)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 21, 4096), nn.ReLU(),
            nn.Linear(4096, 4096),     nn.ReLU(),
            nn.Linear(4096, 256),
        )
        # Match Keras default (glorot_uniform = Xavier uniform).
        # The published Benadjila paper relies on this init; PyTorch's Kaiming default
        # plus the all-ReLU + no-BN + 67M-param + lr=1e-5 combination causes the model
        # to memorise the training set (train_loss -> 0, val_loss explodes).
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, 700) -> (B, 1, 700)
        return self.head(self.features(x))
