#!/usr/bin/env python3

import torch
import torch.nn as nn


class SimpleControllerNet(nn.Module):
    def __init__(self, input_dim: int = 22, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.main_stick_head = nn.Linear(hidden_dim, 2)
        self.c_stick_head = nn.Linear(hidden_dim, 2)
        self.shoulder_head = nn.Linear(hidden_dim, 1)
        self.buttons_head = nn.Linear(hidden_dim, 7)

    def forward(self, x: torch.Tensor) -> dict:
        z = self.encoder(x)

        return {
            'main_stick': torch.sigmoid(self.main_stick_head(z)),
            'c_stick': torch.sigmoid(self.c_stick_head(z)),
            'shoulder': torch.sigmoid(self.shoulder_head(z)),
            'buttons': torch.sigmoid(
                self.buttons_head(z)
            ),  # Or use logits if using BCEWithLogitsLoss
        }


if __name__ == '__main__':
    # Example usage
    model = SimpleControllerNet()
    example_input = torch.randn(1, 22)  # Batch size of 1, input dimension of 22
    output = model(example_input)
    print(output)
