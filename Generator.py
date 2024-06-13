import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(300, 1000),
            nn.ReLU(),
            nn.LayerNorm(1000),
            nn.Linear(1000, 3 * 128 * 128),
            nn.Sigmoid()
        )

        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, inputs):
        return self.model(inputs)
    
    def train(self, D, inputs, targets):

        g_output = self.forward(inputs)
        d_output = D.forward(g_output)

        loss = D.loss_function(d_output, targets)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()