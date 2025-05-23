import torch.nn as nn
import torch.optim as optim

class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)  

    def forward(self, x):
        return self.model(x)
