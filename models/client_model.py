import torch.nn as nn
import torch.optim as optim

class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)  

    def forward(self, x):
        return self.model(x)
