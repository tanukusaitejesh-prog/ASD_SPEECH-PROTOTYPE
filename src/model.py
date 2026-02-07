import torch
import torch.nn as nn
import torch.nn.functional as F

class ASDSpeechCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: (batch, 49, 100)
        self.conv1 = nn.Conv1d(
            in_channels=49,
            out_channels=256,
            kernel_size=3
        )

        self.pool1 = nn.MaxPool1d(kernel_size=3)

        self.conv2 = nn.Conv1d(
            in_channels=256,
            out_channels=256,
            kernel_size=3
        )

        # We compute the flattened size dynamically
        self.fc1 = nn.Linear(256 * 30, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 1)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # x: (batch, 49, 100)

        x = F.relu(self.conv1(x))      # → (batch, 256, 98)
        x = self.pool1(x)              # → (batch, 256, 32)

        x = F.relu(self.conv2(x))      # → (batch, 256, 30)

        x = x.view(x.size(0), -1)      # → (batch, 256*30)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))

        x = self.out(x)                # → (batch, 1)
        return x.squeeze(1)
