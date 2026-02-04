import torch
import torch.nn as nn
import torch.nn.functional as F

class ASDSpeechAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # ---------- Encoder ----------
        self.enc_conv1 = nn.Conv1d(49, 128, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)  # halve time dimension

        # ---------- Decoder ----------
        self.dec_conv1 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv1d(128, 49, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (batch, 49, 100)

        # ----- Encoder -----
        x = F.relu(self.enc_conv1(x))     # (batch, 128, 100)
        x = self.pool(x)                  # (batch, 128, 50)

        x = F.relu(self.enc_conv2(x))     # (batch, 256, 50)
        x = self.pool(x)                  # (batch, 256, 25)

        # ----- Decoder -----
        x = F.relu(self.dec_conv1(x))     # (batch, 128, 50)
        x = self.dec_conv2(x)             # (batch, 49, 50)

        # Upsample to original length
        x = F.interpolate(x, size=100, mode="linear", align_corners=False)

        return x
