import torch
from model import ASDSpeechCNN

model = ASDSpeechCNN()

# Fake batch: 8 samples
x = torch.randn(8, 49, 100)

y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
