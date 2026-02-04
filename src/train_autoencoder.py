import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ASDSpeechDataset
from autoencoder import ASDSpeechAutoencoder

# -------------------------
# Collect .mat files
# -------------------------
data_dir = "../data"
mat_files = [
    os.path.join(data_dir, f)
    for f in os.listdir(data_dir)
    if f.endswith(".mat")
]

# dummy labels (not used)
labels = {
    os.path.splitext(os.path.basename(f))[0]: 0.0
    for f in mat_files
}

# -------------------------
# Dataset + Loader
# -------------------------
dataset = ASDSpeechDataset(mat_files, labels)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

print("Total samples:", len(dataset))

# -------------------------
# Model
# -------------------------
device = torch.device("cpu")
model = ASDSpeechAutoencoder().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -------------------------
# Training
# -------------------------
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for X, _ in loader:
        X = X.to(device)

        optimizer.zero_grad()
        recon = model(X)
        loss = criterion(recon, X)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{epochs}] Reconstruction Loss: {avg_loss:.4f}")

# -------------------------
# Save model
# -------------------------
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../models/autoencoder.pt")
print("Saved autoencoder → models/autoencoder.pt")
