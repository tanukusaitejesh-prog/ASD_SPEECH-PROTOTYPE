import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import os

class ASDSpeechDataset(Dataset):
    def __init__(self, mat_files, labels):
        """
        mat_files: list of .mat paths
        labels: dict {recording_id: dummy_label}
        """
        self.samples = []

        for mat_path in mat_files:
            data = sio.loadmat(mat_path)
            feats = data["features"]  # (10,1) cell array

            rec_id = os.path.splitext(os.path.basename(mat_path))[0]
            label = labels.get(rec_id, 0.0)

            for i in range(feats.shape[0]):
                X = feats[i, 0].astype(np.float32)  # (100,49)
                self.samples.append((X, label))

        # ---- compute normalization stats once ----
        all_X = []
        for X, _ in self.samples:
            t = torch.tensor(X).T  # (49,100)
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            all_X.append(t)

        all_X = torch.stack(all_X)  # (N,49,100)

        self.mean = all_X.mean(dim=(0, 2), keepdim=True)
        self.std = all_X.std(dim=(0, 2), keepdim=True) + 1e-6

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]

        X = torch.tensor(X, dtype=torch.float32).T  # (49,100)

        # ---- sanitize ----
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- normalize ----
        X = (X - self.mean.squeeze(0)) / self.std.squeeze(0)

        y = torch.tensor(y, dtype=torch.float32)
        return X, y
