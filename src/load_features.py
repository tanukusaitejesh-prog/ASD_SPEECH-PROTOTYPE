import scipy.io as sio
import numpy as np

def load_feature_matrices(mat_path):
    """
    Loads ASDSpeech-style .mat feature files.

    Returns:
        X: np.ndarray of shape (N, 100, 49)
    """
    data = sio.loadmat(mat_path)
    features = data["features"]

    matrices = []

    for i in range(features.shape[0]):
        mat = features[i, 0]

        # Safety check
        if mat.shape != (100, 49):
            raise ValueError(f"Unexpected shape at cell {i}: {mat.shape}")

        matrices.append(mat.astype(np.float32))

    X = np.stack(matrices, axis=0)
    return X


if __name__ == "__main__":
    X = load_feature_matrices("../data/0ca84b72.mat")
    print("Loaded feature tensor shape:", X.shape)
