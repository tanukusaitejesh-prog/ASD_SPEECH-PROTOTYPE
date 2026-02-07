import scipy.io as sio
import numpy as np

data = sio.loadmat("../data/0ca84b72.mat")

features = data["features"]

print("features type:", type(features))
print("features shape:", features.shape)
print()

for i in range(features.shape[0]):
    cell = features[i, 0]
    print(f"Cell {i}: type = {type(cell)}")

    if hasattr(cell, "shape"):
        print(f"  shape = {cell.shape}, dtype = {cell.dtype}")
    else:
        print("  no shape attribute")
