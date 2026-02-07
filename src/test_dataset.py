from dataset import ASDSpeechDataset

mat_files = ["../data/0ca84b72.mat"]  # one file for test
labels = {"0ca84b72": 14.0}            # dummy ADOS score

ds = ASDSpeechDataset(mat_files, labels)

print("Dataset size:", len(ds))
X, y = ds[0]
print("X shape:", X.shape)  # should be (49,100)
print("y:", y)
