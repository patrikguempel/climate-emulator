import numpy as np

input = np.load("train/npy_data_v2/train_input.npy")
output = np.load("train/npy_data_v2/train_target.npy")

print("Input-shape:" + str(input.shape))
print("Output-shape:" + str(output.shape))