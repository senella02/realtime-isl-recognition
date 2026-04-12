import numpy as np

FILE = "../TSL-ONE-Pose/01_01_0001.npy"
FRAME_INDEX = 0          # change this to inspect a different frame

data = np.load(FILE, allow_pickle=True)
num_frames, num_values = data.shape
num_keypoints = num_values // 2   # each keypoint has (x, y)

print(f"File        : {FILE}")
print(f"Type        : {type(data)}")
print(data.shape)
print(data.dtype)
print(f"Total frames: {num_frames}")
print(f"Keypoints   : {num_keypoints}  (x, y per keypoint)")
print(f"Frame index : {FRAME_INDEX}")
print(data[:1])

frame = data[FRAME_INDEX]         # shape: (150,)
coords = frame.reshape(num_keypoints, 2)   # shape: (75, 2)

print(f"{'Index':<8} {'X':>12} {'Y':>12}")
print("-" * 34)
for i, (x, y) in enumerate(coords):
    print(f"{i:<8} {x:>12.6f} {y:>12.6f}")
