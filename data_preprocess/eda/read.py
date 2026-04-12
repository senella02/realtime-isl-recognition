import numpy as np
import glob
import os


def get_avg_min_max(folder="."):
    """Return the average min and max values across all .npy files in folder."""
    npy_files = glob.glob(os.path.join(folder, "*.npy"))
    if not npy_files:
        print("No .npy files found.")
        return

    mins, maxs = [], []
    for path in npy_files:
        data = np.load(path)
        mins.append(data.min())
        maxs.append(data.max())

    avg_min = np.mean(mins).round(4)
    avg_max = np.mean(maxs).round(4)

    print(f"Files processed : {len(npy_files)}")
    print(f"Average min     : {avg_min}")
    print(f"Average max     : {avg_max}")
    return avg_min, avg_max

# Average min/max across all .npy files in the same folder
print()
get_avg_min_max(folder=".")