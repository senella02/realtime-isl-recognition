"""
convert_npy_to_csv.py
---------------------
Tasks 2, 3, 4 of the data preparation plan:
  2 - Build gloss-to-integer mapping from tsl_one_s_id_to_label.json
  3 - Extract and remap landmarks from .npy files to SPOTER column format
  4 - Split rows into train_raw.csv and test_raw.csv based on the 'split' field

Flat npy layout (frames, 150):
  [0:66]    33 body landmarks  (x,y pairs, MediaPipe ordering)
  [66:108]  21 left hand joints (x,y pairs, MediaPipe ordering)
  [108:150] 21 right hand joints (x,y pairs, MediaPipe ordering)

After reshape to (frames, 75, 2):
  joints 0-32   → body
  joints 33-53  → left hand
  joints 54-74  → right hand
"""

import json
import os
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__)) # Cuurent file path
NPY_DIR      = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "TSL-ONE-Pose"))
REPO_ROOT    = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

ID_TO_LABEL_PATH = os.path.join(REPO_ROOT, "tsl_one_s_id_to_label.json")
DATASET_PATH     = os.path.join(REPO_ROOT, "tsl_one_s_dataset.json")

OUT_TRAIN    = os.path.join(REPO_ROOT, "train_raw.csv")
OUT_TEST     = os.path.join(REPO_ROOT, "test_raw.csv")
LABEL_MAP_OUT = os.path.join(REPO_ROOT, "label_map.json")

# ── frame normalisation ────────────────────────────────────────────────────────
TARGET_FRAMES = 64

# ── joint definitions ──────────────────────────────────────────────────────────
# (name, mediapipe_body_index) — None means computed
BODY_JOINTS = [
    ("nose",          0),
    ("neck",          None),   # midpoint of leftShoulder(11) + rightShoulder(12)
    ("leftEye",       2),
    ("rightEye",      5),
    ("leftEar",       7),
    ("rightEar",      8),
    ("leftShoulder",  11),
    ("rightShoulder", 12),
    ("leftElbow",     13),
    ("rightElbow",    14),
    ("leftWrist",     15),
    ("rightWrist",    16),
]

HAND_JOINTS = [
    ("wrist",     0),
    ("thumbCMC",  1),
    ("thumbMP",   2),
    ("thumbIP",   3),
    ("thumbTip",  4),
    ("indexMCP",  5),
    ("indexPIP",  6),
    ("indexDIP",  7),
    ("indexTip",  8),
    ("middleMCP", 9),
    ("middlePIP", 10),
    ("middleDIP", 11),
    ("middleTip", 12),
    ("ringMCP",   13),
    ("ringPIP",   14),
    ("ringDIP",   15),
    ("ringTip",   16),
    ("littleMCP", 17),
    ("littlePIP", 18),
    ("littleDIP", 19),
    ("littleTip", 20),
]

# ── Task 2.5: normalise sequence length ───────────────────────────────────────
def normalize_frames(arr, target=TARGET_FRAMES):
    """
    Resample a (frames, 75, 2) array to exactly `target` frames using
    per-joint linear interpolation along the time axis.

    Strategy
    --------
    frames == target  → returned unchanged
    frames  > target  → downsampled (linear interp)
    frames  < target  → upsampled  (linear interp)

    Only numpy is required — no extra dependencies.
    """
    n = arr.shape[0]
    if n == target:
        return arr

    src_idx  = np.arange(n, dtype=np.float64)
    dst_idx  = np.linspace(0, n - 1, num=target)
    out      = np.empty((target, arr.shape[1], arr.shape[2]), dtype=arr.dtype)

    for j in range(arr.shape[1]):       # each joint
        for c in range(arr.shape[2]):   # x, y
            out[:, j, c] = np.interp(dst_idx, src_idx, arr[:, j, c])

    return out


# ── CSV column order ───────────────────────────────────────────────────────────
def build_column_names():
    cols = ["labels"]
    for name, _ in BODY_JOINTS:
        cols += [f"{name}_X", f"{name}_Y"]
    for name, _ in HAND_JOINTS:
        cols += [f"{name}_left_X", f"{name}_left_Y"]   # left hand (_0)
    for name, _ in HAND_JOINTS:
        cols += [f"{name}_right_X", f"{name}_right_Y"]   # right hand (_1)
    return cols


# ── Task 2: build gloss-to-int mapping ────────────────────────────────────────
def build_label_map(id_to_label_path):
    """
    id_to_label.json: {"0": "Country", "1": "Thailand", ...}
    Labels in CSV = id (0-based)
    Returned:
      gloss_to_int : {"Country": 0, "Thailand": 1, ...}
      int_to_gloss : {0: "Country", 1: "Thailand", ...}
    """
    with open(id_to_label_path, encoding="utf-8") as f:
        raw = json.load(f)   # {str_id: gloss}

    gloss_to_int = {}
    int_to_gloss = {}
    for str_id, gloss in raw.items():
        label = int(str_id)
        gloss_to_int[gloss] = label
        int_to_gloss[label] = gloss

    return gloss_to_int, int_to_gloss


# ── Task 3: extract and remap landmarks ───────────────────────────────────────
def extract_row(npy_path, label):
    """
    Load one .npy file (frames, 150), remap joints to SPOTER layout,
    and return a dict {column_name: str_list_of_floats}.
    Returns None if the file is missing or malformed.
    """
    if not os.path.exists(npy_path):
        return None

    arr = np.load(npy_path)            # (frames, 150)
    if arr.ndim != 2 or arr.shape[1] != 150:
        print(f"  [SKIP] unexpected shape {arr.shape}: {npy_path}")
        return None

    arr = arr.reshape(arr.shape[0], 75, 2)   # (frames, 75, 2)
    arr = normalize_frames(arr)              # → (TARGET_FRAMES, 75, 2)
    body       = arr[:, 0:33,  :]            # (TARGET_FRAMES, 33, 2)
    left_hand  = arr[:, 33:54, :]            # (TARGET_FRAMES, 21, 2)
    right_hand = arr[:, 54:75, :]            # (TARGET_FRAMES, 21, 2)

    row = {"labels": label}

    # body joints
    for name, mp_idx in BODY_JOINTS:
        if mp_idx is not None:
            xs = body[:, mp_idx, 0].tolist()
            ys = body[:, mp_idx, 1].tolist()
        else:
            # neck = midpoint of leftShoulder(11) + rightShoulder(12)
            xs = ((body[:, 11, 0] + body[:, 12, 0]) / 2.0).tolist()
            ys = ((body[:, 11, 1] + body[:, 12, 1]) / 2.0).tolist()
        row[f"{name}_X"] = str(xs)
        row[f"{name}_Y"] = str(ys)

    # left hand joints (_0)
    for name, mp_idx in HAND_JOINTS:
        row[f"{name}_left_X"] = str(left_hand[:, mp_idx, 0].tolist())
        row[f"{name}_left_Y"] = str(left_hand[:, mp_idx, 1].tolist())

    # right hand joints (_1)
    for name, mp_idx in HAND_JOINTS:
        row[f"{name}_right_X"] = str(right_hand[:, mp_idx, 0].tolist())
        row[f"{name}_right_Y"] = str(right_hand[:, mp_idx, 1].tolist())

    return row


# ── Task 4: build and split CSVs ──────────────────────────────────────────────
def main():
    print("Loading label map …")
    gloss_to_int, int_to_gloss = build_label_map(ID_TO_LABEL_PATH)
    print(f"  {len(gloss_to_int)} glosses loaded")

    # save label_map.json (Task 2 output)
    label_map = {
        "gloss_to_int": gloss_to_int,
        "int_to_gloss": {str(k): v for k, v in int_to_gloss.items()},
    }
    with open(LABEL_MAP_OUT, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"  Saved {LABEL_MAP_OUT}")

    print("\nLoading dataset JSON …")
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)   # list of {gloss, instances:[{video_id, split, region}]}

    # flatten to video_id → (gloss, split)
    video_info = {}
    for entry in dataset:
        gloss = entry["gloss"]
        for inst in entry["instances"]:
            video_info[inst["video_id"]] = (gloss, inst["split"])
    print(f"  {len(video_info)} video instances indexed")

    columns = build_column_names()
    train_rows, test_rows = [], []
    skipped_no_npy, skipped_no_label, skipped_no_info = 0, 0, 0

    npy_files = [f for f in os.listdir(NPY_DIR) if f.endswith(".npy")]
    print(f"\nProcessing {len(npy_files)} .npy files …")

    for i, fname in enumerate(npy_files, 1):
        if i % 500 == 0:
            print(f"  {i}/{len(npy_files)} …")

        video_id = fname[:-4]   # strip '.npy'

        if video_id not in video_info:
            skipped_no_info += 1
            continue

        gloss, split = video_info[video_id]

        if gloss not in gloss_to_int:
            skipped_no_label += 1
            print(f"  [WARN] gloss '{gloss}' not in id_to_label: {fname}")
            continue

        label = gloss_to_int[gloss]
        npy_path = os.path.join(NPY_DIR, fname)
        row = extract_row(npy_path, label)

        if row is None:
            skipped_no_npy += 1
            continue

        # val goes into train (common practice; plan only defines train / test)
        if split in ("train", "val"):
            train_rows.append(row)
        elif split == "test":
            test_rows.append(row)

    print(f"\nDone processing.")
    print(f"  train rows : {len(train_rows)}")
    print(f"  test rows  : {len(test_rows)}")
    print(f"  skipped (no dataset info) : {skipped_no_info}")
    print(f"  skipped (no npy file)     : {skipped_no_npy}")
    print(f"  skipped (gloss not mapped): {skipped_no_label}")

    train_df = pd.DataFrame(train_rows, columns=columns)
    test_df  = pd.DataFrame(test_rows,  columns=columns)

    train_df.to_csv(OUT_TRAIN, index=False)
    test_df.to_csv(OUT_TEST,   index=False)
    print(f"\nSaved {OUT_TRAIN}")
    print(f"Saved {OUT_TEST}")

    # label distribution check
    print("\nTrain label distribution (top 10):")
    print(train_df["labels"].value_counts().head(10).to_string())
    print("\nTest label distribution (top 10):")
    print(test_df["labels"].value_counts().head(10).to_string())
    print(f"\nUnique labels — train: {train_df['labels'].nunique()}, test: {test_df['labels'].nunique()}")


if __name__ == "__main__":
    main()
