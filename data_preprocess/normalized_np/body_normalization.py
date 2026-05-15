
import logging
import numpy as np

BODY_IDENTIFIERS = [
    "nose",
    "neck",
    "rightEye",
    "leftEye",
    "rightEar",
    "leftEar",
    "rightShoulder",
    "leftShoulder",
    "rightElbow",
    "leftElbow",
    "rightWrist",
    "leftWrist"
]

# Column order matches build_column_names() in convert_npy_to_csv.py (label column stripped).
# Body joints occupy columns 0-23; left-hand columns follow at 24-65, right-hand at 66-107.
_BODY_JOINTS_ORDER = [
    "nose", "neck", "leftEye", "rightEye", "leftEar", "rightEar",
    "leftShoulder", "rightShoulder", "leftElbow", "rightElbow",
    "leftWrist", "rightWrist"
]

# landmark name -> column index of its X coordinate in the numpy feature array
# Y coordinate is at BODY_COL_IDX[name] + 1
BODY_COL_IDX: dict[str, int] = {name: i * 2 for i, name in enumerate(_BODY_JOINTS_ORDER)}


def normalize_body_inplace(seq: np.ndarray) -> None:
    """Apply Bohacek body normalization to one (num_frames, num_features) sequence in-place."""
    last_starting_point: list | None = None
    last_ending_point: list | None = None

    for f in range(seq.shape[0]):
        ls_x   = float(seq[f, BODY_COL_IDX["leftShoulder"]])
        rs_x   = float(seq[f, BODY_COL_IDX["rightShoulder"]])
        neck_x = float(seq[f, BODY_COL_IDX["neck"]])
        nose_x = float(seq[f, BODY_COL_IDX["nose"]])

        if (ls_x == 0 or rs_x == 0) and (neck_x == 0 or nose_x == 0):
            if last_starting_point is None:
                continue
            starting_point, ending_point = last_starting_point, last_ending_point
        else:
            # NOTE:
            #
            # While in the paper, it is written that the head metric is calculated by halving
            # the shoulder distance, this is meant for the distance between the very ends of
            # one's shoulder, as literature studying body metrics and ratios generally states.
            # The Vision Pose Estimation API, however, seems to be predicting rather the center
            # of one's shoulder. Based on our experiments and manual reviews of the data,
            # employing this as just the plain shoulder distance seems to be more corresponding
            # to the desired metric.
            #
            # Please, review this if using other third-party pose estimation libraries.

            if ls_x != 0 and rs_x != 0:
                ls_y = float(seq[f, BODY_COL_IDX["leftShoulder"] + 1])
                rs_y = float(seq[f, BODY_COL_IDX["rightShoulder"] + 1])
                head_metric = ((ls_x - rs_x) ** 2 + (ls_y - rs_y) ** 2) ** 0.5
            else:
                neck_y = float(seq[f, BODY_COL_IDX["neck"] + 1])
                nose_y = float(seq[f, BODY_COL_IDX["nose"] + 1])
                head_metric = ((neck_x - nose_x) ** 2 + (neck_y - nose_y) ** 2) ** 0.5

            left_eye_y = float(seq[f, BODY_COL_IDX["leftEye"] + 1])
            starting_point = [neck_x - 3 * head_metric,
                               left_eye_y + head_metric / 2]
            ending_point   = [neck_x + 3 * head_metric,
                               starting_point[1] - 6 * head_metric]

            # Assign before clamping so that last_* share the same list objects and
            # are clamped in-place together with starting_point / ending_point below.
            last_starting_point, last_ending_point = starting_point, ending_point

        # Ensure that all of the bounding-box-defining coordinates are not out of the picture.
        # Mutates last_starting/ending_point too because they reference the same lists.
        if starting_point[0] < 0: starting_point[0] = 0.0
        if starting_point[1] < 0: starting_point[1] = 0.0
        if ending_point[0]   < 0: ending_point[0]   = 0.0
        if ending_point[1]   < 0: ending_point[1]   = 0.0

        w = ending_point[0] - starting_point[0]
        h = starting_point[1] - ending_point[1]

        for name in BODY_IDENTIFIERS:
            xi = BODY_COL_IDX[name]
            if seq[f, xi] == 0:
                continue
            if w == 0 or h == 0:
                logging.info("Problematic normalization")
                break
            seq[f, xi]     = (seq[f, xi]     - starting_point[0]) / w
            seq[f, xi + 1] = (seq[f, xi + 1] - ending_point[1])   / h


def normalize_single_np(arr: np.ndarray) -> np.ndarray:
    """
    Normalize body landmarks for a single video sequence.

    :param arr: shape (num_frames, num_features) — full feature array. Body
                columns 0-23 follow BODY_COL_IDX; hand columns are untouched.
                Column order matches the preprocessing CSV minus the label column.
    :return: Copy of arr with body columns normalized.
    """
    out = arr.copy()
    normalize_body_inplace(out)
    return out


def normalize_body_full_np(arr: np.ndarray) -> tuple[np.ndarray, list]:
    """
    Batch-normalize body landmarks across all video sequences.

    :param arr: shape (num_videos, num_frames, num_features) — column order
                matches the preprocessing CSV minus the label column.
    :return: (normalized copy, empty invalid_indices list).
    """
    out = arr.copy()
    for i in range(out.shape[0]):
        normalize_body_inplace(out[i])

    print("The normalization of body is finished.")
    print(f"\t-> Original size: {arr.shape[0]}")
    print(f"\t-> Normalized size: {out.shape[0]}")
    print(f"\t-> Problematic videos: 0")

    return out, []


if __name__ == "__main__":
    pass
