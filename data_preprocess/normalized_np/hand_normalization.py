"""
normalized_np/hand_normalization.py
------------------------------------

Numpy-array implementation of Bohacek hand normalization.

Algorithm is identical to normalize_single_dict() in
normalized_csv/hand_normalization.py.  Only the data-access layer changes:
instead of a dict-of-lists the input is a (num_frames, num_features) ndarray.

Expected column layout (produced by select_features() in main.py):
    cols  0:24  — 12 body joints  × (x, y)
    cols 24:66  — 21 left-hand joints × (x, y)  in MediaPipe order
    cols 66:108 — 21 right-hand joints × (x, y) in MediaPipe order
"""

import numpy as np

_N_HAND_JOINTS = 21
_HAND_STRIDE   = _N_HAND_JOINTS * 2  # 42 values per hand

LEFT_HAND_START  = 24
RIGHT_HAND_START = 66


def normalize_hand_inplace(seq: np.ndarray, hand_start: int) -> None:
    """
    Apply Bohacek hand normalization to one hand's columns in-place.

    :param seq: (num_frames, num_features) array
    :param hand_start: flat column index where this hand's 42 values begin
    """
    x_cols = np.arange(hand_start, hand_start + _HAND_STRIDE, 2, dtype=np.intp)
    y_cols = x_cols + 1

    for f in range(seq.shape[0]):
        xs = seq[f, x_cols]
        ys = seq[f, y_cols]

        # Bounding box uses separate non-zero masks for x and y,
        # matching normalize_single_dict exactly.
        mask_x = xs != 0
        mask_y = ys != 0

        if not np.any(mask_x) or not np.any(mask_y):
            continue

        min_x, max_x = float(xs[mask_x].min()), float(xs[mask_x].max())
        min_y, max_y = float(ys[mask_y].min()), float(ys[mask_y].max())

        width  = max_x - min_x
        height = max_y - min_y

        if width > height:
            delta_x = 0.1 * width
            delta_y = delta_x + (width - height) * 0.5
        else:
            delta_y = 0.1 * height
            delta_x = delta_y + (height - width) * 0.5

        start_x = min_x - delta_x
        start_y = min_y - delta_y
        end_x   = max_x + delta_x
        end_y   = max_y + delta_y

        span_x = end_x - start_x
        span_y = end_y - start_y

        if span_x == 0 or span_y == 0:
            continue

        # Per-landmark skip: x == 0 means not detected (mirrors normalize_single_dict)
        active = mask_x
        seq[f, x_cols[active]] = (seq[f, x_cols[active]] - start_x) / span_x
        seq[f, y_cols[active]] = (seq[f, y_cols[active]] - start_y) / span_y


def normalize_hands_inplace(seq: np.ndarray) -> None:
    """
    Apply Bohacek hand normalization to both hands in-place.

    :param seq: (num_frames, 108) array produced by select_features()
    """
    normalize_hand_inplace(seq, LEFT_HAND_START)
    normalize_hand_inplace(seq, RIGHT_HAND_START)


def normalize_single_np(arr: np.ndarray) -> np.ndarray:
    """
    Normalize hand landmarks for a single video sequence.

    :param arr: (num_frames, num_features)
    :return: Copy with hand columns normalized.
    """
    out = arr.copy()
    normalize_hands_inplace(out)
    return out


def normalize_hands_full_np(arr: np.ndarray) -> np.ndarray:
    """
    Batch-normalize hand landmarks across all video sequences.

    :param arr: (num_videos, num_frames, num_features)
    :return: Normalized copy.
    """
    out = arr.copy()
    for i in range(out.shape[0]):
        normalize_hands_inplace(out[i])
    return out


if __name__ == "__main__":
    pass
