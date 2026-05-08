BODY_JOINTS_INPUT_INDEX = [
    ("nose",          0),
    ("neck",          None),   # midpoint of leftShoulder(12) + rightShoulder(11)
    ("leftEye",       5),
    ("rightEye",      2),
    ("leftEar",       8),
    ("rightEar",      7),
    ("leftShoulder",  12),
    ("rightShoulder", 11),
    ("leftElbow",     14),
    ("rightElbow",    13),
    ("leftWrist",     16),
    ("rightWrist",    15),
]

HAND_JOINTS_INPUT_INDEX = [
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

import numpy as np
import pandas as pd


def build_column_names() -> list[str]:
    """Return the 109 column names matching train_normalized.csv: 108 features then 'labels'."""
    cols = []
    for name, _ in BODY_JOINTS_INPUT_INDEX:
        cols += [f"{name}_X", f"{name}_Y"]
    for name, _ in HAND_JOINTS_INPUT_INDEX:
        cols += [f"{name}_0_X", f"{name}_0_Y"]   # left hand
    for name, _ in HAND_JOINTS_INPUT_INDEX:
        cols += [f"{name}_1_X", f"{name}_1_Y"]   # right hand
    cols.append("labels")
    return cols


def buffer_to_dataframe(norm_buf: np.ndarray, label: int) -> pd.DataFrame:
    """
    Wrap a normalized buffer from m3.take_buffer() as a labeled DataFrame.

    take_buffer() already applies select_features + body + hand normalization,
    so norm_buf arrives as (64, 108) float32 — no re-normalization needed.
    The returned DataFrame has 64 rows and 109 columns matching build_column_names()
    (body → left-hand → right-hand → labels), identical to train_normalized.csv.

    :param norm_buf: (64, 108) float32 array from m3.take_buffer()
    :param label:   integer class label for this sign
    :return:        DataFrame shape (64, 109)
    """
    cols = build_column_names()          # 108 feature cols + "labels"
    df = pd.DataFrame(norm_buf, columns=cols[:-1])
    df["labels"] = label
    return df
