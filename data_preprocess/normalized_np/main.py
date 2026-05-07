"""
normalized_np/main.py
---------------------

Input: Flat npy layout (frames, 130):
  [0:46]    23 body landmarks  (x,y pairs, MediaPipe ordering)
  [46:88]   21 left hand joints (x,y pairs, MediaPipe ordering)
  [88:130]  21 right hand joints (x,y pairs, MediaPipe ordering)

After reshape to (frames, 108):
  cols  0-23  → 12 selected body joints (x,y)
  cols 24-65  → left hand (all 21 joints, x,y)
  cols 66-107 → right hand (all 21 joints, x,y)
"""

import numpy as np
from body_normalization import normalize_body_inplace
from hand_normalization import normalize_hands_inplace

# New 23-joint MediaPipe body layout:
#   0:nose  1:rightEyeInner  2:rightEye   3:rightEyeOuter
#   4:leftEyeInner  5:leftEye  6:leftEyeOuter
#   7:rightEar  8:leftEar  9:mouthRight  10:mouthLeft
#   11:rightShoulder  12:leftShoulder
#   13:rightElbow  14:leftElbow  15:rightWrist  16:leftWrist
#   17:rightPinky  18:leftPinky  19:rightIndex  20:leftIndex
#   21:rightThumb  22:leftThumb

# (name, mediapipe_body_index) — None means computed
# output features should be in this order: [nose_x, nose_y, neck_x, neck_y, ...]
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

# Reuse for both hands
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

# ---------------------------------------------------------------------------
# Precomputed flat indices — built once at import, reused every frame
# ---------------------------------------------------------------------------

# Body joints excluding neck (which is computed as a midpoint).
# Order matches BODY_JOINTS_INPUT_INDEX with neck omitted:
#   nose, leftEye, rightEye, leftEar, rightEar,
#   leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist
_BODY_MP_INDICES = [mp for _, mp in BODY_JOINTS_INPUT_INDEX if mp is not None]
_BODY_IDX = np.array(
    [i for mp in _BODY_MP_INDICES for i in (2 * mp, 2 * mp + 1)], dtype=np.intp
)

# leftShoulder (MP 12) and rightShoulder (MP 11) flat indices — for neck midpoint
_LS_IDX = np.array([24, 25], dtype=np.intp)
_RS_IDX = np.array([22, 23], dtype=np.intp)


def select_features(frames: np.ndarray) -> np.ndarray:
    """
    Select 54 joints (108 values) from a batch of flat (N, 130) MediaPipe frames.

    Input layout  (N, 130): [body x,y = 23 | left-hand x,y = 21 | right-hand x,y = 21]
    Output layout (N, 108): [body x,y = 12 | left-hand x,y = 21 | right-hand x,y = 21]

    Body output order: nose, neck*, leftEye, rightEye, leftEar, rightEar,
                       leftShoulder, rightShoulder, leftElbow, rightElbow,
                       leftWrist, rightWrist
    * neck = midpoint(leftShoulder, rightShoulder)
    """
    # 11 selected body joints → (N, 22) values (nose first, then leftEye…rightWrist)
    body = frames[:, _BODY_IDX]

    # neck = midpoint of leftShoulder and rightShoulder → (N, 2)
    neck = (frames[:, _LS_IDX] + frames[:, _RS_IDX]) * 0.5

    # hands: left [46:88] + right [88:130] → (N, 84) values (all joints, unchanged)
    hands = frames[:, 46:130]

    # assemble: [nose(2) | neck(2) | leftEye…rightWrist(20) | hands(84)]
    out = np.empty((frames.shape[0], 108), dtype=frames.dtype)
    out[:, 0:2]    = body[:, 0:2]   # nose
    out[:, 2:4]    = neck            # neck (computed)
    out[:, 4:24]   = body[:, 2:]    # leftEye … rightWrist
    out[:, 24:108] = hands
    return out


def normalized_batch(frames: np.ndarray) -> np.ndarray:
    """
    Select 54 joints and normalize body landmarks for a batch of frames.

    :param frames: (N, 130) raw MediaPipe flat array
    :return: (N, 108) selected + body-normalized array
    """
    out = select_features(frames)   # already a fresh allocation — no copy needed
    normalize_body_inplace(out)
    normalize_hands_inplace(out)
    return out
