# realtime-isl-recognition
Real-time isl(isolated sign language) recognition project

# data_preprocess

**CSV**: Normalizes raw landmark CSV files (hand + body pose) so they are ready for model training.

**Real-time**: Normalizes raw mediapipe coordination in this order
```
Body -> Left Hand -> Right Hand
```

Each coordinate following this structure
```
# (name, mediapipe_body_index) — None means computed
BODY_JOINTS = [
    ("nose",          0),
    ("neck",          None),   # midpoint calculated from leftShoulder(11) and rightShoulder(12)
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
```
