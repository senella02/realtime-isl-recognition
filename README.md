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
    ("nose",             0),
    ("rightEyeInner",    1),
    ("rightEye",         2),
    ("rightEyeOuter",    3),
    ("leftEyeInner",     4),
    ("leftEye",          5),
    ("leftEyeOuter",     6),
    ("rightEar",         7),
    ("leftEar",          8),
    ("mouthRight",       9),
    ("mouthLeft",        10),
    ("rightShoulder",    11),
    ("leftShoulder",     12),
    ("rightElbow",       13),
    ("leftElbow",        14),
    ("rightWrist",       15),
    ("leftWrist",        16),
    ("rightPinky",       17),
    ("leftPinky",        18),
    ("rightIndex",       19),
    ("leftIndex",        20),
    ("rightThumb",       21),
    ("leftThumb",        22),
    ("neck",             None),  # synthetic — midpoint of rightShoulder(11) and leftShoulder(12)
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

## Full workflow

1. M1 ใช้ mediapipe

- ดึงค่าแต่ละเฟรมออกมา ดึง features ออกมา 75 อัน ตามข้างบน
- กำหนดเฟรมเรทว่าจะดึงกี่อัน
- Normalized data ให้เป็น npy ขนาด (1, 150) สำหรับ 1 เฟรม แล้วส่งให้ M3
- ส่งค่า box มาให้ M2 ว่าตอนนี้คนอยู่ตรงไหน เพื่อใช้แสดงผล
- ส่ง timestamp ของแต่ละ frame ไว้คำนวณด้วย ส่งให้ M3

2. M3

- รับเฟรมมาเก็บไว้ใน buffer
- ถ้า status เป็น active ส่งให้ M4
- M3 ส่ง timestamp ของเฟรมแรกใน buffer ให้ M4

3. M4

- คำนวณผล prediction ส่งให้ M2
- คำนวณ RTF ส่งให้ M2

# Run

```
pip install -r requirements.txt
```

```
curl -L -o models/holistic_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
```

## Or on Windows PowerShell:

```
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task" -OutFile "models\holistic_landmarker.task"
```
