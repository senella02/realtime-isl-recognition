# CLAUDE.md — realtime-isl-recognition

## Project Overview

Real-time Isolated Sign Language (ISL) recognition system built on the **TSL-ONE-S dataset** (184 Thai Sign Language classes). The project has two phases:

- **Phase 1 (done):** Offline training and evaluation of deep learning models (LSTM, SPOTER) on preprocessed landmark sequences. Best model selected and frozen.
- **Phase 2 (current):** Embed the trained model into a live webcam pipeline that segments, normalizes, and classifies signs in real time.

The core constraint: **normalization at inference must be byte-for-byte identical to training**. Any drift kills accuracy.

---

## Repository Structure

```
realtime-isl-recognition/
├── data_preprocess/
│   ├── main.py                          # Batch preprocessing entry point
│   ├── requirements.txt                 # pandas>=2.0
│   ├── normalized_csv/
│   │   ├── body_normalization.py        # Bohacek body normalization (CSV + real-time)
│   │   └── hand_normalization.py        # Bohacek hand normalization (CSV + real-time)
│   ├── npy_to_csv/
│   │   └── convert_npy_to_csv.py
│   ├── output_csv/
│   │   ├── label_map.json               # 184-class gloss↔int mapping
│   │   ├── train_normalized.csv
│   │   └── test_normalized.csv
│   └── eda/
│       ├── inspect_frame.py
│       └── read.py
└── CLAUDE.md
```

**Phase 2 modules to be added (not yet created):**

| Module | Responsibility |
|--------|---------------|
| `M1/`  | Landmark extraction (MediaPipe), adaptive bounding box |
| `M3/`  | State machine, frame buffer, trigger logic |
| `M4/`  | Model loading, inference, optimization |
| `M2/`  | Overlay rendering, latency measurement, evaluation |

---

## Dataset

- **Name:** TSL-ONE-S
- **Classes:** 184 Thai Sign Language glosses (countries, landmarks, numbers 0–10, Thai consonants/vowels, family terms, sports, food)
- **Label map:** `data_preprocess/output_csv/label_map.json` — use `gloss_to_int` and `int_to_gloss` keys
- **Raw data:** hosted externally (Google Drive); not in repo due to size
- **Landmarks:** 75 points per frame extracted via MediaPipe (pose + both hands)

---

## Normalization — Critical Constraint

**Never reimplement normalization.** Always reuse the existing code in `data_preprocess/normalized_csv/`.

### Body Normalization (`body_normalization.py`)

- **Algorithm:** Bohacek-normalization
- **12 body landmarks:** `nose, neck, rightEye, leftEye, rightEar, leftEar, rightShoulder, leftShoulder, rightElbow, leftElbow, rightWrist, leftWrist`
- **Head metric:** shoulder distance (when both shoulders visible) OR neck–nose distance (fallback)
- **Bounding box:** centered on neck, spans ±3× head_metric horizontally, 6× head_metric vertically
- **Fallback:** if no anchor landmarks in a frame, carries forward `last_starting_point`

**For real-time use:** call `normalize_single_dict(row: dict)` where `row` maps landmark names (e.g. `"leftShoulder"`) to a list of `[x, y, ...]` per frame.

### Hand Normalization (`hand_normalization.py`)

- **21 hand landmarks per hand:** wrist + 4 fingers × 4 joints + thumb × 4 joints
- **Column naming convention (CSV):** `{landmark}_{0|1}_{X|Y}` where 0=left, 1=right (converted from `_left_`/`_right_` prefixes on load)
- **Bounding box:** tight box around all non-zero landmarks with 10% padding; square-padded so aspect ratio is 1:1
- **Skips frames** where all landmark values are zero (no hand detected)

**For real-time use:** call `normalize_single_dict(row: dict)` where `row` maps e.g. `"wrist_0"` to a list of `[x, y]` per frame.

---

## Real-Time Pipeline (Phase 2)

### End-to-End Flow

```
Webcam frame
  → [M1] MediaPipe landmark extraction (75 points)
  → [M1] Adaptive body bounding box + crop
  → [M3] Compute hand motion score → classify frame: Idle (H) / Active (A)
  → [M3] State machine update:
           Idle→Active: TA consecutive active frames → start buffering
           Active→Idle: TR consecutive idle frames  → trigger inference
  → [M3] Sliding buffer (64 frames, fills only during active signing)
  → [M1] SPOTER normalization (body_normalization + hand_normalization, reused exactly)
  → [M4] Model inference → class probabilities
  → [M4] Extract top-3 predictions
  → [M2] Render overlay (bbox, state, top-3 + confidence, FPS)
  → [M2] Log latency metrics
  → [M3] Reset buffer → Idle
```

### State Machine Parameters

| Parameter | Meaning |
|-----------|---------|
| `TA` | Consecutive active frames required to start recording |
| `TR` | Consecutive idle frames required to trigger inference |

These are tunable hyperparameters — default values TBD from calibration.

### Buffer

- Fixed length: **64 frames**
- Fills only while state == Active
- Finalized (sent to inference) when Active→Idle transition fires

---

## Performance Targets

| Metric | Target |
|--------|--------|
| End-to-end latency | < 500 ms |
| Inference latency | < 100 ms |
| Real-time factor (RTF) | < 1.0 |
| Live accuracy | Comparable to Phase 1 offline accuracy |

---

## Models

- **SPOTER** — transformer-based skeleton pose recognizer (primary)
- **LSTM** — sequence model (baseline / fallback)
- Model selection frozen at end of Phase 1. Do not retrain in Phase 2.
- Load from checkpoint; inference only.

---

## Development Rules

1. **Do not touch normalization logic.** If a bug is found in `body_normalization.py` or `hand_normalization.py`, fix it in the existing file — do not create a parallel version. Any change must be validated against offline accuracy first.

2. **Reuse `normalize_single_dict`** (not `normalize_*_full`) for real-time inference. The `*_full` variants operate on full DataFrames and are for batch preprocessing only.

3. **Label map is the source of truth.** Always load `output_csv/label_map.json` for class index↔gloss mappings. Do not hardcode class names.

4. **Latency must be measured explicitly.** M2 is responsible for end-to-end timing. Log per-inference latency to verify RTF < 1.0 in live conditions.

5. **Inference is event-driven, not frame-wise.** The model runs once per detected sign, not once per frame. Do not add per-frame prediction.

6. **Module boundaries are real.** M1/M2/M3/M4 are separate concerns. Keep state machine logic in M3, not scattered across M1 or M4.

---

## Setup

```bash
pip install -r data_preprocess/requirements.txt
# Additional Phase 2 deps (to be specified): mediapipe, opencv-python, torch
```

Raw dataset: download from Google Drive (link in `data_preprocess/README.md`), place in project root, ensure paths match calls to `preprocess()`.

---

## Current Status (Phase 2 start)

- [x] Phase 1 complete: models trained and evaluated offline
- [x] Normalization pipeline implemented and validated
- [x] Label map and preprocessed CSVs ready
- [ ] M1: MediaPipe landmark extractor
- [ ] M3: State machine + frame buffer
- [ ] M4: Model loader + inference wrapper
- [ ] M2: Overlay renderer + latency logger
- [ ] Integration: end-to-end pipeline
- [ ] Evaluation: latency benchmarks, live accuracy comparison
