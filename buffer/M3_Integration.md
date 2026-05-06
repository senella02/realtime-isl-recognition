# M3 Integration — Change Log

**Date:** 2026-05-06
**Scope:** Integrate `realtime_engine.py` (M3 Phase 2) into `realtime-isl-recognition`

---

## What Was Done

### Files Created

| File | Purpose |
|---|---|
| `M3/__init__.py` | Package init — exposes `M3StateMachine` |
| `M3/realtime_engine.py` | Copy of `../realtime_engine.py` (source of truth for state machine logic) |
| `M3/engine.py` | `M3StateMachine` — contracts-compatible wrapper |
| `new.md` | This file |

### Files NOT Touched

Everything else is unchanged: `contract/contracts.py`, `ui_render/`, `data_preprocess/`, `main.py`.

---

## Architecture

### Why two files inside M3/

`realtime_engine.py` is the raw state machine with its own `State` enum and `TriggerEvent` dataclasses. It works standalone and owns all error instrumentation.

`engine.py` is the thin adapter that:
1. Wraps `RealtimeEngine` behind the `M3StateMachine` interface the pipeline expects
2. Converts between `RealtimeEngine`'s internal types and the shared contracts (`FramePacket`, `StateUpdate`, `SignEvent`)
3. Computes the H/A classification from landmark motion (since `FramePacket` has no `is_active` field yet)

This keeps the original state machine code unmodified so error analysis stays reproducible.

---

## How to Use (from `pipeline/run.py` or `main.py`)

```python
from M3 import M3StateMachine

m3 = M3StateMachine(ta=5, tr=10)   # TA/TR are tunable hyperparameters

while running:
    packet = m1.extract(raw_frame, capture_ts)       # FramePacket from M1
    state_update = m3.update(packet)                  # StateUpdate every frame

    prediction = None
    if state_update.triggered:
        buffer = m3.take_buffer()                     # list of landmarks_normalized dicts
        prediction = m4.infer(state_update.sign_event, buffer)

    m2.render(packet, state_update, prediction)
    m2.log(packet, state_update, prediction)
```

---

## H/A Classification

`FramePacket` (contract v1.0.0) has no `is_active` field. M3 computes it internally using `_compute_is_active()`:

- Compares `landmarks_normalized` between consecutive frames
- Sums mean absolute displacement of hand landmarks (keys ending in `_0` or `_1`)
- Classifies as Active if displacement > `motion_threshold` (default 0.02, normalized units)

**When to update:** If M1 is extended to add `is_active: bool` to `FramePacket`, replace the body of `_compute_is_active()` with `return packet.is_active`. No other change needed.

**Tuning:** Pass `motion_threshold=` to `M3StateMachine(...)` to calibrate for your environment.

---

## What Goes Into the Buffer

Each element in the 64-frame buffer is `packet.landmarks_normalized` — the dict of normalized landmark coordinates produced by M1. This is exactly what SPOTER/M4 expects for inference.

If `landmarks_normalized` is `None` (MediaPipe failed on a frame), M3 classifies that frame as Idle and nothing is added to the buffer. M4 never receives `None` entries.

---

## Error Instrumentation

All error-tracking from `realtime_engine.py` is preserved and forwarded:

```python
# After a live session:
m3.save_trigger_log("trigger_error_log.csv")   # per-segment CSV

summary = m3.error_summary()
# keys: false_start_count, false_start_rate_pct,
#       missed_sign_count, missed_sign_rate_pct,
#       premature_trigger_count, premature_trigger_rate_pct,
#       over_segmentation_count, over_segmentation_rate_pct,
#       total_inference_calls, total_signs_gt, ta, tr

# Ground-truth count for rate calculations:
m3.set_total_signs(n)

# M2 evaluator calls these after labeling:
m3.mark_false_start()
m3.mark_missed_sign(frame_start, frame_end)
```

---

## TR/TA Threshold Sweep

The sweep utility is in `M3/realtime_engine.py`. To fill `boundary_error_summary.md`:

```python
from M3.realtime_engine import sweep_thresholds, print_sensitivity_table

# frames_with_labels: list of (landmarks_normalized_dict, is_active_bool)
results = sweep_thresholds(frames_with_labels, ground_truth_signs=N)
print_sensitivity_table(results)
```

---

## What Still Needs to Happen (not M3's job)

| Item | Owner |
|---|---|
| `pipeline/run.py` — wire all modules together | Shared / team |
| M1 `extract()` returning `FramePacket` with `landmarks_normalized` | M1 |
| M4 `infer(sign_event, buffer)` returning `Prediction` | M4 |
| Populate `boundary_error_summary.md` after real test runs | M3 (Prem) |
| Co-author `error_analysis.md` Part B | M3 + M2 |
