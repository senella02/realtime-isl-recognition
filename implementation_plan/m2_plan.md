# M2 — Implementation Plan

**Module:** M2 (Output + Evaluation)
**Owner:** M2 lead / senior project manager
**Pipeline position:** `[M1: Input] → [M3: Timing] → [M4: Model] → [M2: Output + Evaluation]`
**One-line scope:** Convert predictions into visible output, measure real-time performance, and produce evidence of real-world model behavior.

---

## 0. How to Read This Document

Section 2 is the **integration contract** — every other team member should read it. Sections 3–7 are M2-internal: file layout, phased plan, testing, risks, exit criteria. Treat each phase checklist as the definition of done.

---

## 1. Scope

| Allowed | Not Allowed |
|---|---|
| Read outputs of M1, M3, M4 | Modify M1 landmark extraction or normalization |
| Render UI overlay | Modify M3 state machine / segmentation |
| Measure latency, FPS, RTF | Modify M4 model or inference |
| Run live data collection | Retrain models |
| Compute confusion matrices, per-class F1 | Touch `body_normalization.py` or `hand_normalization.py` |
| Author classification section of `error_analysis.md` | Patch upstream bugs (only diagnose + report them) |

**Failure handling principle:** when accuracy degrades, M2 *localizes* the cause (M1 / M3 / M4) with metrics and examples. M2 does not fix upstream code.

---

## 2. Inter-Module Contracts

This is the binding interface. All teams must agree before coding starts. Lives in a shared file: `pipeline/contracts.py` (proposed).

### 2.1 Architecture decision — direct calls, single-threaded

The pipeline runs as **one synchronous loop**, not a multi-threaded producer/consumer system. Each module exposes plain functions/methods; the main loop calls them in order and passes the dataclasses below as arguments. There is **no queue, no event bus, no worker thread**.

Rationale: this is a one-time-use, single-webcam, single-user demo. The only parallelism a queue would buy is overlapping inference with frame capture, but inference fires once per sign (event-driven, ~100 ms) and the user pauses naturally between signs. The cost of a queue (threading, shutdown coordination, race conditions, harder debugging) outweighs the benefit. See §7 for the residual risks this creates and how they're handled.

The dataclasses in §2.2 remain the binding contract — they are the shape of arguments and return values between modules, regardless of transport.

### 2.2 Shared types

```python
# pipeline/contracts.py
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np

class SignState(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"

@dataclass
class FramePacket:
    """Returned by M1 once per webcam frame. Consumed by M3 and M2."""
    frame_id: int                     # monotonically increasing
    capture_ts: float                 # time.perf_counter() at frame grab
    image_bgr: np.ndarray             # raw webcam frame (for M2 overlay)
    bbox: Optional[tuple]             # (x1, y1, x2, y2) in image coords; None if no body
    landmarks_raw: Optional[dict]     # 75-point dict; M2 may use for debug overlay
    landmarks_normalized: Optional[dict]  # post body+hand normalization; consumed by M3/M4

@dataclass
class StateUpdate:
    """Returned by M3 once per frame. Consumed by M2."""
    frame_id: int
    state: SignState
    active_frame_count: int           # frames since active onset (0 when idle)
    sign_id: Optional[int]            # set during ACTIVE; same id from onset to trigger
    triggered: bool                   # True on the frame that completes Active→Idle
    sign_event: Optional["SignEvent"] # populated iff triggered == True

@dataclass
class SignEvent:
    """Produced by M3 on Active→Idle. Passed directly to M4. Also surfaced to M2 via StateUpdate."""
    sign_id: int
    sign_start_ts: float              # capture_ts of first ACTIVE frame
    sign_end_ts: float                # capture_ts of last ACTIVE frame
    sign_duration_s: float            # sign_end_ts - sign_start_ts
    buffer_length: int                # number of frames in the 64-frame buffer used

@dataclass
class Prediction:
    """Returned by M4 once per SignEvent. Passed directly to M2."""
    sign_id: int                      # matches SignEvent.sign_id
    inference_start_ts: float
    inference_end_ts: float
    probs: np.ndarray                 # shape (184,), softmax probabilities
    top_k_indices: list               # length 3, sorted desc by prob
    top_k_probs: list                 # length 3, aligned with indices
    top_k_glosses: list               # length 3, resolved via label_map.json
```

### 2.3 Call graph (the main loop)

```python
# pipeline/run.py — pseudocode, the canonical wiring
m1 = M1Extractor(...)
m3 = M3StateMachine(TA=..., TR=..., buffer_size=64)
m4 = M4Inferencer(checkpoint_path=...)
m2 = M2Output(label_map=..., display=True)

while running:
    raw_frame, capture_ts = m1.grab()                        # webcam read
    packet: FramePacket = m1.extract(raw_frame, capture_ts)  # landmarks + normalization
    state_update: StateUpdate = m3.update(packet)            # state machine + buffer fill

    prediction: Optional[Prediction] = None
    if state_update.triggered:
        buffer = m3.take_buffer()                            # consume the 64-frame buffer
        prediction = m4.infer(state_update.sign_event, buffer)

    m2.render(packet, state_update, prediction)              # overlay
    m2.log(packet, state_update, prediction)                 # CSV append + metrics
```

**Properties:**
- Each call returns before the next one starts. No callbacks, no futures, no threads.
- `prediction` is `None` on most frames; M2 must accept that and just draw the latest known top-3 (or nothing).
- `m4.infer` blocks the loop for ~100 ms when it fires. Frames captured during that window are dropped (the user just stopped signing, so this is acceptable — see §7).

### 2.4 Identity & ordering rules

1. `frame_id` is the universal join key. M2 correlates `FramePacket`, `StateUpdate`, and (via `sign_id`) `Prediction`.
2. `sign_id` is assigned by M3 at the IDLE→ACTIVE transition. M4 must echo it back unchanged in `Prediction.sign_id`. M2 uses it to compute end-to-end latency: `Prediction.inference_end_ts − sign_start_ts_of(sign_id)`.
3. Timestamps are all `time.perf_counter()` floats. No mixing wall clock and monotonic.
4. M1 owns image acquisition. No other module reads the webcam.
5. M3 owns the 64-frame buffer. M2 never touches it directly.

### 2.5 What M2 commits to

- `m2.render` and `m2.log` must each return in **< 5 ms** on average so they don't drag the capture loop. Heavy work (confusion matrix plotting, report generation) is offline-only — Phase D and E run after the live session ends.
- M2 will run even if `prediction is None` — overlay still draws bbox + state + last-known prediction.
- M2 will not assume a specific GUI backend; rendering uses OpenCV `imshow` with a `display=False` flag for headless eval runs (writes annotated frames to disk instead).

---

## 3. M2 Module Layout

```
M2/
├── __init__.py
├── output.py                # M2Output facade — exposes render() and log()
├── ui_overlay.py            # Phase A — render bbox, state, top-3, FPS
├── latency_logger.py        # Phase B — per-event metrics + structured log
├── data_recorder.py         # Phase C — live sample capture for eval
├── eval_classification.py   # Phase D — confusion matrix, F1, deltas (offline)
├── report_writer.py         # Phase E — error_analysis.md generator (offline)
├── runtime/
│   └── fps_meter.py
├── artifacts/               # generated outputs land here
│   ├── live_predictions.csv
│   ├── latency_metrics.csv
│   ├── confusion_matrix_live.png
│   ├── per_class_f1_delta.csv
│   └── confusable_pairs_live.md
└── tests/
    ├── test_contracts_roundtrip.py
    ├── test_latency_math.py
    └── test_eval_metrics.py
```

`M2Output` is a thin facade that holds the FPS meter, the in-memory pending-prediction cache (so `render` can keep showing the last top-3 between signs), and open file handles for the CSV loggers. It exposes two methods called from the main loop: `render(packet, state_update, prediction)` and `log(packet, state_update, prediction)`.

---

## 4. Phased Implementation Plan

Each phase has a **demo gate** — what must work before moving on.

### Phase A — UI Overlay (`ui_overlay.py`)

**Goal:** draw a working overlay on a live webcam feed using mock predictions.

Steps:
1. Implement `OverlayRenderer.draw(frame, frame_packet, state_update, latest_prediction)`.
2. Render: bounding box rectangle, state text (top-left), top-3 labels with confidence bars (right side), FPS counter (top-right).
3. Implement `FpsMeter` (rolling window, 30 frames).
4. Wire to a stub main loop that constructs canned `FramePacket` / `StateUpdate` / `Prediction` objects directly and calls `render()` to verify visuals before M1/M3/M4 land.

**Demo gate:** record a 10-second clip showing the overlay updating against mock data.

### Phase B — Latency & RTF Instrumentation (`latency_logger.py`)

**Goal:** numbers that prove RTF < 1.0.

Steps:
1. `M2Output.log` receives the `Prediction` directly when `state_update.triggered` is true. It already has `state_update.sign_event` and the corresponding `FramePacket.capture_ts`.
2. Compute per sign:
   - `end_to_end_latency = inference_end_ts − sign_end_ts`
   - `inference_latency = inference_end_ts − inference_start_ts`
   - `rtf = (inference_end_ts − sign_start_ts) / sign_duration_s`
3. Append a row to `artifacts/latency_metrics.csv`: `sign_id, sign_duration_s, e2e_latency_ms, inference_latency_ms, rtf, top1_gloss, top1_prob`.
4. After ≥ 30 samples, compute mean / std / min / max / `% RTF<1.0` and print summary.

**Demo gate:** 30-sample run with summary printed and CSV verified.

### Phase C — Live Data Collection (`data_recorder.py`)

**Goal:** labeled dataset to feed Phase D.

Steps:
1. CLI flow: operator types/selects ground-truth gloss before each sign. Tooling validates against `label_map.json`.
2. On each `Prediction`, append row to `artifacts/live_predictions.csv`: `sign_id, true_gloss, pred_gloss_top1..3, prob_top1..3, sign_duration_s, e2e_latency_ms`.
3. Target ≥ 5 samples × ≥ 30 classes for meaningful F1 (calibrate based on time budget; document actual coverage).

**Demo gate:** CSV with at least 150 labeled samples, no missing fields.

### Phase D — Classification Analysis (`eval_classification.py`)

**Goal:** quantitative diagnosis. **Offline** — runs after the live session ends, reading the CSVs from Phase B/C.

Steps:
1. Load `live_predictions.csv` and offline F1 scores (Phase 1 artifact — coordinate with model owner for path/format).
2. Build confusion matrix on top-1 predictions; render with `matplotlib`, save `confusion_matrix_live.png`. Use class indices from `label_map.json`; sort axes by class id for reproducibility.
3. Compute per-class precision, recall, F1 (handle zero-support classes — log them, don't crash).
4. Compute `delta_f1 = offline_f1 − live_f1` per class; save `per_class_f1_delta.csv`.
5. Rank classes by (a) lowest live F1, (b) largest delta_f1.
6. Extract top confused pairs from off-diagonal mass; save `confusable_pairs_live.md` with at least the top 10 pairs and per-pair sample counts.

**Demo gate:** all three artifacts generated; ranking matches a manual spot-check of the CSV.

### Phase E — Report Writer (`report_writer.py` → `error_analysis.md`)

**Goal:** explain what failed, by how much, and why. **Offline.**

Steps:
1. Section 1 — Worst-performing classes (top 10 by live F1).
2. Section 2 — Largest F1 drops (top 10 by delta_f1).
3. Section 3 — Top confused pairs with example `sign_id`s.
4. Section 4 — Root-cause hypotheses, mapped to upstream module:
   - **M1 candidates:** occlusion patterns, MediaPipe dropout, normalization mismatch (compare a few frames' normalized output to training-time output).
   - **M3 candidates:** premature trigger (TR too low), late trigger (TA too high), buffer truncation when sign > 64 frames.
   - **M4 candidates:** miscalibrated confidence, class imbalance from training.
5. Section 5 — Proposed fixes (no implementation), tagged with owner module.
6. Merge into joint report with M3.

**Demo gate:** `error_analysis.md` reviewed by at least one other module owner; every claim cited to a row in the live CSV or an artifact image.

---

## 5. Integration Milestones

| Milestone | Trigger | Verification |
|---|---|---|
| **I0 — Contracts frozen** | Section 2 signed off by M1/M3/M4 owners | `pipeline/contracts.py` merged, dataclasses imported by all modules |
| **I1 — UI on mock** | Phase A complete | 10-sec demo clip |
| **I2 — End-to-end dry run** | M1+M3+M4 wired into the main loop in `pipeline/run.py` | Overlay shows non-mock predictions |
| **I3 — Latency baseline** | Phase B complete on real pipeline | `latency_metrics.csv` with ≥ 30 samples |
| **I4 — Live data collected** | Phase C complete | `live_predictions.csv` ≥ 150 rows |
| **I5 — Final report** | Phase E complete | Joint `error_analysis.md` shipped |

---

## 6. Testing Strategy

- **`test_contracts_roundtrip.py`** — construct each dataclass, serialize/deserialize, assert no field loss. Catches schema drift between modules.
- **`test_latency_math.py`** — feed synthetic timestamps, assert e2e/inference/RTF computations are correct to 0.1 ms.
- **`test_eval_metrics.py`** — feed a tiny known confusion matrix, assert per-class F1 matches a hand-computed value.
- **Manual smoke test** — Phase A overlay against canned data before integration.

CI gate (if available): all three test files must pass before merging M2 changes.

---

## 7. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Inference (~100 ms) blocks the capture loop, dropping 2–3 frames at sign-end | High | Low | Acceptable: inference fires only at Active→Idle, user is between signs. If next sign starts immediately, M3 still detects onset once `TA` accumulates — minor latency cost only. |
| M4 inference much slower than budget (>300 ms) makes the post-sign blackout user-visible | Low | Medium | Measure inference latency in Phase B before committing. If breached, profile M4 first; only reconsider threading as a last resort. |
| `m2.render` or `m2.log` accidentally heavy (e.g. plotting in the loop) drags FPS | Medium | Medium | Phase D/E artifacts are explicitly offline; Phase B asserts < 5 ms per render/log call in tests. |
| Contract drift after Phase A | Medium | High | Freeze `contracts.py` at I0; bump a `CONTRACT_VERSION` constant on any change |
| Timestamp source mismatch (wall clock vs perf_counter) | Medium | High | Document `time.perf_counter()` as the only allowed source in §2.4 |
| Sign duration > 64 frames (buffer truncation) | Medium | Medium | M3 reports `buffer_length` in `SignEvent`; M2 flags truncated samples in metrics CSV |
| Offline F1 file format unknown | High | Medium | Resolve at I0 — either use the Phase 1 evaluation script or specify a JSON schema |
| Insufficient live samples for F1 | High | High | Collect during Phase B already; don't wait until Phase D |
| Headless / no GUI environment | Low | Medium | `OverlayRenderer` accepts `display=False` and writes annotated frames to disk instead |

---

## 8. Success Criteria (Exit Definition)

- Real-time UI displays bbox, state, top-3 labels with confidence, FPS, latency.
- Latency / RTF measured on ≥ 30 samples, with mean / std / min / max / `% RTF<1.0` reported.
- Confusion matrix and per-class F1 generated from live data.
- `error_analysis.md` localizes each major failure mode to M1, M3, or M4 with cited evidence.
- All contract tests pass; no other module had to change a public field after I0.

---

## 9. Open Questions (resolve before I0)

1. ~~Where does M3 publish `SignEvent` — direct call into M4, or shared queue M2 also subscribes to?~~ **Resolved:** direct call. M3.update returns a `StateUpdate` whose `sign_event` field is populated when `triggered` is true; the main loop hands that to M4.
2. Format and path of Phase-1 offline F1 scores?
3. Is M2 responsible for recording video clips of misclassifications, or only metadata?
4. Headless mode required for evaluation runs?
5. Confidence threshold for displaying a prediction at all (e.g. suppress if top-1 prob < 0.2)?

---

## 10. Quick Reference — One-Page Cheat Sheet

- **Architecture:** single-threaded direct-call loop in `pipeline/run.py`. No queues, no threads.
- **Inputs M2 needs (as function arguments):** `FramePacket` (M1), `StateUpdate` (M3, contains optional `SignEvent`), `Prediction` (M4, only on triggered frames).
- **Outputs M2 produces:** overlay UI, `latency_metrics.csv`, `live_predictions.csv`, `confusion_matrix_live.png`, `per_class_f1_delta.csv`, `confusable_pairs_live.md`, classification section of `error_analysis.md`.
- **Phase order:** A (UI) → B (latency) → C (data) → D (eval, offline) → E (report, offline).
- **Golden rule:** M2 diagnoses, never patches upstream. Render and log calls must each stay under 5 ms.
