"""
video_main.py — ISL recognition pipeline over a recorded .mp4 file.

Identical pipeline to main.py; only the frame source changes.

Usage:
    conda activate dsde && python video_main.py <path/to/video.mp4>

Quit early: press Q or close the window.
"""

import logging
import sys
import time
from typing import Optional

import cv2
import numpy as np

from spoter.realtime_engine import SignLanguageEngine
from buffer import M3StateMachine
from buffer.realtime_engine import State
from contract.contracts import Prediction
from data_preprocess.normalized_np.main import normalized_batch
from extractor.video_pipeline import VideoLandmarkExtractor
from ui_render.output import M2Output

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(name)-4s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("video_main")

_WINDOW_NAME = "ISL Recognition"

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "spoter/spoter_model_final.pt"
LABEL_PATH = "spoter/label_map.json"

engine = SignLanguageEngine(MODEL_PATH, LABEL_PATH)


# ── Buffer helpers (video_main-only) ─────────────────────────────────────────

def _interp_to_64(raw_frames: list, target: int = 64) -> np.ndarray:
    """Resample a variable-length frame sequence to `target` frames via linear interpolation."""
    valid = [f for f in raw_frames if f is not None]
    n = len(valid)
    if n == 0:
        return np.zeros((target, 108), dtype=np.float32)

    flat = np.stack(valid, axis=0).reshape(n, 130).astype(np.float32)
    normed = normalized_batch(flat)  # (n, 108)
    if n == target:
        return normed.astype(np.float32)

    old_t = np.linspace(0.0, 1.0, n)
    new_t = np.linspace(0.0, 1.0, target)
    out = np.zeros((target, normed.shape[1]), dtype=np.float32)
    for col in range(normed.shape[1]):
        out[:, col] = np.interp(new_t, old_t, normed[:, col])
    return out


def _flush_buffer(m3: M3StateMachine) -> Optional[np.ndarray]:
    """
    Drain the M3 buffer, reset engine to IDLE, and return a (64, 108) interpolated array.
    Returns None if the buffer is empty.
    """
    eng = m3._engine
    raw_frames = [f for f in eng.buffer if f is not None]
    if not raw_frames:
        return None

    log.info("  flush: interpolating %d frames → 64", len(raw_frames))
    eng.buffer.clear()
    eng._consecutive_active = 0
    eng._consecutive_rest = 0
    eng.state = State.IDLE
    m3._current_sign_id = None
    m3._sign_start_ts = None
    m3._frames_since_onset = 0

    return _interp_to_64(raw_frames)


def _make_prediction(norm_buf: np.ndarray, sign_id: int) -> Prediction:
    result = engine.run_inference(norm_buf)
    return Prediction(
        sign_id=sign_id,
        inference_start_ts=result["inference_start_ts"],
        inference_end_ts=result["inference_end_ts"],
        probs=result["probs"],
        top_k_indices=result["top_k_indices"],
        top_k_probs=result["top_k_probs"],
        top_k_glosses=result["top_k_glosses"],
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main(video_path: str) -> None:
    log.info("ISL pipeline starting (video mode)")
    log.info("  source: %s", video_path)
    log.info("  TA=5  TR=10")
    log.info("-" * 60)

    m1 = VideoLandmarkExtractor(video_path)
    m3 = M3StateMachine(ta=5, tr=10)
    m2 = M2Output(display=True, display_scale=1.0)

    frame_budget = 1.0 / m1.fps

    try:
        while True:
            t_frame_start = time.perf_counter()

            # ── M1: read frame + extract landmarks ───────────────────────────
            raw_frame, capture_ts = m1.grab()
            if raw_frame is None:
                log.info("Video ended — flushing remaining buffer")
                eng = m3._engine
                if eng.state == State.ACTIVE and len(eng.buffer) > 0:
                    sign_id = m3._current_sign_id or -1
                    norm_buf = _flush_buffer(m3)
                    if norm_buf is not None:
                        prediction = _make_prediction(norm_buf, sign_id)
                        log.info(
                            "→ M4 (video-end): sign #%d  top=%s (%.1f%%)  %.1f ms",
                            sign_id,
                            prediction.top_k_glosses[0],
                            prediction.top_k_probs[0] * 100,
                            (prediction.inference_end_ts - prediction.inference_start_ts) * 1000,
                        )
                break

            packet = m1.extract(raw_frame, capture_ts)

            # ── M3: state machine ─────────────────────────────────────────────
            state_update = m3.update(packet)

            # ── M4: inference ─────────────────────────────────────────────────
            prediction = None
            eng = m3._engine

            if (eng.state == State.ACTIVE
                    and eng._consecutive_rest >= eng.tr
                    and len(eng.buffer) > 0):
                # TR trigger: sign ended naturally with a partial buffer.
                # The shared engine tracks _consecutive_rest but never checks it,
                # so we detect and fire here instead.
                sign_id = m3._current_sign_id or state_update.sign_id or -1
                log.info("■ TR trigger: sign #%d  buffer=%d frames", sign_id, len(eng.buffer))
                norm_buf = _flush_buffer(m3)
                if norm_buf is not None:
                    prediction = _make_prediction(norm_buf, sign_id)
                    log.info(
                        "→ M4 (TR): sign #%d  top=%s (%.1f%%)  %.1f ms",
                        sign_id,
                        prediction.top_k_glosses[0],
                        prediction.top_k_probs[0] * 100,
                        (prediction.inference_end_ts - prediction.inference_start_ts) * 1000,
                    )

            elif state_update.triggered:
                # Buffer-full trigger: normal 64-frame path
                norm_buf = m3.take_buffer()  # already (64, 108) ndarray
                se = state_update.sign_event
                prediction = _make_prediction(norm_buf, se.sign_id)
                log.info(
                    "→ M4: sign #%d  top=%s (%.1f%%)  %.1f ms",
                    se.sign_id,
                    prediction.top_k_glosses[0],
                    prediction.top_k_probs[0] * 100,
                    (prediction.inference_end_ts - prediction.inference_start_ts) * 1000,
                )

            # ── M2: render overlay ────────────────────────────────────────────
            m2.render(packet, state_update, prediction)
            m2.log(packet, state_update, prediction)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("User quit")
                break
            if cv2.getWindowProperty(_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            elapsed = time.perf_counter() - t_frame_start
            time.sleep(max(0.0, frame_budget - elapsed))

    finally:
        m1.release()
        cv2.destroyAllWindows()

    m3.save_trigger_log("trigger_error_log_video.csv")
    log.info("-" * 60)
    log.info("Session ended — error summary:")
    for k, v in m3.error_summary().items():
        log.info("  %s: %s", k, v)
    log.info("trigger_error_log_video.csv saved")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_main.py <path/to/video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
