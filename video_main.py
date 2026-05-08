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

import cv2
import numpy as np
import pandas as pd

from spoter.realtime_engine import SignLanguageEngine
from spoter.normalization.body_normalization import BODY_IDENTIFIERS
from spoter.normalization.hand_normalization import HAND_IDENTIFIERS
from buffer import M3StateMachine
from contract.contracts import Prediction
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


# ── Buffer → numpy conversion (identical to main.py) ─────────────────────────

def _csv_buf_to_numpy(df: pd.DataFrame, n_frames: int = 64) -> np.ndarray:
    if df.empty:
        return np.zeros((n_frames, 108), dtype=np.float32)

    row = df.iloc[0]
    arrays = []

    for name in BODY_IDENTIFIERS:
        arrays.append(np.column_stack([row[f"{name}_X"], row[f"{name}_Y"]]))

    for name in HAND_IDENTIFIERS:
        arrays.append(np.column_stack([row[f"{name}_left_X"], row[f"{name}_left_Y"]]))

    for name in HAND_IDENTIFIERS:
        arrays.append(np.column_stack([row[f"{name}_right_X"], row[f"{name}_right_Y"]]))

    return np.stack(arrays, axis=0).transpose(1, 0, 2).reshape(n_frames, 108).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(video_path: str) -> None:
    log.info("ISL pipeline starting (video mode)")
    log.info("  source: %s", video_path)
    log.info("  TA=1  TR=1")
    log.info("-" * 60)

    m1 = VideoLandmarkExtractor(video_path)
    m3 = M3StateMachine(ta=5, tr=10)
    m2 = M2Output(display=True, display_scale=1.0)

    frame_budget = 1.0 / m1.fps   # pace display to video's native speed

    try:
        while True:
            t_frame_start = time.perf_counter()

            # ── M1: read frame + extract landmarks ───────────────────────────
            raw_frame, capture_ts = m1.grab()
            if raw_frame is None:
                log.info("Video ended — exiting")
                break

            packet = m1.extract(raw_frame, capture_ts)

            # ── M3: state machine ─────────────────────────────────────────────
            state_update = m3.update(packet)

            # ── M4: inference on sign trigger ─────────────────────────────────
            prediction = None
            if state_update.triggered:
                csv_buf  = m3.take_buffer()
                norm_buf = _csv_buf_to_numpy(csv_buf)
                se = state_update.sign_event

                result = engine.run_inference(norm_buf)
                prediction = Prediction(
                    sign_id=se.sign_id,
                    inference_start_ts=result["inference_start_ts"],
                    inference_end_ts=result["inference_end_ts"],
                    probs=result["probs"],
                    top_k_indices=result["top_k_indices"],
                    top_k_probs=result["top_k_probs"],
                    top_k_glosses=result["top_k_glosses"],
                )
                log.info(
                    "→ M4: sign #%d  top=%s (%.1f%%)  %.1f ms",
                    se.sign_id,
                    prediction.top_k_glosses[0],
                    prediction.top_k_probs[0] * 100,
                    (result["inference_end_ts"] - result["inference_start_ts"]) * 1000,
                )

            # ── M2: render overlay ────────────────────────────────────────────
            m2.render(packet, state_update, prediction)
            m2.log(packet, state_update, prediction)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("User quit")
                break
            if cv2.getWindowProperty(_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            # Pace to video's native FPS so the display doesn't race ahead
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
