"""
main.py — ISL recognition pipeline

M1 : extractor.LandmarkExtractor          (HolisticLandmarker → 65-point (65,2) array)
M3 : buffer.M3StateMachine                (state machine + 64-frame buffer)
M4 : spoter.realtime_engine.SignLanguageEngine  (SPOTER inference, event-driven)
M2 : ui_render.output.M2Output            (overlay: landmarks, state, predictions)

Run:  conda activate dsde && python main.py
Quit: press Q or close the window
"""

import logging
import time
import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

from spoter.realtime_engine import SignLanguageEngine
from spoter.normalization.body_normalization import BODY_IDENTIFIERS
from spoter.normalization.hand_normalization import HAND_IDENTIFIERS
from buffer import M3StateMachine
from contract.contracts import Prediction
from extractor.mediapipe_pipeline import LandmarkExtractor
from ui_render.output import M2Output

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(name)-4s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

TARGET_FPS = 30
_FRAME_BUDGET = 1.0 / TARGET_FPS

_WINDOW_NAME = "ISL Recognition"

# ── Load model ────────────────────────────────────────────────────────────────

MODEL_PATH = "spoter/spoter_model_final.pt"
LABEL_PATH = "spoter/label_map.json"

engine = SignLanguageEngine(MODEL_PATH, LABEL_PATH)

# ── Output directory for buffer CSVs ──────────────────────────────────────────
BUFFER_OUTPUT_DIR = "buffer_outputs"
os.makedirs(BUFFER_OUTPUT_DIR, exist_ok=True)


# ── Save buffer to CSV ────────────────────────────────────────────────────────

def save_buffer_to_csv(df: pd.DataFrame, sign_id: int, prediction_label: str = "") -> str:
    """
    Write take_buffer() DataFrame to a CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by m3.take_buffer()
    sign_id : int
        Sign ID for labeling
    prediction_label : str
        Top-1 prediction gloss (optional, for naming)
    
    Returns
    -------
    str
        Path to the saved CSV file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    label_suffix = f"_{prediction_label}" if prediction_label else ""
    filename = f"sign_{sign_id:03d}_{timestamp}{label_suffix}.csv"
    filepath = os.path.join(BUFFER_OUTPUT_DIR, filename)
    
    df.to_csv(filepath, index=False)
    log.info(f"✓ Buffer saved: {filepath}")
    return filepath


# ── Buffer → numpy conversion ─────────────────────────────────────────────────

def _csv_buf_to_numpy(df: pd.DataFrame, n_frames: int = 64) -> np.ndarray:
    """
    Convert a take_buffer() DataFrame to (n_frames, 108) float32 for the model.

    Each DataFrame cell is a list of n_frames floats.  Columns are looked up by
    the identifier names SPOTER expects, so the output order matches all_ids in
    SignLanguageEngine._internal_preprocess.
    """
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

    # (54, n_frames, 2) → (n_frames, 54, 2) → (n_frames, 108)
    return np.stack(arrays, axis=0).transpose(1, 0, 2).reshape(n_frames, 108).astype(np.float32)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("ISL pipeline starting")
    log.info("  TA=5  TR=10")
    log.info("-" * 60)

    m1 = LandmarkExtractor()
    m3 = M3StateMachine(ta=5, tr=10)
    m2 = M2Output(display=True, display_scale=1.0)

    try:
        while True:
            t_frame_start = time.perf_counter()

            # ── M1: capture + extract landmarks ──────────────────────────────
            raw_frame, capture_ts = m1.grab()
            if raw_frame is None:
                log.warning("Webcam read failed — exiting")
                break

            raw_frame = cv2.flip(raw_frame, 1)
            packet = m1.extract(raw_frame, capture_ts)

            # ── M3: state machine ─────────────────────────────────────────────
            state_update = m3.update(packet)

            # ── M4: inference on sign trigger ─────────────────────────────────
            prediction = None
            if state_update.triggered:
                csv_buf  = m3.take_buffer()
                
                # Save buffer to CSV
                pred_label = ""
                if not csv_buf.empty:
                    norm_buf = _csv_buf_to_numpy(csv_buf)
                    se = state_update.sign_event

                    result = engine.run_inference(norm_buf)
                    pred_label = result["top_k_glosses"][0] if result["top_k_glosses"] else ""
                    
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
                
                # Write buffer to CSV file
                save_buffer_to_csv(csv_buf, state_update.sign_id or 0, pred_label)

            # ── M2: render overlay (landmarks + state + predictions) ──────────
            m2.render(packet, state_update, prediction)
            m2.log(packet, state_update, prediction)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty(_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            # ── FPS pacing ────────────────────────────────────────────────────
            elapsed = time.perf_counter() - t_frame_start
            time.sleep(max(0.0, _FRAME_BUDGET - elapsed))

    finally:
        m1.release()
        cv2.destroyAllWindows()

    m3.save_trigger_log("trigger_error_log.csv")
    log.info("-" * 60)
    log.info("Session ended — error summary:")
    for k, v in m3.error_summary().items():
        log.info("  %s: %s", k, v)
    log.info("trigger_error_log.csv saved")


if __name__ == "__main__":
    main()
