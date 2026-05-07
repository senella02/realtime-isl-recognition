"""
main.py — ISL recognition pipeline (M3 debug mode)

M1 : extractor.LandmarkExtractor  (HolisticLandmarker → 65-point (65,2) array)
M3 : buffer.M3StateMachine         (state machine + 64-frame buffer)
     H/A classification uses only hand rows [23:65] — head/body motion is ignored
M4 stub : prints buffer info when triggered
M2 stub : inline OpenCV overlay

Run:  conda activate dsde && python main.py
Quit: press Q or close the window
"""

import csv
import logging
import time
from collections import deque

import cv2
import numpy as np

from spoter.realtime_engine import SignLanguageEngine
from buffer import M3StateMachine
from contract.contracts import SignState
from extractor.mediapipe_pipeline import LandmarkExtractor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(name)-4s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

# Uncomment to also see per-frame motion scores (very verbose):
# logging.getLogger("M3").setLevel(logging.DEBUG)

TARGET_FPS = 30
_FRAME_BUDGET = 1.0 / TARGET_FPS  # 33.3 ms

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── M4: Load model once before loop ──────────────────────────────────────────

MODEL_PATH = "spoter/spoter_model_final.pt"
LABEL_PATH = "spoter/label_map.json"

engine = SignLanguageEngine(MODEL_PATH, LABEL_PATH)
last_results = None

# ── M2 stub: overlay ──────────────────────────────────────────────────────────

def _draw_overlay(frame: np.ndarray, state_update, last_trigger: dict, fps: float,
                  landmark_count: int = 0) -> None:
    h, w = frame.shape[:2]

    if state_update.state == SignState.ACTIVE:
        color = (0, 220, 0)
        label = f"ACTIVE  [{state_update.active_frame_count} fr]"
    else:
        color = (160, 160, 160)
        label = "IDLE"
    cv2.putText(frame, label, (10, 36), _FONT, 1.0, color, 2, cv2.LINE_AA)

    cv2.putText(frame, f"landmarks: {landmark_count}/65",
                (10, 62), _FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    if last_trigger:
        txt = (f"last: sign #{last_trigger['id']}  "
               f"{last_trigger['buf']} frames  "
               f"{last_trigger['dur']:.2f}s")
        cv2.putText(frame, txt, (10, h - 14), _FONT, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    fps_txt = f"FPS: {fps:.1f}/{TARGET_FPS}"
    (tw, _), _ = cv2.getTextSize(fps_txt, _FONT, 0.45, 1)
    cv2.putText(frame, fps_txt, (w - tw - 8, 20), _FONT, 0.45, (100, 100, 100), 1, cv2.LINE_AA)


def _draw_landmarks(frame: np.ndarray, raw: np.ndarray) -> None:
    """Draw body (white), left-hand (blue), right-hand (red) landmark dots."""
    h, w = frame.shape[:2]
    regions = [
        (raw[0:23],  (255, 255, 255), 4),  # body — white, larger
        (raw[23:44], (220, 100,   0), 3),  # left hand — blue
        (raw[44:65], (  0,  80, 220), 3),  # right hand — red
    ]
    for pts, color, r in regions:
        for x, y in pts:
            if x == 0.0 and y == 0.0:
                continue
            cv2.circle(frame, (int(x * w), int(y * h)), r, color, -1)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("ISL pipeline starting — M3 debug mode")
    log.info("  TA=5  TR=10  motion_threshold=0.020")
    log.info("  H/A classification uses HAND landmarks only (rows 23-65)")
    log.info("  Head / body movement will NOT trigger ACTIVE")
    log.info("-" * 60)

    m1 = LandmarkExtractor()
    # m3 = M3StateMachine(ta=5, tr=10, motion_threshold=0.01)
    m3 = M3StateMachine(ta=5, tr=10)

    last_trigger: dict = {}
    cycle_times: deque = deque(maxlen=30)

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

            # ── M3: state machine (landmark-based H/A from hands only) ───────
            state_update = m3.update(packet)

            # ── M4 stub ───────────────────────────────────────────────────────
            if state_update.triggered:
                norm_buf = m3.take_buffer()  # (64, 108) float32, already normalized
                se = state_update.sign_event
                last_trigger = {"id": se.sign_id, "buf": se.buffer_length, "dur": se.sign_duration_s}
                log.info("→ M4 stub: sign #%d  norm=%s  (%.2fs)",
                         se.sign_id, norm_buf.shape, se.sign_duration_s)
                
                # Save normalized buffer to CSV file
                norm_csv = f'norm_buf_{se.sign_id}.csv'
                with open(norm_csv, 'w', newline='') as f:
                    writer = csv.writer(f)
                    header = ['frame'] + [f'feature{i}' for i in range(norm_buf.shape[1])]
                    writer.writerow(header)
                    for i, row in enumerate(norm_buf):
                        writer.writerow([i] + row.tolist())
                
                # Save normalized buffer to NPY file
                norm_npy = f'norm_buf_{se.sign_id}.npy'
                np.save(norm_npy, norm_buf)
                
                # prediction 
                if norm_buf is not None:
                    last_results = engine.run_inference(norm_buf, t_frame_start)
                    
                    if last_results:
                        print(f"Predicted: {last_results['label']} ({last_results['confidence']}\nPredict time: {last_results['predict_time_ms']} ms\nTotal process time: {last_results['total_process_time_ms']}")
    
            # ── M2 stub: overlay ──────────────────────────────────────────────
            cycle_times.append(time.perf_counter() - t_frame_start)
            fps = (len(cycle_times) - 1) / sum(cycle_times) if len(cycle_times) > 1 else 0.0

            display = packet.image_bgr.copy()
            lm_count = 0
            if packet.landmarks_raw is not None:
                lm_count = int(np.any(packet.landmarks_raw != 0, axis=1).sum())
                _draw_landmarks(display, packet.landmarks_raw)
            _draw_overlay(display, state_update, last_trigger, fps, lm_count)

            cv2.imshow("ISL Recognition — M3 debug", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty("ISL Recognition — M3 debug", cv2.WND_PROP_VISIBLE) < 1:
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
