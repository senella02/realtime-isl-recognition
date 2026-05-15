"""
main.py — ISL recognition pipeline

M1 : extractor.LandmarkExtractor          (HolisticLandmarker → 65-point (65,2) array)
M3 : buffer.M3StateMachine                (state machine + 64-frame buffer)
M4 : spoter.realtime_engine.SignLanguageEngine  (SPOTER inference, event-driven)
M2 : ui_render.output.M2Output            (overlay: landmarks, state, predictions)

Run:  conda activate dsde && python main.py
Quit: press Q or close the window
"""

import csv
import logging
import time

import cv2

from spoter.realtime_engine import SignLanguageEngine
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


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("ISL pipeline starting")
    log.info("  TA=5  TR=10")
    log.info("-" * 60)

    m1 = LandmarkExtractor()
    m3 = M3StateMachine(ta=5, tr=10)
    m2 = M2Output(display=True, display_scale=1.0)

    rtf_log_path = "rtf_results.csv"
    rtf_file = open(rtf_log_path, "w", newline="")
    rtf_writer = csv.writer(rtf_file)
    rtf_writer.writerow([
        "sign_id", "sign_duration_s", "inference_ms",
        "e2e_latency_ms", "rtf", "top_gloss", "top_prob_pct"
    ])

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
                norm_buf = m3.take_buffer()
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
                inference_ms = (result["inference_end_ts"] - result["inference_start_ts"]) * 1000
                e2e_ms = (result["inference_end_ts"] - se.sign_end_ts) * 1000
                rtf = (result["inference_end_ts"] - result["inference_start_ts"]) / se.sign_duration_s
                log.info(
                    "→ M4: sign #%d  top=%s (%.1f%%)  inference=%.1fms  e2e=%.1fms  RTF=%.4f",
                    se.sign_id,
                    prediction.top_k_glosses[0],
                    prediction.top_k_probs[0] * 100,
                    inference_ms, e2e_ms, rtf,
                )
                rtf_writer.writerow([
                    se.sign_id,
                    f"{se.sign_duration_s:.3f}",
                    f"{inference_ms:.1f}",
                    f"{e2e_ms:.1f}",
                    f"{rtf:.4f}",
                    prediction.top_k_glosses[0],
                    f"{prediction.top_k_probs[0] * 100:.1f}",
                ])
                rtf_file.flush()

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
        rtf_file.close()
        log.info("RTF results saved → %s", rtf_log_path)

    m3.save_trigger_log("trigger_error_log.csv")
    log.info("-" * 60)
    log.info("Session ended — error summary:")
    for k, v in m3.error_summary().items():
        log.info("  %s: %s", k, v)
    log.info("trigger_error_log.csv saved")


if __name__ == "__main__":
    main()
