"""
main.py — ISL recognition pipeline (M3 debug mode)

M1 stub : OpenCV frame-differencing → is_active signal (no MediaPipe yet)
M3      : buffer.M3StateMachine — state machine + 64-frame buffer
M4 stub : prints buffer size when triggered
M2 stub : inline OpenCV overlay

Run:  conda activate dsde && python main.py
Quit: press Q or close the window
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np

from buffer import M3StateMachine
from contract.contracts import FramePacket, SignState

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(name)-4s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

# Uncomment to also see per-frame motion scores (very verbose):
# logging.getLogger("M3").setLevel(logging.DEBUG)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Motion detection config ───────────────────────────────────────────────────
# Minimum fraction of pixels that must change for a frame to count as "active".
# Raise if small background movement keeps triggering; lower if signs are missed.
MOTION_PIXEL_THRESHOLD = 0.02   # 2 % of frame pixels must move


# ── M1 stub: frame-differencing motion detector ───────────────────────────────

class MotionDetector:
    """
    Simple frame-differencing H/A classifier.
    No MediaPipe required — works purely on pixel change between consecutive frames.
    Replace with real MediaPipe landmark extraction when M1 is implemented.
    """

    def __init__(self, pixel_threshold: float = MOTION_PIXEL_THRESHOLD) -> None:
        self._threshold = pixel_threshold
        self._prev_gray: Optional[np.ndarray] = None
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=False
        )

    def is_active(self, bgr: np.ndarray) -> bool:
        """Return True if significant motion is detected in this frame."""
        fg_mask = self._bg_sub.apply(bgr)
        moving_fraction = np.count_nonzero(fg_mask) / fg_mask.size
        return moving_fraction > self._threshold

    def debug_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Return a 3-channel view of the foreground mask for overlay."""
        mask = self._bg_sub.apply(bgr, learningRate=0)
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


# ── M2 stub: overlay ──────────────────────────────────────────────────────────

def _draw_overlay(frame: np.ndarray, state_update, last_trigger: dict) -> None:
    h, w = frame.shape[:2]

    # State label (top-left)
    if state_update.state == SignState.ACTIVE:
        color = (0, 220, 0)
        label = f"ACTIVE  [{state_update.active_frame_count} fr]"
    else:
        color = (160, 160, 160)
        label = "IDLE"
    cv2.putText(frame, label, (10, 36), _FONT, 1.0, color, 2, cv2.LINE_AA)

    # Last trigger info (bottom-left)
    if last_trigger:
        txt = (f"last: sign #{last_trigger['id']}  "
               f"{last_trigger['buf']} frames  "
               f"{last_trigger['dur']:.2f}s")
        cv2.putText(frame, txt, (10, h - 14), _FONT, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    # Quit hint (top-right)
    hint = "Q=quit"
    (tw, _), _ = cv2.getTextSize(hint, _FONT, 0.45, 1)
    cv2.putText(frame, hint, (w - tw - 8, 20), _FONT, 0.45, (100, 100, 100), 1, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("ISL pipeline starting — M3 debug mode (motion-detection stub)")
    log.info("  TA=5  TR=10  motion_pixel_threshold=%.1f%%", MOTION_PIXEL_THRESHOLD * 100)
    log.info("  Move in front of camera → ACTIVE; hold still ~10 frames → inference fires")
    log.info("-" * 60)

    m3 = M3StateMachine(ta=5, tr=10)
    detector = MotionDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open webcam")
        return

    frame_id = 0
    last_trigger: dict = {}

    while True:
        ret, bgr = cap.read()
        if not ret:
            log.warning("Webcam read failed — exiting")
            break

        bgr = cv2.flip(bgr, 1)
        frame_id += 1
        ts = time.perf_counter()

        # ── M1 stub: motion-based H/A classification ──────────────────────────
        active = detector.is_active(bgr)

        packet = FramePacket(
            frame_id=frame_id,
            capture_ts=ts,
            image_bgr=bgr,
            bbox=None,
            landmarks_raw=None,   # no landmarks yet — M1 will fill this
        )

        # ── M3: pass is_active directly (bypasses landmark-based classifier) ──
        state_update = m3.update(packet, is_active_override=active)

        # ── M4 stub: consume buffer on trigger ────────────────────────────────
        if state_update.triggered:
            buffer = m3.take_buffer()
            se = state_update.sign_event
            last_trigger = {"id": se.sign_id, "buf": len(buffer), "dur": se.sign_duration_s}
            log.info("→ M4 stub: %d-frame buffer ready for sign #%d  (%.2fs)",
                     len(buffer), se.sign_id, se.sign_duration_s)
            # TODO: prediction = m4.infer(se, buffer)

        # ── M2 stub: overlay ──────────────────────────────────────────────────
        _draw_overlay(bgr, state_update, last_trigger)
        cv2.imshow("ISL Recognition — M3 debug", bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty("ISL Recognition — M3 debug", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

    m3.save_trigger_log("trigger_error_log.csv")
    log.info("-" * 60)
    log.info("Session ended — error summary:")
    for k, v in m3.error_summary().items():
        log.info("  %s: %s", k, v)
    log.info("trigger_error_log.csv saved")


if __name__ == "__main__":
    main()
