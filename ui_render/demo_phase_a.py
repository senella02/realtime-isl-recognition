"""
Phase A demo gate — runs the overlay against canned mock data.
No webcam, no model, no M1/M3/M4 required.

Press 'q' to quit.
Press 's' to toggle sign simulation (triggers a mock Prediction).
"""

import time
import numpy as np
import cv2

from contract.contracts import (
    FramePacket, StateUpdate, SignState, SignEvent, Prediction
)
from ui_render.output import M2Output

# ── mock helpers ──────────────────────────────────────────────────────────────

def _mock_frame(width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # subtle grid pattern so motion is visible
    for i in range(0, width, 40):
        cv2.line(frame, (i, 0), (i, height), (30, 30, 30), 1)
    for j in range(0, height, 40):
        cv2.line(frame, (0, j), (width, j), (30, 30, 30), 1)
    cv2.putText(frame, "MOCK FRAME", (220, 240), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (50, 50, 50), 2, cv2.LINE_AA)
    return frame


def _mock_packet(frame_id: int, ts: float, active: bool) -> FramePacket:
    frame = _mock_frame()
    bbox = (80, 60, 560, 420) if active else None
    return FramePacket(
        frame_id=frame_id,
        capture_ts=ts,
        image_bgr=frame,
        bbox=bbox,
        landmarks_raw=None,
        landmarks_normalized=None,
    )


def _mock_state(frame_id: int, active: bool, count: int,
                triggered: bool = False, sign_id: int = 0) -> StateUpdate:
    sign_event = None
    if triggered:
        now = time.perf_counter()
        sign_event = SignEvent(
            sign_id=sign_id,
            sign_start_ts=now - 1.2,
            sign_end_ts=now,
            sign_duration_s=1.2,
            buffer_length=48,
        )
    return StateUpdate(
        frame_id=frame_id,
        state=SignState.ACTIVE if active else SignState.IDLE,
        active_frame_count=count,
        sign_id=sign_id if active else None,
        triggered=triggered,
        sign_event=sign_event,
    )


def _mock_prediction(sign_id: int) -> Prediction:
    probs = np.random.dirichlet(np.ones(184)).astype(np.float32)
    top_k = np.argsort(probs)[::-1][:3].tolist()
    glosses = ["สวัสดี", "ขอบคุณ", "ประเทศไทย"]  # sample Thai glosses
    return Prediction(
        sign_id=sign_id,
        inference_start_ts=time.perf_counter() - 0.08,
        inference_end_ts=time.perf_counter(),
        probs=probs,
        top_k_indices=top_k,
        top_k_probs=[float(probs[i]) for i in top_k],
        top_k_glosses=glosses,
    )


# ── main loop ─────────────────────────────────────────────────────────────────

def run_demo(duration_s: float = 30.0) -> None:
    m2 = M2Output(display=True)

    frame_id = 0
    sign_id = 0
    active = False
    active_count = 0
    sign_toggle_at = time.perf_counter() + 2.0  # first sign starts at t=2 s

    start = time.perf_counter()
    print("Phase A demo running — press 'q' to quit, 's' to toggle sign state.")

    while time.perf_counter() - start < duration_s:
        now = time.perf_counter()

        # auto-toggle sign state every 3 s for a more interesting demo
        triggered = False
        if now >= sign_toggle_at:
            active = not active
            sign_toggle_at = now + 3.0
            if active:
                sign_id += 1
                active_count = 0
                print(f"  [demo] sign {sign_id} started")
            else:
                triggered = True  # Active→Idle transition: fire prediction
                print(f"  [demo] sign {sign_id} ended → triggering mock prediction")

        if active:
            active_count += 1

        packet = _mock_packet(frame_id, now, active)
        state_update = _mock_state(frame_id, active, active_count,
                                   triggered=triggered, sign_id=sign_id)
        prediction = _mock_prediction(sign_id) if triggered else None

        m2.render(packet, state_update, prediction)
        m2.log(packet, state_update, prediction)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            active = not active
            if active:
                sign_id += 1
                active_count = 0
            else:
                triggered = True

        frame_id += 1
        # target ~30 fps
        time.sleep(max(0, 1 / 30 - (time.perf_counter() - now)))

    cv2.destroyAllWindows()
    print("Demo finished.")


if __name__ == "__main__":
    run_demo()
