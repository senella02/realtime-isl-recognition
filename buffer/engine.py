"""
buffer/engine.py — contracts-compatible wrapper around RealtimeEngine.

Contract (v1.2.0):
  FramePacket.landmarks_raw  Optional[np.ndarray]  shape (65, 2) float32
      [0:23]  23 upper-body pose landmarks (MediaPipe indices 0–22)
      [23:44] 21 left hand landmarks
      [44:65] 21 right hand landmarks

M3 responsibilities:
  - Compute per-frame H/A classification from hand-landmark presence.
  - Drive the A/H state machine and 64-frame sliding buffer.
  - Each buffer entry is the raw (65, 2) array; take_buffer() normalizes
    the full sequence via normalized_batch before returning to the caller.
  - Expose StateUpdate every frame; sign_event + triggered on Active→Idle.
"""

import logging
from typing import Optional

import numpy as np

from contract.contracts import FramePacket, SignEvent, SignState, StateUpdate
from .realtime_engine import RealtimeEngine, State, BUFFER_SIZE
from data_preprocess.normalized_np.main import normalized_batch

NORM_FEATURES = 108   # 12 body × 2 + 21 left-hand × 2 + 21 right-hand × 2

log = logging.getLogger("M3")

# Hand rows in the (65, 2) landmarks_raw array
_HAND_ROWS = slice(23, 65)

# Fraction of hand-landmark rows that must be non-zero to classify a frame Active.
# 0.1 means at least ~4 of the 42 hand-landmark rows must have a detected position.
DEFAULT_PRESENCE_THRESHOLD = 0.1


class M3StateMachine:
    """
    Pipeline-facing M3 module.

    Parameters
    ----------
    ta : int
        Consecutive active frames required to enter Active state (default 5).
    tr : int
        Consecutive rest frames required to fire inference (default 10).
    presence_threshold : float
        Fraction of hand-landmark rows that must be non-zero to classify a frame
        Active (default 0.1 ≈ ~4 of 42 rows).
    """

    def __init__(
        self,
        ta: int = 5,
        tr: int = 10,
        presence_threshold: float = DEFAULT_PRESENCE_THRESHOLD,
    ) -> None:
        self._engine = RealtimeEngine(ta=ta, tr=tr, inference_callback=self._on_trigger)
        self._presence_threshold = presence_threshold

        # Sign-level tracking
        self._sign_id_counter: int = 0
        self._current_sign_id: Optional[int] = None
        self._sign_start_ts: Optional[float] = None
        self._last_active_ts: Optional[float] = None
        self._frames_since_onset: int = 0

        # Trigger handoff to main loop
        self._triggered: bool = False
        self._triggered_buffer: Optional[list] = None


    # ── Public API (pipeline loop) ─────────────────────────────────────────

    def update(
        self,
        packet: FramePacket,
        is_active_override: Optional[bool] = None,
    ) -> StateUpdate:
        """
        Process one frame. Returns a StateUpdate for M2.
        When StateUpdate.triggered is True, call take_buffer() immediately.

        is_active_override: when provided, skips landmark-based classification
        and uses this value directly. Use this when M1 supplies its own H/A
        signal or when running without landmarks (motion-only stub).
        """
        self._triggered = False

        # H/A classification
        if is_active_override is not None:
            score, is_active = 0.0, is_active_override
        else:
            score, is_active = self._classify(packet.landmarks_raw)

        log.debug("frame=%d  score=%.4f  is_active=%s  consec_active=%d  consec_rest=%d",
                  packet.frame_id, score, is_active,
                  self._engine._consecutive_active, self._engine._consecutive_rest)

        prev_engine_state = self._engine.state
        self._engine.feed_frame(packet.landmarks_raw, is_active)
        curr_engine_state = self._engine.state

        # Detect IDLE → ACTIVE onset
        if prev_engine_state == State.IDLE and curr_engine_state == State.ACTIVE:
            self._sign_id_counter += 1
            self._current_sign_id = self._sign_id_counter
            self._sign_start_ts = packet.capture_ts
            self._frames_since_onset = 0
            log.info("▶ SIGN #%d STARTED  (frame=%d, TA=%d met)",
                     self._current_sign_id, packet.frame_id, self._engine.ta)

        if curr_engine_state == State.ACTIVE:
            self._frames_since_onset += 1
            if is_active:
                self._last_active_ts = packet.capture_ts
            buf_len = len(self._engine.buffer)
            if buf_len > 0 and buf_len % 10 == 0 and is_active:
                log.info("  buffer filling… %d/%d frames", buf_len, BUFFER_SIZE)

        # Capture sign_id before the trigger may reset it
        sign_id_for_update = self._current_sign_id

        sign_event: Optional[SignEvent] = None
        if self._triggered and self._current_sign_id is not None:
            start_ts = self._sign_start_ts or packet.capture_ts
            end_ts = self._last_active_ts or packet.capture_ts
            sign_event = SignEvent(
                sign_id=self._current_sign_id,
                sign_start_ts=start_ts,
                sign_end_ts=end_ts,
                sign_duration_s=max(0.0, end_ts - start_ts),
                buffer_length=len(self._triggered_buffer or []),
            )
            log.info("■ SIGN #%d COMPLETE  duration=%.2fs  buffer=%d frames",
                     sign_event.sign_id, sign_event.sign_duration_s, sign_event.buffer_length)
            self._current_sign_id = None
            self._sign_start_ts = None
            self._last_active_ts = None
            self._frames_since_onset = 0

        return StateUpdate(
            frame_id=packet.frame_id,
            state=SignState.ACTIVE if curr_engine_state == State.ACTIVE else SignState.IDLE,
            active_frame_count=self._frames_since_onset,
            sign_id=sign_id_for_update,
            triggered=self._triggered,
            sign_event=sign_event,
        )

    def take_buffer(self, pad_to: int = BUFFER_SIZE) -> np.ndarray:
        """
        Consume the buffer snapshot from the last trigger and return a normalized
        (pad_to, 108) float32 array ready for model inference.

        Must be called in the same loop iteration that saw triggered=True.
        Returns a zero array of shape (pad_to, 108) if called out of order.
        """
        raw_list = self._triggered_buffer or []
        self._triggered_buffer = None

        valid = [f for f in raw_list if f is not None]
        if not valid:
            return np.zeros((pad_to, NORM_FEATURES), dtype=np.float32)

        # (N, 65, 2) → (N, 130) then normalize via normalized_batch → (N, 108)
        flat = np.stack(valid, axis=0).reshape(len(valid), 130).astype(np.float32)
        out = normalized_batch(flat)  # calls select_features + body + hand norm

        # Pad / truncate to pad_to frames
        n = len(out)
        if n >= pad_to:
            return out[:pad_to]
        padded = np.zeros((pad_to, NORM_FEATURES), dtype=np.float32)
        padded[:n] = out
        return padded

    # ── Error instrumentation ────────────────────────────────────────────

    def save_trigger_log(self, path: str = "trigger_error_log.csv") -> None:
        self._engine.save_trigger_log(path)

    def error_summary(self) -> dict:
        return self._engine.error_summary()

    def set_total_signs(self, n: int) -> None:
        self._engine.set_total_signs(n)

    def mark_false_start(self) -> None:
        self._engine.mark_false_start()

    def mark_missed_sign(self, frame_start: int, frame_end: int) -> None:
        self._engine.mark_missed_sign(frame_start, frame_end)

    @property
    def ta(self) -> int:
        return self._engine.ta

    @property
    def tr(self) -> int:
        return self._engine.tr

    # ── Internals ─────────────────────────────────────────────────────────

    def _on_trigger(self, buffer: list) -> None:
        self._triggered = True
        self._triggered_buffer = list(buffer)
        log.info("■ TRIGGER FIRED  buffer_length=%d", len(buffer))

    def _classify(self, raw: Optional[np.ndarray]) -> tuple[float, bool]:
        """
        Compute hand-presence score and H/A label from landmarks_raw.
        Returns (score, is_active).
        """
        if raw is None:
            return 0.0, False

        score = self._hand_presence_score(raw)
        return score, score > self._presence_threshold

    def _hand_presence_score(self, current: np.ndarray) -> float:
        """
        Fraction of hand-landmark rows [23:65] with at least one non-zero coordinate.
        Returns a value in [0.0, 1.0]; 0.0 means no hand detected.
        """
        hand = current[_HAND_ROWS]
        detected_rows = np.any(hand != 0, axis=1)
        return float(np.mean(detected_rows))


