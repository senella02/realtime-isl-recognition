"""
M3/engine.py — contracts-compatible wrapper around RealtimeEngine.

Adapts the raw A/H state machine to the shared dataclasses in contract/contracts.py
so the main pipeline loop can call:

    state_update = m3.update(packet)          # every frame
    if state_update.triggered:
        buf = m3.take_buffer()
        prediction = m4.infer(state_update.sign_event, buf)

Design notes:
- H/A classification is computed internally from hand-landmark motion because
  FramePacket does not carry an is_active field. When M1 is extended to provide
  this signal directly, replace _compute_is_active() with that value.
- The buffer contains landmarks_normalized dicts — exactly what M4 expects.
- All error-instrumentation methods (save_trigger_log, error_summary, etc.) are
  forwarded to the underlying RealtimeEngine unchanged.
"""

from typing import Optional

from contract.contracts import FramePacket, SignEvent, SignState, StateUpdate
from .realtime_engine import RealtimeEngine, State

# Mean absolute displacement threshold for H/A classification (normalized coords).
# Calibrate with real clips; sweep via error_summary() after a session.
DEFAULT_MOTION_THRESHOLD = 0.02


class M3StateMachine:
    """
    Pipeline-facing M3 module.

    Parameters
    ----------
    ta : int
        Consecutive active frames required to enter Active state (default 5).
    tr : int
        Consecutive rest frames required to fire inference (default 10).
    motion_threshold : float
        Hand-motion score cutoff for H/A classification (default 0.02).
        Tune upward to ignore fidgeting; tune downward to catch slow signs.
    """

    def __init__(
        self,
        ta: int = 5,
        tr: int = 10,
        motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
    ) -> None:
        self._engine = RealtimeEngine(ta=ta, tr=tr, inference_callback=self._on_trigger)
        self._motion_threshold = motion_threshold

        # Sign-level tracking
        self._sign_id_counter: int = 0
        self._current_sign_id: Optional[int] = None
        self._sign_start_ts: Optional[float] = None
        self._last_active_ts: Optional[float] = None
        self._frames_since_onset: int = 0

        # Trigger handoff to main loop
        self._triggered: bool = False
        self._triggered_buffer: Optional[list] = None

        # Previous-frame landmarks for motion score
        self._prev_landmarks: Optional[dict] = None

    # ── Public API (pipeline loop) ─────────────────────────────────────────

    def update(self, packet: FramePacket) -> StateUpdate:
        """
        Process one frame. Returns a StateUpdate for M2.
        When StateUpdate.triggered is True, call take_buffer() immediately.
        """
        self._triggered = False

        is_active = self._compute_is_active(packet)

        prev_engine_state = self._engine.state
        self._engine.feed_frame(packet.landmarks_normalized, is_active)
        curr_engine_state = self._engine.state

        # Detect IDLE → ACTIVE onset
        if prev_engine_state == State.IDLE and curr_engine_state == State.ACTIVE:
            self._sign_id_counter += 1
            self._current_sign_id = self._sign_id_counter
            self._sign_start_ts = packet.capture_ts
            self._frames_since_onset = 0

        if curr_engine_state == State.ACTIVE:
            self._frames_since_onset += 1
            if is_active:
                self._last_active_ts = packet.capture_ts

        # Capture sign_id before the trigger resets it
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
            # Reset sign tracking after trigger; next sign gets a fresh id
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

    def take_buffer(self) -> list:
        """
        Consume the 64-frame buffer snapshot from the last trigger.
        Must be called in the same loop iteration that saw triggered=True.
        Returns an empty list if called out of order.
        """
        buf = self._triggered_buffer or []
        self._triggered_buffer = None
        return buf

    # ── Error instrumentation (delegate to RealtimeEngine) ────────────────

    def save_trigger_log(self, path: str = "trigger_error_log.csv") -> None:
        """Write per-segment state-transition log to CSV (M3 deliverable)."""
        self._engine.save_trigger_log(path)

    def error_summary(self) -> dict:
        """Return all 4 error counts + rates. Use for boundary_error_summary.md."""
        return self._engine.error_summary()

    def set_total_signs(self, n: int) -> None:
        """Supply ground-truth sign count for missed/over-seg rate computation."""
        self._engine.set_total_signs(n)

    def mark_false_start(self) -> None:
        """Mark the most recent trigger as a false start. Called by M2 evaluator."""
        self._engine.mark_false_start()

    def mark_missed_sign(self, frame_start: int, frame_end: int) -> None:
        """Log a sign that was performed but never triggered inference."""
        self._engine.mark_missed_sign(frame_start, frame_end)

    @property
    def ta(self) -> int:
        return self._engine.ta

    @property
    def tr(self) -> int:
        return self._engine.tr

    # ── Internals ─────────────────────────────────────────────────────────

    def _on_trigger(self, buffer: list) -> None:
        """Callback wired into RealtimeEngine. Fires on Active→Idle transition."""
        self._triggered = True
        self._triggered_buffer = list(buffer)

    def _compute_is_active(self, packet: FramePacket) -> bool:
        """
        Classify this frame as Active (signing) or Idle (resting) based on
        the mean absolute displacement of normalized hand landmarks vs the
        previous frame.

        When M1 is extended to expose an explicit is_active signal in FramePacket,
        replace this method body with: return packet.is_active
        """
        if packet.landmarks_normalized is None:
            self._prev_landmarks = None
            return False

        score = self._hand_motion_score(packet.landmarks_normalized)
        self._prev_landmarks = packet.landmarks_normalized
        return score > self._motion_threshold

    def _hand_motion_score(self, current: dict) -> float:
        """
        Mean absolute displacement of hand landmarks between this frame and the
        previous frame. Hand landmarks are identified by their '_0' (left) or
        '_1' (right) key suffix in the normalized dict.

        Returns 0.0 on the first frame or when no hand landmarks are found.
        """
        if self._prev_landmarks is None:
            return 0.0

        total = 0.0
        n = 0
        for key, val in current.items():
            if not (key.endswith("_0") or key.endswith("_1")):
                continue
            prev_val = self._prev_landmarks.get(key)
            if prev_val is None or not isinstance(val, (list, tuple)):
                continue
            for c, p in zip(val, prev_val):
                try:
                    total += abs(float(c) - float(p))
                    n += 1
                except (TypeError, ValueError):
                    continue
        return total / n if n > 0 else 0.0
