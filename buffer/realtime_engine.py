"""
realtime_engine.py — M3 Phase 2: Real-Time Sign Segmentation Engine
Sliding 64-frame buffer, A/H state machine, inference trigger + full error instrumentation.

Interfaces:
  Input  (M1): per-frame is_active bool + feature frame
  Output (M4): filled 64-frame buffer via inference_callback on sign-end
"""

from collections import deque
from dataclasses import dataclass, asdict
from enum import Enum, auto
import csv
import time
from typing import Callable, List, Optional, Tuple

# ── Defaults (sweep ±1 for sensitivity analysis) ──────────────────────────
DEFAULT_TA = 5   # consecutive active frames needed to enter Active state
DEFAULT_TR = 10  # consecutive rest frames needed to fire inference
BUFFER_SIZE = 64


class State(Enum):
    IDLE = auto()    # H — hand at rest / not signing
    ACTIVE = auto()  # A — sign in progress, filling buffer


@dataclass
class TriggerEvent:
    timestamp: float
    frame_idx: int
    from_state: str
    to_state: str
    buffer_size: int
    reason: str
    error_label: str  # "" | "false_start" | "missed" | "premature" | "over_segmented"


class RealtimeEngine:
    """
    Sign segmentation engine.

    Usage:
        engine = RealtimeEngine(ta=5, tr=10, inference_callback=m4_infer)
        for frame, is_active in stream:
            engine.feed_frame(frame, is_active)
        engine.save_trigger_log("trigger_error_log.csv")
    """

    def __init__(
        self,
        ta: int = DEFAULT_TA,
        tr: int = DEFAULT_TR,
        inference_callback: Optional[Callable[[list], None]] = None,
    ) -> None:
        self.ta = ta
        self.tr = tr
        self.inference_callback = inference_callback

        self.state = State.IDLE
        self.buffer: deque = deque(maxlen=BUFFER_SIZE)
        self._consecutive_active = 0
        self._consecutive_rest = 0
        self.frame_idx = 0

        # Error counters
        self._false_start_count = 0
        self._missed_sign_count = 0
        self._premature_trigger_count = 0
        self._over_seg_count = 0
        self._inference_calls = 0
        self._total_signs = 0  # set externally via set_total_signs()

        self._events: List[TriggerEvent] = []
        self._last_trigger_frame: Optional[int] = None

    # ── Public API ────────────────────────────────────────────────────────

    def feed_frame(self, frame, is_active: bool) -> None:
        """
        Process one frame.
          frame      — feature vector / raw frame passed through to M4
          is_active  — H/A label from M1 (True = Active / signing)
        """
        self.frame_idx += 1
        if self.state == State.IDLE:
            self._handle_idle(frame, is_active)
        else:
            self._handle_active(frame, is_active)

    def set_total_signs(self, n: int) -> None:
        """Supply ground-truth sign count so missed/over-seg rates are meaningful."""
        self._total_signs = n

    def mark_false_start(self) -> None:
        """Mark the most recent inference call as a false start (evaluator/M2 calls this)."""
        self._false_start_count += 1
        if self._events:
            last = self._events[-1]
            self._events[-1] = TriggerEvent(
                **{**asdict(last), "error_label": "false_start"}
            )

    def mark_missed_sign(self, frame_start: int, frame_end: int) -> None:
        """Log a sign that was performed but never triggered inference."""
        self._missed_sign_count += 1
        self._events.append(TriggerEvent(
            timestamp=time.time(),
            frame_idx=frame_end,
            from_state="IDLE",
            to_state="IDLE",
            buffer_size=0,
            reason=f"missed sign frames {frame_start}–{frame_end}",
            error_label="missed",
        ))

    def error_summary(self) -> dict:
        """Return counts and rates for all 4 error types."""
        n_triggers = max(self._inference_calls, 1)
        n_signs = max(self._total_signs, 1)
        return {
            "false_start_count": self._false_start_count,
            "false_start_rate_pct": round(100 * self._false_start_count / n_triggers, 2),
            "missed_sign_count": self._missed_sign_count,
            "missed_sign_rate_pct": round(100 * self._missed_sign_count / n_signs, 2),
            "premature_trigger_count": self._premature_trigger_count,
            "premature_trigger_rate_pct": round(100 * self._premature_trigger_count / n_triggers, 2),
            "over_segmentation_count": self._over_seg_count,
            "over_segmentation_rate_pct": round(100 * self._over_seg_count / n_signs, 2),
            "total_inference_calls": self._inference_calls,
            "total_signs_gt": self._total_signs,
            "ta": self.ta,
            "tr": self.tr,
        }

    def get_events(self) -> List[TriggerEvent]:
        return list(self._events)

    def save_trigger_log(self, path: str = "trigger_error_log.csv") -> None:
        """Write per-segment state-transition log to CSV."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "frame_idx", "from_state", "to_state",
                "buffer_size", "reason", "error_label",
            ])
            for e in self._events:
                writer.writerow([
                    f"{e.timestamp:.4f}", e.frame_idx, e.from_state, e.to_state,
                    e.buffer_size, e.reason, e.error_label,
                ])

    # ── State machine internals ───────────────────────────────────────────

    def _handle_idle(self, frame, is_active: bool) -> None:
        if is_active:
            self._consecutive_active += 1
            self._consecutive_rest = 0
            if self._consecutive_active >= self.ta:
                self._transition(State.ACTIVE, reason=f"TA={self.ta} consecutive active frames")
                self.buffer.append(frame)
        else:
            self._consecutive_active = 0
            self._consecutive_rest += 1

    def _handle_active(self, frame, is_active: bool) -> None:
        if is_active:
            self._consecutive_rest = 0
            self._consecutive_active += 1
            self.buffer.append(frame)
        else:
            self._consecutive_active = 0
            self._consecutive_rest += 1
            if self._consecutive_rest >= self.tr:
                self._fire_inference()

    def _transition(
        self, new_state: State, reason: str, error_label: str = ""
    ) -> None:
        self._events.append(TriggerEvent(
            timestamp=time.time(),
            frame_idx=self.frame_idx,
            from_state=self.state.name,
            to_state=new_state.name,
            buffer_size=len(self.buffer),
            reason=reason,
            error_label=error_label,
        ))
        self.state = new_state

    def _fire_inference(self) -> None:
        """Active → Idle: snapshot buffer, detect errors, call M4, reset."""
        buf_snapshot = list(self.buffer)
        buf_len = len(buf_snapshot)
        error_label = ""

        # Premature: buffer less than half-full when rest detected
        if buf_len < BUFFER_SIZE // 2:
            self._premature_trigger_count += 1
            error_label = "premature"

        # Over-segmentation: previous trigger was too recent (gap < TA + TR)
        if self._last_trigger_frame is not None:
            gap = self.frame_idx - self._last_trigger_frame
            if gap < self.ta + self.tr:
                self._over_seg_count += 1
                error_label = error_label or "over_segmented"

        self._transition(
            State.IDLE,
            reason=f"TR={self.tr} consecutive rest frames → inference fired",
            error_label=error_label,
        )
        self._last_trigger_frame = self.frame_idx
        self._inference_calls += 1

        # Reset for next sign
        self.buffer.clear()
        self._consecutive_active = 0
        self._consecutive_rest = 0

        if self.inference_callback and buf_snapshot:
            self.inference_callback(buf_snapshot)


# ── TR/TA Threshold Sweep ─────────────────────────────────────────────────

def sweep_thresholds(
    frames_with_labels: List[Tuple],  # list of (frame, is_active)
    ground_truth_signs: int,
    ta_default: int = DEFAULT_TA,
    tr_default: int = DEFAULT_TR,
    delta: int = 1,
) -> List[dict]:
    """
    Sweep TA and TR ±delta from defaults.
    Returns a list of error_summary dicts, one per (TA, TR) combination.
    """
    results = []
    for ta in range(ta_default - delta, ta_default + delta + 1):
        for tr in range(tr_default - delta, tr_default + delta + 1):
            if ta < 1 or tr < 1:
                continue
            engine = RealtimeEngine(ta=ta, tr=tr)
            engine.set_total_signs(ground_truth_signs)
            for frame, is_active in frames_with_labels:
                engine.feed_frame(frame, is_active)
            results.append(engine.error_summary())
    return results


def print_sensitivity_table(sweep_results: List[dict]) -> None:
    header = (
        f"{'TA':>4} {'TR':>4} | "
        f"{'FalseStart%':>12} {'Missed%':>9} {'Premature%':>11} {'OverSeg%':>9} | "
        f"{'Triggers':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in sweep_results:
        print(
            f"{r['ta']:>4} {r['tr']:>4} | "
            f"{r['false_start_rate_pct']:>12.1f} "
            f"{r['missed_sign_rate_pct']:>9.1f} "
            f"{r['premature_trigger_rate_pct']:>11.1f} "
            f"{r['over_segmentation_rate_pct']:>9.1f} | "
            f"{r['total_inference_calls']:>9}"
        )


# ── Entrypoint (integration test / demo) ─────────────────────────────────

if __name__ == "__main__":
    import random

    random.seed(42)

    # Simulate a stream: 3 signs embedded in noise
    # Each sign = 80 active frames; gaps = 20 rest frames between signs
    def make_stream(n_signs: int = 3):
        stream = []
        for _ in range(20):           # leading rest
            stream.append((None, False))
        for _ in range(n_signs):
            for _ in range(80):       # sign active frames
                stream.append((None, True))
            for _ in range(20):       # rest after sign
                stream.append((None, False))
        return stream

    stream = make_stream(n_signs=3)

    received_buffers = []
    engine = RealtimeEngine(
        ta=DEFAULT_TA,
        tr=DEFAULT_TR,
        inference_callback=lambda buf: received_buffers.append(len(buf)),
    )
    engine.set_total_signs(3)

    for frame, is_active in stream:
        engine.feed_frame(frame, is_active)

    engine.save_trigger_log("trigger_error_log.csv")

    print("=== Error Summary ===")
    for k, v in engine.error_summary().items():
        print(f"  {k}: {v}")

    print(f"\nInference calls received buffers of size: {received_buffers}")

    print("\n=== TR/TA Sensitivity Sweep ===")
    sweep = sweep_thresholds(stream, ground_truth_signs=3)
    print_sensitivity_table(sweep)
