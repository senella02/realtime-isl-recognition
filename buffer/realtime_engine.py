"""
realtime_engine.py — M3 Phase 2: Real-Time Sign Segmentation Engine
Sliding 64-frame buffer, A/H state machine, inference trigger.

Interfaces:
  Input  (M1): per-frame is_active bool + feature frame
  Output (M4): filled 64-frame buffer via inference_callback on sign-end
"""

from collections import deque
from enum import Enum, auto
from typing import Callable, Optional

DEFAULT_TA = 5   # consecutive active frames needed to enter Active state
DEFAULT_TR = 10  # consecutive rest frames needed to fire inference
BUFFER_SIZE = 64


class State(Enum):
    IDLE = auto()
    ACTIVE = auto()


class RealtimeEngine:
    """
    Sign segmentation engine.

    Usage:
        engine = RealtimeEngine(ta=5, tr=10, inference_callback=m4_infer)
        for frame, is_active in stream:
            engine.feed_frame(frame, is_active)
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

    def feed_frame(self, frame, is_active: bool) -> None:
        self.frame_idx += 1
        if self.state == State.IDLE:
            self._handle_idle(frame, is_active)
        else:
            self._handle_active(frame, is_active)

    def _handle_idle(self, frame, is_active: bool) -> None:
        if is_active:
            self._consecutive_active += 1
            self._consecutive_rest = 0
            if self._consecutive_active >= self.ta:
                self.state = State.ACTIVE
                self.buffer.append(frame)
        else:
            self._consecutive_active = 0
            self._consecutive_rest += 1

    def _handle_active(self, frame, is_active: bool) -> None:
        if is_active:
            self._consecutive_rest = 0
            self._consecutive_active += 1
            self.buffer.append(frame)
            if len(self.buffer) >= BUFFER_SIZE:
                self._fire_inference()
        else:
            self._consecutive_active = 0
            self._consecutive_rest += 1

    def _fire_inference(self) -> None:
        buf_snapshot = list(self.buffer)
        self.state = State.IDLE
        self.buffer.clear()
        self._consecutive_active = 0
        self._consecutive_rest = 0
        if self.inference_callback and buf_snapshot:
            self.inference_callback(buf_snapshot)
