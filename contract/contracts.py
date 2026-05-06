from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np

CONTRACT_VERSION = "1.0.0"


class SignState(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"


@dataclass
class FramePacket:
    """Returned by M1 once per webcam frame. Consumed by M3 and M2."""
    frame_id: int
    capture_ts: float
    image_bgr: np.ndarray
    bbox: Optional[tuple]              # (x1, y1, x2, y2) in image coords; None if no body
    landmarks_raw: Optional[dict]
    landmarks_normalized: Optional[dict]


@dataclass
class SignEvent:
    """Produced by M3 on Active→Idle. Passed directly to M4. Also surfaced to M2 via StateUpdate."""
    sign_id: int
    sign_start_ts: float
    sign_end_ts: float
    sign_duration_s: float
    buffer_length: int


@dataclass
class StateUpdate:
    """Returned by M3 once per frame. Consumed by M2."""
    frame_id: int
    state: SignState
    active_frame_count: int
    sign_id: Optional[int]
    triggered: bool
    sign_event: Optional[SignEvent]


@dataclass
class Prediction:
    """Returned by M4 once per SignEvent. Passed directly to M2."""
    sign_id: int
    inference_start_ts: float
    inference_end_ts: float
    probs: np.ndarray                  # shape (184,)
    top_k_indices: list                # length 3, sorted desc by prob
    top_k_probs: list                  # length 3
    top_k_glosses: list                # length 3
