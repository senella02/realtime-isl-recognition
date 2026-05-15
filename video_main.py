"""
video_main.py — ISL recognition pipeline for MP4 video input.

Key differences from main.py (webcam):
  - Input:   cv2.VideoCapture(path) instead of webcam index
  - Buffer:  ALL active frames accumulated; linearly interpolated to 64 at sign-end
  - Trigger: TR consecutive rest frames (replaces the 64-frame buffer-full cutoff)
  - No FPS pacing — frames processed as fast as MediaPipe allows
  - EOF flush: last sign fired even if TR never completes

Run:  python video_main.py --video path/to/file.mp4
      python video_main.py --video path/to/file.mp4 --ta 5 --tr 10 --no-display
"""

import argparse
import csv
import logging
import time
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from contract.contracts import FramePacket, Prediction, SignEvent, SignState, StateUpdate
from data_preprocess.normalized_np.main import normalized_batch
from extractor.mediapipe_pipeline import LandmarkExtractor
from spoter.realtime_engine import SignLanguageEngine
from ui_render.output import M2Output

# ── Constants ─────────────────────────────────────────────────────────────────

BUFFER_SIZE = 64
MAX_SIGN_FRAMES = 512       # safety cap: force-fire if a sign exceeds this many frames
PRESENCE_THRESHOLD = 0.1   # fraction of hand rows [23:65] that must be non-zero → active

_DEFAULT_MODEL = Path(__file__).parent / "models" / "holistic_landmarker.task"
MODEL_PATH = "spoter/spoter_model_final.pt"
LABEL_PATH = "spoter/label_map.json"

log = logging.getLogger("video_main")


# ── Linear interpolation ──────────────────────────────────────────────────────

def _interp_to_64(flat: np.ndarray, target: int = BUFFER_SIZE) -> np.ndarray:
    """
    Resize an (N, 130) frame sequence to (target, 130) via per-column linear
    interpolation along the time axis. Handles both squeeze (N > target) and
    expand (N < target). N == target returns a copy without allocating.
    """
    n = flat.shape[0]
    if n == target:
        return flat.copy()
    src = np.arange(n, dtype=np.float64)
    tgt = np.linspace(0, n - 1, target)
    out = np.empty((target, flat.shape[1]), dtype=np.float32)
    for col in range(flat.shape[1]):
        out[:, col] = np.interp(tgt, src, flat[:, col])
    return out


# ── Video-file M1 ─────────────────────────────────────────────────────────────

class VideoLandmarkExtractor:
    """
    M1 for video-file input. Extraction logic identical to LandmarkExtractor;
    timestamps derived from frame number so MediaPipe always sees a monotonic stream.
    """

    def __init__(
        self,
        video_path: str,
        model_path: str | Path = _DEFAULT_MODEL,
    ) -> None:
        options = mp_vision.HolisticLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
        )
        self._holistic = mp_vision.HolisticLandmarker.create_from_options(options)
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self._fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_id: int = 0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def read_frame(self) -> tuple[Optional[np.ndarray], float]:
        """Returns (frame_bgr, capture_ts) or (None, ts) at end of file."""
        capture_ts = time.perf_counter()
        ret, frame = self._cap.read()
        if not ret:
            return None, capture_ts
        return frame, capture_ts

    def extract(self, frame: np.ndarray, capture_ts: float) -> FramePacket:
        self._frame_id += 1
        # Frame-number timestamp guarantees monotonicity regardless of processing speed.
        timestamp_ms = int(self._frame_id * (1000.0 / self._fps))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._holistic.detect_for_video(mp_image, timestamp_ms)

        if not results.pose_landmarks:
            return FramePacket(
                frame_id=self._frame_id,
                capture_ts=capture_ts,
                image_bgr=frame,
                bbox=None,
                landmarks_raw=None,
            )

        pose = results.pose_landmarks
        h, w = frame.shape[:2]
        return FramePacket(
            frame_id=self._frame_id,
            capture_ts=capture_ts,
            image_bgr=frame,
            bbox=LandmarkExtractor._compute_bbox(pose, w, h),
            landmarks_raw=LandmarkExtractor._build_landmarks_raw(
                pose, results.left_hand_landmarks, results.right_hand_landmarks
            ),
        )

    def release(self) -> None:
        self._cap.release()
        self._holistic.close()


# ── Video state machine ────────────────────────────────────────────────────────

class _State(Enum):
    IDLE = auto()
    ACTIVE = auto()


class VideoStateMachine:
    """
    Drop-in M3 replacement for video-file input.

    Trigger policy:
      - IDLE → ACTIVE after TA consecutive active frames.
      - ACTIVE → IDLE (fire inference) after TR consecutive rest frames.
      - Safety: also fires if accumulated frames exceed max_sign_frames.
      - EOF: call flush() to fire any in-progress sign at end of file.

    All active frames are kept in an unbounded accumulator; take_buffer()
    linearly interpolates them to exactly 64 frames before normalization.
    """

    def __init__(
        self,
        ta: int = 5,
        tr: int = 10,
        max_sign_frames: int = MAX_SIGN_FRAMES,
        presence_threshold: float = PRESENCE_THRESHOLD,
    ) -> None:
        self.ta = ta
        self.tr = tr
        self._max_sign_frames = max_sign_frames
        self._presence_threshold = presence_threshold

        self._state = _State.IDLE
        self._consec_active = 0
        self._consec_rest = 0
        self._accumulator: list = []

        self._sign_id_counter = 0
        self._current_sign_id: Optional[int] = None
        self._sign_start_ts: Optional[float] = None
        self._last_active_ts: Optional[float] = None
        self._frames_since_onset = 0

        self._triggered = False
        self._triggered_frames: Optional[list] = None

        # Error instrumentation
        self._inference_calls = 0
        self._total_signs = 0
        self._frame_idx = 0
        self._events: list = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, packet: FramePacket) -> StateUpdate:
        """Process one frame. Call take_buffer() immediately when triggered=True."""
        self._triggered = False
        self._triggered_frames = None
        self._frame_idx += 1

        is_active = self._classify(packet.landmarks_raw)
        prev_state = self._state

        if self._state == _State.IDLE:
            self._handle_idle(packet, is_active)
        else:
            self._handle_active(packet, is_active)

        if prev_state == _State.IDLE and self._state == _State.ACTIVE:
            self._sign_id_counter += 1
            self._current_sign_id = self._sign_id_counter
            self._sign_start_ts = packet.capture_ts
            self._frames_since_onset = 0
            log.info("▶ SIGN #%d STARTED  (frame=%d)", self._current_sign_id, self._frame_idx)

        if self._state == _State.ACTIVE:
            self._frames_since_onset += 1
            if is_active:
                self._last_active_ts = packet.capture_ts

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
                buffer_length=len(self._triggered_frames or []),
            )
            log.info(
                "■ SIGN #%d COMPLETE  duration=%.2fs  raw_frames=%d → interp→%d",
                sign_event.sign_id, sign_event.sign_duration_s,
                sign_event.buffer_length, BUFFER_SIZE,
            )
            self._current_sign_id = None
            self._sign_start_ts = None
            self._last_active_ts = None
            self._frames_since_onset = 0

        return StateUpdate(
            frame_id=packet.frame_id,
            state=SignState.ACTIVE if self._state == _State.ACTIVE else SignState.IDLE,
            active_frame_count=self._frames_since_onset,
            sign_id=sign_id_for_update,
            triggered=self._triggered,
            sign_event=sign_event,
        )

    def take_buffer(self) -> np.ndarray:
        """
        Consume the last trigger's frames.
        Returns a normalized (64, 108) float32 array ready for model inference.
        Returns zeros if called out of order.
        """
        raw = self._triggered_frames or []
        self._triggered_frames = None

        valid = [f for f in raw if f is not None]
        if not valid:
            return np.zeros((BUFFER_SIZE, 108), dtype=np.float32)

        # (N, 65, 2) → (N, 130) → interpolate → (64, 130) → normalize → (64, 108)
        flat = np.stack(valid, axis=0).reshape(len(valid), 130).astype(np.float32)
        interped = _interp_to_64(flat, BUFFER_SIZE)
        return normalized_batch(interped)

    def flush(self, packet: FramePacket) -> Optional[StateUpdate]:
        """
        Force-fire inference for any sign still in progress at EOF.
        Returns a StateUpdate with triggered=True, or None if already IDLE.
        Call take_buffer() after this when triggered=True.
        """
        if self._state != _State.ACTIVE or not self._accumulator:
            return None

        log.info(
            "EOF flush: firing inference for SIGN #%d  (%d frames accumulated)",
            self._current_sign_id, len(self._accumulator),
        )
        self._fire_inference("eof_flush")

        sign_id_for_update = self._current_sign_id
        start_ts = self._sign_start_ts or packet.capture_ts
        end_ts = self._last_active_ts or packet.capture_ts
        sign_event = SignEvent(
            sign_id=self._current_sign_id or 0,
            sign_start_ts=start_ts,
            sign_end_ts=end_ts,
            sign_duration_s=max(0.0, end_ts - start_ts),
            buffer_length=len(self._triggered_frames or []),
        )
        self._current_sign_id = None
        self._sign_start_ts = None
        self._last_active_ts = None
        self._frames_since_onset = 0

        return StateUpdate(
            frame_id=packet.frame_id,
            state=SignState.IDLE,
            active_frame_count=0,
            sign_id=sign_id_for_update,
            triggered=True,
            sign_event=sign_event,
        )

    def set_total_signs(self, n: int) -> None:
        self._total_signs = n

    def mark_false_start(self) -> None:
        if self._events:
            self._events[-1]["error_label"] = "false_start"

    def error_summary(self) -> dict:
        return {
            "total_inference_calls": self._inference_calls,
            "total_signs_gt": self._total_signs,
            "ta": self.ta,
            "tr": self.tr,
        }


    # ── Internals ──────────────────────────────────────────────────────────────

    def _handle_idle(self, packet: FramePacket, is_active: bool) -> None:
        if is_active:
            self._consec_active += 1
            self._consec_rest = 0
            if self._consec_active >= self.ta:
                self._state = _State.ACTIVE
                self._accumulator = [packet.landmarks_raw]
                self._consec_active = 0
                self._consec_rest = 0
                self._log_event("IDLE", "ACTIVE", f"TA={self.ta} met", buf_size=1)
        else:
            self._consec_active = 0
            self._consec_rest += 1

    def _handle_active(self, packet: FramePacket, is_active: bool) -> None:
        if is_active:
            self._consec_rest = 0
            self._consec_active += 1
            self._accumulator.append(packet.landmarks_raw)
            if len(self._accumulator) >= self._max_sign_frames:
                self._fire_inference(f"safety_cap ({self._max_sign_frames} frames)")
        else:
            self._consec_active = 0
            self._consec_rest += 1
            if self._consec_rest >= self.tr:
                self._fire_inference(f"TR={self.tr} rest frames")

    def _fire_inference(self, reason: str) -> None:
        buf_size = len(self._accumulator)
        self._log_event("ACTIVE", "IDLE", reason, buf_size)
        self._triggered = True
        self._triggered_frames = list(self._accumulator)
        self._state = _State.IDLE
        self._accumulator = []
        self._consec_active = 0
        self._consec_rest = 0
        self._inference_calls += 1

    def _log_event(
        self,
        from_state: str,
        to_state: str,
        reason: str,
        buf_size: Optional[int] = None,
    ) -> None:
        self._events.append({
            "timestamp": time.time(),
            "frame_idx": self._frame_idx,
            "from_state": from_state,
            "to_state": to_state,
            "buffer_size": buf_size if buf_size is not None else len(self._accumulator),
            "reason": reason,
            "error_label": "",
        })

    def _classify(self, raw: Optional[np.ndarray]) -> bool:
        if raw is None:
            return False
        hand = raw[23:65]
        score = float(np.mean(np.any(hand != 0, axis=1)))
        return score > self._presence_threshold


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ISL recognition on MP4 video")
    parser.add_argument("--video", required=True, help="Path to input .mp4 file")
    parser.add_argument("--ta", type=int, default=5, help="Onset frames (default 5)")
    parser.add_argument("--tr", type=int, default=10, help="Rest frames to trigger (default 10)")
    # parser.add_argument("--log", default="trigger_error_log_video.csv",
    #                     help="Path for trigger event CSV log")
    parser.add_argument("--no-display", action="store_true", help="Skip imshow")
    parser.add_argument("--display-scale", type=float, default=1.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d  %(name)-12s  %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Video ISL pipeline starting")
    log.info("  video=%s  TA=%d  TR=%d", args.video, args.ta, args.tr)
    log.info("-" * 60)

    m1 = VideoLandmarkExtractor(args.video)
    m3 = VideoStateMachine(ta=args.ta, tr=args.tr)
    m4 = SignLanguageEngine(MODEL_PATH, LABEL_PATH)
    m2 = M2Output(display=not args.no_display, display_scale=args.display_scale)

    log.info("Video: %.1f fps, %d frames total", m1.fps, m1.total_frames)

    last_packet: Optional[FramePacket] = None

    def _run_inference(se: SignEvent, video_name: str) -> Prediction:
        norm_buf = m3.take_buffer()
        result = m4.run_inference(norm_buf, video_name)
        pred = Prediction(
            sign_id=se.sign_id,
            inference_start_ts=result["inference_start_ts"],
            inference_end_ts=result["inference_end_ts"],
            probs=result["probs"],
            top_k_indices=result["top_k_indices"],
            top_k_probs=result["top_k_probs"],
            top_k_glosses=result["top_k_glosses"],
        )
        log.info(
            "→ M4: sign #%d  top=%s (%.1f%%)  raw=%d frames → %d  %.1f ms",
            se.sign_id,
            pred.top_k_glosses[0],
            pred.top_k_probs[0] * 100,
            se.buffer_length,
            BUFFER_SIZE,
            (result["inference_end_ts"] - result["inference_start_ts"]) * 1000,
        )
        return pred

    try:
        while True:
            frame, capture_ts = m1.read_frame()
            if frame is None:
                break

            packet = m1.extract(frame, capture_ts)
            last_packet = packet

            state_update = m3.update(packet)

            prediction = None
            if state_update.triggered:
                prediction = _run_inference(state_update.sign_event, args.video)

            m2.render(packet, state_update, prediction)
            m2.log(packet, state_update, prediction)

            if not args.no_display and cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Fire inference for any sign still active at EOF (or after Q-quit).
        if last_packet is not None:
            flush_update = m3.flush(last_packet)
            if flush_update is not None:
                prediction = _run_inference(flush_update.sign_event, args.video)
                m2.render(last_packet, flush_update, prediction)
                m2.log(last_packet, flush_update, prediction)

        m1.release()
        cv2.destroyAllWindows()

    #m3.save_trigger_log(args.log)
    log.info("-" * 60)
    log.info("Session ended — summary:")
    for k, v in m3.error_summary().items():
        log.info("  %s: %s", k, v)
    #log.info("Trigger log saved → %s", args.log)


if __name__ == "__main__":
    main()
