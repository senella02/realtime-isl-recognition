from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from contract.contracts import FramePacket

_DEFAULT_MODEL = Path(__file__).parent.parent / "models" / "holistic_landmarker.task"
_VIS_THRESHOLD: float = 0.5
_UPPER_BODY_COUNT: int = 23   # MediaPipe pose indices 0–22 (upper body only)


class LandmarkExtractor:
    """
    Wraps MediaPipe HolisticLandmarker for single-threaded, per-frame extraction.

    Responsibilities:
      - Webcam capture (grab)
      - Extract 65 landmarks as (65, 2) numpy array: upper-body(23) + left(21) + right(21)
      - Adaptive body bounding box (Paper Eqs. 4-6)
    """

    def __init__(
        self,
        model_path: str | Path = _DEFAULT_MODEL,
        camera_index: int = 0,
        target_fps: int = 60,
    ) -> None:
        options = mp_vision.HolisticLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
        )
        self._holistic = mp_vision.HolisticLandmarker.create_from_options(options)

        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {camera_index}")

        # Request target FPS from the webcam driver.
        # cap.read() blocks naturally at the camera's frame rate, so this also
        # paces grab() without any manual sleep.
        self._cap.set(cv2.CAP_PROP_FPS, target_fps)
        # Buffer size 1 — always return the latest frame, not a stale queued one.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"[LandmarkExtractor] requested FPS={target_fps}, "
              f"camera reports FPS={actual_fps:.1f}")
        if actual_fps < target_fps * 0.9:
            print(f"[LandmarkExtractor] WARNING: hardware may not support {target_fps} FPS")

        self._frame_id: int = 0
        self._start_ts: float = time.perf_counter()

    # ── public API ────────────────────────────────────────────────────────────

    def grab(self) -> tuple[Optional[np.ndarray], float]:
        """
        Capture one frame from the webcam.
        Returns (frame_bgr, capture_ts). frame_bgr is None on read failure.
        capture_ts = time.perf_counter() at moment of capture.
        """
        capture_ts = time.perf_counter()
        ret, frame = self._cap.read()
        if not ret:
            return None, capture_ts
        return frame, capture_ts

    def extract(self, frame: np.ndarray, capture_ts: float) -> FramePacket:
        """
        Process one BGR frame.

        Undetected body landmarks (visibility <= 0.5) stored as [0.0, 0.0].
        Hand landmarks stored as-is; undetected hands stored as all zeros.
        Returns FramePacket with None landmarks/bbox if no person detected.
        """
        self._frame_id += 1
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((capture_ts - self._start_ts) * 1000)

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
        landmarks_raw = self._build_landmarks_raw(
            pose, results.left_hand_landmarks, results.right_hand_landmarks
        )
        bbox = self._compute_bbox(pose, results.left_hand_landmarks, results.right_hand_landmarks, w, h)

        return FramePacket(
            frame_id=self._frame_id,
            capture_ts=capture_ts,
            image_bgr=frame,
            bbox=bbox,
            landmarks_raw=landmarks_raw,
        )

    def release(self) -> None:
        """Release webcam and MediaPipe resources. Call on shutdown."""
        self._cap.release()
        self._holistic.close()

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_landmarks_raw(pose, left_hand, right_hand) -> np.ndarray:
        """
        Pack 65 landmarks into a (65, 2) float32 array.
        Body: indices 0–22 only (upper body). Lower body (23–32) discarded.
        Hands: all 21 joints each, stored as-is (no visibility filter).
        """
        arr = np.zeros((65, 2), dtype=np.float32)

        # upper body: pose indices 0–22
        for i in range(_UPPER_BODY_COUNT):
            lm = pose[i]
            vis = lm.visibility if lm.visibility is not None else 0.0
            if vis > _VIS_THRESHOLD:
                arr[i] = [lm.x, lm.y]

        # left hand: indices 23–43
        if left_hand:
            for i, lm in enumerate(left_hand):
                arr[23 + i] = [lm.x, lm.y]

        # right hand: indices 44–64
        if right_hand:
            for i, lm in enumerate(right_hand):
                arr[44 + i] = [lm.x, lm.y]

        return arr

    @staticmethod
    def _compute_bbox(pose, left_hand, right_hand, frame_w: int, frame_h: int) -> Optional[tuple]:
        """Adaptive body bbox — formula.md. L = pose + left hand + right hand."""
        points: list[tuple[float, float]] = []

        for lm in pose:
            if lm.visibility is not None and lm.visibility > _VIS_THRESHOLD and lm.x != 0.0:
                points.append((lm.x, lm.y))

        for hand in (left_hand, right_hand):
            if hand:
                for lm in hand:
                    if lm.x != 0.0 or lm.y != 0.0:
                        points.append((lm.x, lm.y))

        if not points:
            return None

        x_min = min(x for x, _ in points)
        y_min = min(y for _, y in points)
        x_max = max(x for x, _ in points)
        y_max = max(y for _, y in points)

        pad_x    = (x_max - x_min) * 0.10
        pad_y    = (y_max - y_min) * 0.10
        pad_head = (y_max - y_min) * 0.20

        x1 = int(max(0,       (x_min - pad_x)    * frame_w))
        y1 = int(max(0,       (y_min - pad_head)  * frame_h))
        x2 = int(min(frame_w, (x_max + pad_x)    * frame_w))
        y2 = int(min(frame_h, (y_max + pad_y)    * frame_h))

        return (x1, y1, x2, y2)
