from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from contract.contracts import FramePacket
from extractor.mediapipe_pipeline import LandmarkExtractor

_DEFAULT_MODEL = Path(__file__).parent.parent / "models" / "holistic_landmarker.task"


class VideoLandmarkExtractor(LandmarkExtractor):
    """
    Drop-in replacement for LandmarkExtractor that reads from an .mp4 file
    instead of the webcam. All landmark extraction logic is inherited unchanged.

    capture_ts is derived from the frame position in the video (frame_idx / fps)
    so MediaPipe's monotonic timestamp requirement is always satisfied.
    """

    def __init__(
        self,
        video_path: str | Path,
        model_path: str | Path = _DEFAULT_MODEL,
    ) -> None:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        options = mp_vision.HolisticLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
        )
        self._holistic = mp_vision.HolisticLandmarker.create_from_options(options)

        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self._fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_id: int = 0
        self._start_ts: float = 0.0   # video time starts at 0

        print(
            f"[VideoLandmarkExtractor] {video_path.name}  "
            f"fps={self._fps:.1f}  frames={self._total_frames}"
        )

    # capture_ts = frame_index / fps  →  increases monotonically by 1/fps per frame
    def grab(self) -> tuple[Optional[np.ndarray], float]:
        capture_ts = self._frame_id / self._fps
        ret, frame = self._cap.read()
        if not ret:
            return None, capture_ts
        return frame, capture_ts

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames
