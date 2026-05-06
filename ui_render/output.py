"""M2Output — thin facade called from the main pipeline loop."""

from typing import Optional
import numpy as np
import cv2

from contract.contracts import FramePacket, Prediction, StateUpdate
from ui_render.ui_overlay import OverlayRenderer
from ui_render.runtime.fps_meter import FpsMeter


class M2Output:
    """
    Exposes render() and log() — both must return in < 5 ms on average.
    Heavy offline work (Phase D/E) is never called from the live loop.
    """

    def __init__(self, display: bool = True):
        self._display = display
        self._renderer = OverlayRenderer()
        self._fps = FpsMeter(window=30)
        self._latest_prediction: Optional[Prediction] = None

    def render(
        self,
        packet: FramePacket,
        state_update: StateUpdate,
        prediction: Optional[Prediction],
    ) -> np.ndarray:
        """Draw overlay and show (or return) the annotated frame."""
        self._fps.tick()
        if prediction is not None:
            self._latest_prediction = prediction

        annotated = self._renderer.draw(
            frame=packet.image_bgr,
            frame_packet=packet,
            state_update=state_update,
            latest_prediction=self._latest_prediction,
            fps=self._fps.fps(),
        )

        if self._display:
            cv2.imshow("ISL Recognition", annotated)
            cv2.waitKey(1)

        return annotated

    def log(
        self,
        packet: FramePacket,
        state_update: StateUpdate,
        prediction: Optional[Prediction],
    ) -> None:
        """Phase B will fill this out. Stub here to keep the call graph intact."""
        pass
