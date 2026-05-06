"""Phase A — UI overlay drawn on each webcam frame via OpenCV."""

from typing import Optional

import cv2
import numpy as np

from contract.contracts import FramePacket, Prediction, SignState, StateUpdate

# ── layout constants ──────────────────────────────────────────────────────────
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PAD = 10          # generic padding (px)
_BAR_W = 160       # max width of a confidence bar (px)
_BAR_H = 14        # height of a confidence bar (px)
_RIGHT_PANEL_W = 220

# colours (BGR)
_C_GREEN = (0, 220, 0)
_C_RED = (0, 0, 220)
_C_WHITE = (255, 255, 255)
_C_BLACK = (0, 0, 0)
_C_YELLOW = (0, 220, 220)
_C_CYAN = (220, 220, 0)
_C_ORANGE = (0, 165, 255)
_C_OVERLAY_BG = (20, 20, 20)

_STATE_COLOR = {
    SignState.IDLE: _C_CYAN,
    SignState.ACTIVE: _C_GREEN,
}


def _put_text(
    img: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    color: tuple = _C_WHITE,
    thickness: int = 1,
    shadow: bool = True,
) -> None:
    x, y = origin
    if shadow:
        cv2.putText(img, text, (x + 1, y + 1), _FONT, scale, _C_BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), _FONT, scale, color, thickness, cv2.LINE_AA)


def _confidence_bar(
    img: np.ndarray,
    origin: tuple[int, int],
    prob: float,
    rank: int,
) -> None:
    """Draw a filled rectangle representing `prob` as a fraction of _BAR_W."""
    x, y = origin
    filled = max(1, int(prob * _BAR_W))
    bar_colors = [_C_GREEN, _C_YELLOW, _C_ORANGE]
    color = bar_colors[rank % len(bar_colors)]
    # background track
    cv2.rectangle(img, (x, y), (x + _BAR_W, y + _BAR_H), (60, 60, 60), -1)
    # filled portion
    cv2.rectangle(img, (x, y), (x + filled, y + _BAR_H), color, -1)
    # border
    cv2.rectangle(img, (x, y), (x + _BAR_W, y + _BAR_H), (120, 120, 120), 1)


class OverlayRenderer:
    """Draws all HUD elements onto a copy of the webcam frame."""

    def draw(
        self,
        frame: np.ndarray,
        frame_packet: FramePacket,
        state_update: StateUpdate,
        latest_prediction: Optional[Prediction],
        fps: float = 0.0,
    ) -> np.ndarray:
        """Return an annotated copy of `frame`. Never mutates the original."""
        out = frame.copy()
        h, w = out.shape[:2]

        self._draw_bbox(out, frame_packet)
        self._draw_state(out, state_update)
        self._draw_fps(out, fps, w)
        self._draw_frame_id(out, frame_packet.frame_id, w, h)
        if latest_prediction is not None:
            self._draw_predictions(out, latest_prediction, h)
        return out

    # ── private helpers ───────────────────────────────────────────────────────

    def _draw_bbox(self, img: np.ndarray, packet: FramePacket) -> None:
        if packet.bbox is None:
            return
        x1, y1, x2, y2 = (int(v) for v in packet.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), _C_GREEN, 2)

    def _draw_state(self, img: np.ndarray, su: StateUpdate) -> None:
        color = _STATE_COLOR.get(su.state, _C_WHITE)
        label = su.state.value.upper()
        if su.state == SignState.ACTIVE:
            label += f"  [{su.active_frame_count} fr]"
        _put_text(img, label, (_PAD, _PAD + 18), scale=0.65, color=color, thickness=2)

    def _draw_fps(self, img: np.ndarray, fps: float, width: int) -> None:
        text = f"FPS: {fps:.1f}"
        (tw, _), _ = cv2.getTextSize(text, _FONT, 0.55, 1)
        _put_text(img, text, (width - tw - _PAD, _PAD + 18), scale=0.55, color=_C_WHITE)

    def _draw_frame_id(self, img: np.ndarray, frame_id: int, width: int, height: int) -> None:
        text = f"frame {frame_id}"
        (tw, _), _ = cv2.getTextSize(text, _FONT, 0.4, 1)
        _put_text(img, text, (width - tw - _PAD, height - _PAD), scale=0.4, color=(140, 140, 140))

    def _draw_predictions(self, img: np.ndarray, pred: Prediction, height: int) -> None:
        """Render top-3 predictions with confidence bars on the right side."""
        panel_x = img.shape[1] - _RIGHT_PANEL_W - _PAD
        y = height // 2 - 60

        # semi-transparent background panel
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (panel_x - _PAD, y - 24),
            (img.shape[1] - _PAD // 2, y + 3 * 54 + _PAD),
            _C_OVERLAY_BG,
            -1,
        )
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        _put_text(img, "TOP-3 PREDICTIONS", (panel_x, y - 6), scale=0.45, color=_C_CYAN)

        for rank, (gloss, prob) in enumerate(zip(pred.top_k_glosses, pred.top_k_probs)):
            row_y = y + rank * 54 + 20
            # gloss label
            _put_text(img, f"{rank + 1}. {gloss}", (panel_x, row_y), scale=0.52, color=_C_WHITE)
            # probability text
            prob_text = f"{prob * 100:.1f}%"
            _put_text(img, prob_text, (panel_x + _BAR_W + 6, row_y + _BAR_H), scale=0.45, color=_C_WHITE)
            # bar
            _confidence_bar(img, (panel_x, row_y + 6), prob, rank)
