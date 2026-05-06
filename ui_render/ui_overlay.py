"""Phase A — UI overlay drawn on each webcam frame via OpenCV."""

from typing import Optional

import cv2
import numpy as np

from contract.contracts import FramePacket, Prediction, SignState, StateUpdate

# ── layout constants ──────────────────────────────────────────────────────────
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PAD = 10          # generic padding (px)
_PRED_PANEL_W = 240  # prediction panel width (px)
_PRED_ROW_H   = 28   # height per prediction row (px)

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
        """Minimal bottom-right panel: three rows, rank-dimmed, no bars."""
        w = img.shape[1]
        n = min(len(pred.top_k_glosses), 3)

        panel_h = n * _PRED_ROW_H + _PAD
        x0 = w - _PRED_PANEL_W - _PAD
        y0 = height - panel_h - _PAD

        # semi-transparent dark background
        overlay = img.copy()
        cv2.rectangle(overlay,
                      (x0 - _PAD, y0 - _PAD // 2),
                      (w - _PAD // 2, height - _PAD // 2),
                      _C_OVERLAY_BG, -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

        # thin green accent line on the left edge — marks rank 1
        cv2.line(img,
                 (x0 - _PAD + 2, y0 + 2),
                 (x0 - _PAD + 2, y0 + _PRED_ROW_H - 2),
                 _C_GREEN, 2)

        row_colors = [_C_WHITE, (175, 175, 175), (110, 110, 110)]

        for rank, (gloss, prob) in enumerate(
                zip(pred.top_k_glosses[:n], pred.top_k_probs[:n])):
            row_y = y0 + rank * _PRED_ROW_H + _PRED_ROW_H // 2 + 4
            color  = row_colors[rank]
            scale  = 0.52 if rank == 0 else 0.46
            thick  = 2    if rank == 0 else 1

            _put_text(img, f"{rank + 1}.  {gloss}",
                      (x0, row_y), scale=scale, color=color, thickness=thick)

            prob_text = f"{prob * 100:.1f}%"
            (tw, _), _ = cv2.getTextSize(prob_text, _FONT, 0.45, 1)
            _put_text(img, prob_text,
                      (w - _PAD * 2 - tw, row_y),
                      scale=0.45,
                      color=_C_CYAN if rank == 0 else color)
