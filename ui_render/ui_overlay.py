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
FPS = 29.97
FRAME_COUNT = 64
TOTAL_SIGN_S = FRAME_COUNT/FPS

# colours (BGR)
_C_GREEN = (0, 220, 0)
_C_RED = (0, 0, 220)
_C_BLUE = (220, 100, 0)    # left hand
_C_RHAND = (0, 80, 220)    # right hand
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


def _bbox_from_points(pts: np.ndarray, frame_w: int, frame_h: int,
                      pad: float = 0.05) -> tuple | None:
    nonzero = pts[~np.all(pts == 0, axis=1)]
    if len(nonzero) == 0:
        return None
    x_min, y_min = nonzero.min(axis=0)
    x_max, y_max = nonzero.max(axis=0)
    if x_max - x_min < 1e-4 or y_max - y_min < 1e-4:
        return None
    px = (x_max - x_min) * pad
    py = (y_max - y_min) * pad
    return (
        int(max(0,       (x_min - px) * frame_w)),
        int(max(0,       (y_min - py) * frame_h)),
        int(min(frame_w, (x_max + px) * frame_w)),
        int(min(frame_h, (y_max + py) * frame_h)),
    )


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
        if frame_packet.landmarks_raw is not None:
            self._draw_landmarks(out, frame_packet.landmarks_raw)
        self._draw_state(out, state_update)
        self._draw_fps(out, fps, w)
        if latest_prediction is not None:
            self._draw_predictions(out, latest_prediction, state_update, h)
        return out

    # ── private helpers ───────────────────────────────────────────────────────

    def _draw_landmarks(self, img: np.ndarray, landmarks_raw: np.ndarray) -> None:
        h, w = img.shape[:2]
        regions = [
            (landmarks_raw[0:23],  _C_WHITE,        4),  # body
            (landmarks_raw[23:44], (220, 100,   0), 3),  # left hand
            (landmarks_raw[44:65], (  0,  80, 220), 3),  # right hand
        ]
        for pts, color, r in regions:
            for x, y in pts:
                if x == 0.0 and y == 0.0:
                    continue
                cv2.circle(img, (int(x * w), int(y * h)), r, color, -1)

    def _draw_bbox(self, img: np.ndarray, packet: FramePacket) -> None:
        h, w = img.shape[:2]
        if packet.bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in packet.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), _C_GREEN, 2)
            cv2.putText(img, "body", (x1, y1 - 6), _FONT, 0.45, _C_GREEN, 1, cv2.LINE_AA)
        if packet.landmarks_raw is not None:
            lbox = _bbox_from_points(packet.landmarks_raw[23:44], w, h)
            if lbox is not None:
                x1, y1, x2, y2 = lbox
                cv2.rectangle(img, (x1, y1), (x2, y2), _C_BLUE, 2)
                cv2.putText(img, "left hand", (x1, y1 - 6), _FONT, 0.45, _C_BLUE, 1, cv2.LINE_AA)
            rbox = _bbox_from_points(packet.landmarks_raw[44:65], w, h)
            if rbox is not None:
                x1, y1, x2, y2 = rbox
                cv2.rectangle(img, (x1, y1), (x2, y2), _C_RHAND, 2)
                cv2.putText(img, "right hand", (x1, y1 - 6), _FONT, 0.45, _C_RHAND, 1, cv2.LINE_AA)

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

    # def _draw_frame_id(self, img: np.ndarray, frame_id: int, width: int, height: int) -> None:
    #     text = f"frame {frame_id}"
    #     (tw, _), _ = cv2.getTextSize(text, _FONT, 0.4, 1)
    #     _put_text(img, text, (width - tw - _PAD, height - _PAD), scale=0.4, color=(140, 140, 140))

    def _draw_predictions(self, img: np.ndarray, pred: Prediction, state_update : StateUpdate, height: int) -> None:
        """Minimal bottom-right panel: three rows, rank-dimmed, no bars."""
        w = img.shape[1]
        n = min(len(pred.top_k_glosses), 3)
        footer_h = 26
        panel_h = n * _PRED_ROW_H + _PAD + footer_h
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
        footer_y = y0 + n * _PRED_ROW_H + 18
 
        RTF = (pred.inference_end_ts - pred.inference_start_ts)/TOTAL_SIGN_S
        footer_text = f"RTF: {RTF}"

        _put_text(
            img,
            footer_text,
            (x0, footer_y),
            scale=0.45,
            color=_C_YELLOW,
            thickness=1,
        )
