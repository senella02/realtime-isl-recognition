import time
import cv2
import numpy as np
from extractor.mediapipe_pipeline import LandmarkExtractor

# colours (BGR)
_GREEN = (0, 220, 0)
_BLUE  = (220, 100, 0)
_RED   = (0, 80, 220)
_WHITE = (255, 255, 255)
_FONT  = cv2.FONT_HERSHEY_SIMPLEX


def _bbox_from_points(pts: np.ndarray, frame_w: int, frame_h: int,
                      pad: float = 0.05) -> tuple | None:
    """Compute bbox from a set of (N, 2) landmark points. Skips zero rows."""
    nonzero = pts[~np.all(pts == 0, axis=1)]
    if len(nonzero) == 0:
        return None
    x_min, y_min = nonzero.min(axis=0)
    x_max, y_max = nonzero.max(axis=0)
    if x_max - x_min < 1e-4 or y_max - y_min < 1e-4:   # single-point degenerate
        return None
    px = (x_max - x_min) * pad
    py = (y_max - y_min) * pad
    return (
        int(max(0,       (x_min - px) * frame_w)),
        int(max(0,       (y_min - py) * frame_h)),
        int(min(frame_w, (x_max + px) * frame_w)),
        int(min(frame_h, (y_max + py) * frame_h)),
    )


def _draw_bbox(img, box, color, label):
    if box is None:
        return
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 6), _FONT, 0.45, color, 1, cv2.LINE_AA)


def _draw_landmarks(img, raw: np.ndarray, frame_w: int, frame_h: int):
    """Draw all 75 landmarks as colored dots."""
    regions = [
        (raw[0:23],  _WHITE, 3),   # body  — white, larger
        (raw[23:44], _BLUE,  2),   # left  — blue
        (raw[44:65], _RED,   2),   # right — red
    ]
    for pts, color, r in regions:
        for x, y in pts:
            if x == 0.0 and y == 0.0:
                continue
            cx, cy = int(x * frame_w), int(y * frame_h)
            cv2.circle(img, (cx, cy), r, color, -1)


def _detection_counts(raw: np.ndarray) -> tuple[int, int, int]:
    """Count non-zero (detected) landmarks per region."""
    body  = int(np.any(raw[0:23]  != 0, axis=1).sum())
    lhand = int(np.any(raw[23:44] != 0, axis=1).sum())
    rhand = int(np.any(raw[44:65] != 0, axis=1).sum())
    return body, lhand, rhand


# ── main ──────────────────────────────────────────────────────────────────────

ext = LandmarkExtractor()
print("Starting — press 'q' to quit")

frame_times = []

while True:
    frame, ts = ext.grab()
    if frame is None:
        break

    t0 = time.perf_counter()
    pkt = ext.extract(frame, ts)
    extract_ms = (time.perf_counter() - t0) * 1000

    frame_times.append(extract_ms)
    fps = 1000 / (sum(frame_times[-30:]) / min(len(frame_times), 30))

    display = pkt.image_bgr.copy()
    h, w = display.shape[:2]

    if pkt.landmarks_raw is not None:
        raw = pkt.landmarks_raw                     # (65, 2)
        body_count, lhand_count, rhand_count = _detection_counts(raw)

        # draw landmark dots
        _draw_landmarks(display, raw, w, h)

        # 3 separate bboxes
        _draw_bbox(display, pkt.bbox,                              _GREEN, "body")
        _draw_bbox(display, _bbox_from_points(raw[23:44], w, h),  _BLUE,  "left hand")
        _draw_bbox(display, _bbox_from_points(raw[44:65], w, h),  _RED,   "right hand")

        status = (f"body:{body_count}/23  "
                  f"L-hand:{lhand_count}/21  "
                  f"R-hand:{rhand_count}/21  "
                  f"total:{body_count+lhand_count+rhand_count}/65")
    else:
        status = "no body detected"

    # HUD
    cv2.putText(display, f"FPS:{fps:.1f}  extract:{extract_ms:.1f}ms",
                (10, 25), _FONT, 0.6, _WHITE, 1, cv2.LINE_AA)
    cv2.putText(display, status,
                (10, 50), _FONT, 0.55, _WHITE, 1, cv2.LINE_AA)
    cv2.putText(display, f"frame_id:{pkt.frame_id}",
                (10, 75), _FONT, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    cv2.imshow("M1 Test", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("M1 Test", cv2.WND_PROP_VISIBLE) < 1:
        break

ext.release()
cv2.destroyAllWindows()

if frame_times:
    avg_ms = sum(frame_times) / len(frame_times)
    print(f"\nAvg extract: {avg_ms:.1f} ms  |  Avg FPS: {1000/avg_ms:.1f}")
    print("AC-4 (>=20 FPS):", "PASS" if 1000 / avg_ms >= 20 else "FAIL")
