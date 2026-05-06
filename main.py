import cv2

# ── M4: load model once before loop ──────────────────────────────────────────
# model = m4.load()
last_results = None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ── M1: extract landmarks ─────────────────────────────────────────────────
    # landmarks, bbox = m1.extract(frame) (1, 150)

    # ── M3: update state machine, get buffer if sign detected ─────────────────
    # buffer, state = m3.update(landmarks) 

    # ── M4: run inference if buffer is ready ──────────────────────────────────
    # if buffer is not None:
    #     last_results = m4.infer(buffer) 

    # ── M2: render overlay on frame ───────────────────────────────────────────
    # frame = m2.render(frame, bbox, state, last_results)

    cv2.imshow("ISL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("ISL Recognition", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()