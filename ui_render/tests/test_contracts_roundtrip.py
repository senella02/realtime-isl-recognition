"""Ensure each contract dataclass can be instantiated and round-trips its fields."""
import time
import numpy as np
import pytest

from contract.contracts import (
    FramePacket, StateUpdate, SignEvent, Prediction, SignState
)


def test_frame_packet_fields():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    ts = time.perf_counter()
    p = FramePacket(
        frame_id=1, capture_ts=ts, image_bgr=img,
        bbox=(10, 20, 300, 400), landmarks_raw=None, landmarks_normalized=None,
    )
    assert p.frame_id == 1
    assert p.bbox == (10, 20, 300, 400)
    assert p.landmarks_raw is None


def test_state_update_idle():
    su = StateUpdate(
        frame_id=5, state=SignState.IDLE, active_frame_count=0,
        sign_id=None, triggered=False, sign_event=None,
    )
    assert su.state == SignState.IDLE
    assert not su.triggered


def test_state_update_triggered():
    now = time.perf_counter()
    event = SignEvent(sign_id=3, sign_start_ts=now - 1.0,
                      sign_end_ts=now, sign_duration_s=1.0, buffer_length=40)
    su = StateUpdate(
        frame_id=99, state=SignState.IDLE, active_frame_count=0,
        sign_id=3, triggered=True, sign_event=event,
    )
    assert su.triggered
    assert su.sign_event is not None
    assert su.sign_event.sign_id == su.sign_id


def test_prediction_fields():
    probs = np.ones(184, dtype=np.float32) / 184
    now = time.perf_counter()
    pred = Prediction(
        sign_id=3,
        inference_start_ts=now - 0.08,
        inference_end_ts=now,
        probs=probs,
        top_k_indices=[0, 1, 2],
        top_k_probs=[probs[0], probs[1], probs[2]],
        top_k_glosses=["A", "B", "C"],
    )
    assert len(pred.probs) == 184
    assert len(pred.top_k_glosses) == 3
