"""The vectorized YOLO decode must reproduce the original per-anchor Python loop exactly."""

import math
import time
import unittest
from typing import Iterable

import numpy as np

from screen_eye_tracking.backend import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    EYE_CLASS_ID,
    HEAD_CLASS_ID,
    MOUTH_CLASS_ID,
    YOLO_CLASS_COUNT,
    YOLO_EYE_SCORE_THRESHOLD,
    YOLO_INPUT_HEIGHT,
    YOLO_INPUT_WIDTH,
    YOLO_NMS_IOU_THRESHOLD,
    Detection,
    DetectionResult,
    detection_iou,
    parse_yolo_output,
    select_best_in_head,
    select_eye_pair,
)


def _reference_nms(detections: Iterable[Detection], iou_threshold: float) -> list[Detection]:
    remaining = sorted(detections, key=lambda det: det.score, reverse=True)
    selected: list[Detection] = []
    while remaining:
        current = remaining.pop(0)
        selected.append(current)
        remaining = [det for det in remaining if detection_iou(current, det) <= iou_threshold]
    return selected


def reference_parse_yolo_output(output, score_threshold, image_w=CAMERA_WIDTH, image_h=CAMERA_HEIGHT):
    """Verbatim copy of the pre-vectorization implementation (upstream 48b7e37)."""
    values = np.asarray(output, dtype=np.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    scale_x = float(image_w) / float(YOLO_INPUT_WIDTH)
    scale_y = float(image_h) / float(YOLO_INPUT_HEIGHT)
    heads, eyes, mouths = [], [], []
    for index in range(values.shape[1]):
        cx, cy, box_w, box_h = (float(values[k, index]) for k in range(4))
        if not all(math.isfinite(v) for v in (cx, cy, box_w, box_h)) or box_w <= 0.0 or box_h <= 0.0:
            continue
        x1 = max(0.0, min(float(image_w - 1), (cx - box_w * 0.5) * scale_x))
        y1 = max(0.0, min(float(image_h - 1), (cy - box_h * 0.5) * scale_y))
        x2 = max(0.0, min(float(image_w - 1), (cx + box_w * 0.5) * scale_x))
        y2 = max(0.0, min(float(image_h - 1), (cy + box_h * 0.5) * scale_y))
        if x2 <= x1 or y2 <= y1:
            continue
        head_score = float(values[4 + HEAD_CLASS_ID, index])
        if math.isfinite(head_score) and head_score >= score_threshold:
            heads.append(Detection(HEAD_CLASS_ID, head_score, x1, y1, x2, y2))
        eye_score = float(values[4 + EYE_CLASS_ID, index])
        if math.isfinite(eye_score) and eye_score >= YOLO_EYE_SCORE_THRESHOLD:
            eyes.append(Detection(EYE_CLASS_ID, eye_score, x1, y1, x2, y2))
        mouth_score = float(values[4 + MOUTH_CLASS_ID, index])
        if math.isfinite(mouth_score) and mouth_score >= score_threshold:
            mouths.append(Detection(MOUTH_CLASS_ID, mouth_score, x1, y1, x2, y2))
    heads = _reference_nms(heads, YOLO_NMS_IOU_THRESHOLD)
    eyes = _reference_nms(eyes, YOLO_NMS_IOU_THRESHOLD)
    mouths = _reference_nms(mouths, YOLO_NMS_IOU_THRESHOLD)
    if not heads:
        return DetectionResult(None, [])
    head = heads[0]
    return DetectionResult(head, select_eye_pair(head, eyes), select_best_in_head(head, mouths))


def random_output(seed: int, score_scale: float, anchors: int = 6300, plant_face: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = rng.random((1, 4 + YOLO_CLASS_COUNT, anchors), dtype=np.float32)
    out[0, 0] *= YOLO_INPUT_WIDTH
    out[0, 1] *= YOLO_INPUT_HEIGHT
    out[0, 2:4] = out[0, 2:4] * 80.0 - 5.0  # some non-positive sizes
    out[0, 4:] *= score_scale
    out[0, 0, :7] = [np.nan, np.inf, 100.0, 100.0, 100.0, 100.0, 100.0]  # non-finite anchors
    out[0, 4 + EYE_CLASS_ID, 10:14] = 0.9  # exact score ties
    if plant_face:
        a = 20
        out[0, :4, a] = [320, 240, 160, 200]
        out[0, 4 + HEAD_CLASS_ID, a] = 0.95
        for k, (x, y) in enumerate([(290, 220), (350, 220)]):
            out[0, :4, a + 1 + k] = [x, y, 24, 14]
            out[0, 4 + EYE_CLASS_ID, a + 1 + k] = 0.85
        out[0, :4, a + 3] = [320, 300, 50, 24]
        out[0, 4 + MOUTH_CLASS_ID, a + 3] = 0.8
    return out


class VectorizedYoloTest(unittest.TestCase):
    def assert_same(self, out: np.ndarray, threshold: float = 0.5, image_w=CAMERA_WIDTH, image_h=CAMERA_HEIGHT) -> None:
        self.assertEqual(parse_yolo_output(out, threshold, image_w, image_h), reference_parse_yolo_output(out, threshold, image_w, image_h))

    def test_matches_reference_on_realistic_outputs(self) -> None:
        for seed in range(40):
            with self.subTest(seed=seed):
                self.assert_same(random_output(seed, score_scale=0.05))

    def test_matches_reference_on_dense_outputs(self) -> None:
        # Fewer anchors keep the O(n^2) reference fast; every class still has hundreds of candidates.
        for seed in range(6):
            with self.subTest(seed=seed):
                self.assert_same(random_output(seed, score_scale=1.0, anchors=900), threshold=0.4)

    def test_matches_reference_when_no_head(self) -> None:
        self.assert_same(random_output(1, score_scale=0.05, plant_face=False))

    def test_matches_reference_with_scaled_camera(self) -> None:
        self.assert_same(random_output(2, score_scale=0.05), image_w=1920, image_h=1080)

    def test_realistic_frame_is_fast(self) -> None:
        out = random_output(0, score_scale=0.05)
        start = time.perf_counter()
        for _ in range(20):
            parse_yolo_output(out, 0.5)
        per_frame_ms = (time.perf_counter() - start) / 20 * 1000.0
        self.assertLess(per_frame_ms, 5.0)

    def test_worst_case_frame_is_bounded(self) -> None:
        out = random_output(0, score_scale=1.0)  # ~every anchor passes every threshold
        start = time.perf_counter()
        parse_yolo_output(out, 0.0)
        self.assertLess((time.perf_counter() - start) * 1000.0, 250.0)


if __name__ == "__main__":
    unittest.main()
