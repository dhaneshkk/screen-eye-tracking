"""Generate tests/fixtures/projection_cases.json from the Python ScreenProjector.

Both tests/test_projection_parity.py (Python) and web/src/inference/parity.test.ts (TypeScript)
assert against this file, so the two projector implementations cannot silently drift apart.

    PYTHONPATH=src python tests/gen_projection_fixtures.py
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

from screen_eye_tracking.backend import Detection, DisplayGeometry, GazeEstimate, ScreenProjector

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "projection_cases.json"
MODES = ["legacy", "binocular-screen", "binocular-convergence"]


def eye(cx: float, cy: float) -> Detection:
    return Detection(17, 0.9, cx - 6.0, cy - 4.0, cx + 6.0, cy + 4.0)


def build_cases() -> list[dict]:
    rng = np.random.default_rng(20261127)
    cases: list[dict] = []
    configs = itertools.product(
        [True, False],  # flip_x
        [True, False],  # flip_y
        [0.0, 0.2, 0.5, 0.85],  # camera_screen_x
        [0.0, 0.5],  # camera_screen_y
        [60.0, 90.0],  # camera_fov_deg
    )
    for flip_x, flip_y, cam_x, cam_y, fov in configs:
        cfg = {
            "display": [1920, 1080, 31.5],
            "flip_x": flip_x,
            "flip_y": flip_y,
            "camera_screen_x": cam_x,
            "camera_screen_y": cam_y,
            "eye_position_weight_x": 1.0,
            "eye_position_weight_y": 0.25,
            "camera_fov_deg": fov,
            "camera_width": 640,
            "camera_height": 480,
        }
        projector = ScreenProjector(DisplayGeometry(*cfg["display"]), **{k: v for k, v in cfg.items() if k != "display"})
        for sample in range(3):
            distance = float(rng.uniform(0.45, 0.9))
            mid_x, mid_y = float(rng.uniform(220, 420)), float(rng.uniform(180, 300))
            half_ipd_px = 0.0315 * projector.focal_px / distance
            eyes = [eye(mid_x - half_ipd_px, mid_y), eye(mid_x + half_ipd_px, mid_y)]
            yaw, pitch = float(rng.uniform(-20, 20)), float(rng.uniform(-15, 15))
            vergence = math.degrees(math.atan(0.0315 / distance))
            noise = rng.normal(0.0, 0.7, size=4)
            if sample == 2:  # parallel rays -> binocular-convergence must fall back to legacy
                vergence, noise = 0.0, noise * 0.0
            estimate = GazeEstimate(
                yaw_deg=yaw,
                pitch_deg=pitch,
                left_yaw_deg=yaw + vergence + float(noise[0]),
                left_pitch_deg=pitch + float(noise[1]),
                right_yaw_deg=yaw - vergence + float(noise[2]),
                right_pitch_deg=pitch + float(noise[3]),
            )
            for mode in MODES:
                result = projector.project_estimate(mode, eyes, estimate, distance)
                cases.append(
                    {
                        "config": cfg,
                        "mode": mode,
                        "distance_m": distance,
                        "eyes": [[e.class_id, e.score, e.x1, e.y1, e.x2, e.y2] for e in eyes],
                        "estimate": [
                            estimate.yaw_deg,
                            estimate.pitch_deg,
                            estimate.left_yaw_deg,
                            estimate.left_pitch_deg,
                            estimate.right_yaw_deg,
                            estimate.right_pitch_deg,
                        ],
                        "expected_point": list(result.point),
                        "expected_fallback": result.fallback_reason is not None,
                    }
                )
    return cases


def main() -> None:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    FIXTURE_PATH.write_text(json.dumps({"version": 1, "cases": cases}, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
