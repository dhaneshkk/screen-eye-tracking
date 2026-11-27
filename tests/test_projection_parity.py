"""Regression tests for the projection fixes and the shared Python/TypeScript fixture."""

import json
import math
import unittest

from screen_eye_tracking.backend import Detection, DisplayGeometry, GazeEstimate, ScreenProjector

from gen_projection_fixtures import FIXTURE_PATH, build_cases


def eyes_at(projector: ScreenProjector, distance: float, mid_x: float = 320.0, mid_y: float = 240.0) -> list[Detection]:
    half = 0.0315 * projector.focal_px / distance
    return [
        Detection(17, 1.0, mid_x - half - 4, mid_y - 4, mid_x - half + 4, mid_y + 4),
        Detection(17, 1.0, mid_x + half - 4, mid_y - 4, mid_x + half + 4, mid_y + 4),
    ]


class CameraOffsetFlipTest(unittest.TestCase):
    display = DisplayGeometry(1920, 1080, 31.5)

    def test_looking_into_offcenter_camera_hits_camera_position(self) -> None:
        for cam_x in (0.0, 0.2, 0.5, 0.85, 1.0):
            for flip_x in (True, False):
                with self.subTest(cam_x=cam_x, flip_x=flip_x):
                    projector = ScreenProjector(self.display, flip_x=flip_x, camera_screen_x=cam_x)
                    x, _ = projector.project((320.0, 240.0), 0.0, 0.0, 0.6)
                    self.assertAlmostEqual(x, cam_x, places=9)

    def test_flip_mirrors_offsets_around_camera(self) -> None:
        for cam_x in (0.2, 0.5, 0.7):
            flipped = ScreenProjector(self.display, flip_x=True, camera_screen_x=cam_x)
            plain = ScreenProjector(self.display, flip_x=False, camera_screen_x=cam_x)
            xf, _ = flipped.project((320.0, 240.0), 4.0, 0.0, 0.6)
            xp, _ = plain.project((320.0, 240.0), 4.0, 0.0, 0.6)
            self.assertAlmostEqual(xf - cam_x, -(xp - cam_x), places=9)

    def test_centered_camera_is_unchanged_from_upstream(self) -> None:
        projector = ScreenProjector(self.display, flip_x=True, camera_screen_x=0.5)
        for yaw in (-15.0, -3.0, 0.0, 7.0, 25.0):
            hit = projector._screen_hit_m((300.0, 240.0), yaw, 0.0, 0.6)
            upstream = 1.0 - min(1.0, max(0.0, hit[0] / self.display.size_m[0]))
            self.assertAlmostEqual(projector.project((300.0, 240.0), yaw, 0.0, 0.6)[0], upstream, places=12)


class ConvergenceScreenPlaneTest(unittest.TestCase):
    display = DisplayGeometry(1920, 1080, 31.5)

    def test_vergence_depth_error_does_not_move_screen_hit(self) -> None:
        projector = ScreenProjector(self.display, flip_x=False, flip_y=False)
        distance = 0.6
        eyes = eyes_at(projector, distance)
        width_m = self.display.size_m[0]
        origin_x = width_m * 0.5
        target_x = origin_x + 0.25
        yaw_l = math.degrees(math.atan((target_x - (origin_x - 0.0315)) / distance))
        yaw_r = math.degrees(math.atan((target_x - (origin_x + 0.0315)) / distance))
        for vergence_error in (-1.5, -0.75, 0.0, 0.75, 1.5):
            with self.subTest(vergence_error=vergence_error):
                estimate = GazeEstimate(0.0, 0.0, yaw_l + vergence_error, 0.0, yaw_r - vergence_error, 0.0)
                result = projector.project_estimate("binocular-convergence", eyes, estimate, distance)
                self.assertIsNone(result.fallback_reason)
                # Upstream returned 0.727 / 1.0 (clamped) for +-1.5 deg; the true hit is 0.8585.
                self.assertAlmostEqual(result.point[0], target_x / width_m, delta=0.005)

    def test_exact_rays_agree_with_binocular_screen(self) -> None:
        projector = ScreenProjector(self.display, flip_x=True, flip_y=True, camera_screen_x=0.3)
        distance = 0.7
        eyes = eyes_at(projector, distance, 300.0, 230.0)
        left = projector._eye_origin_m(eyes[0].center, distance)
        right = projector._eye_origin_m(eyes[1].center, distance)
        target = ((left[0] + right[0]) * 0.5 + 0.12, (left[1] + right[1]) * 0.5 + 0.05)

        def angles(origin):
            yaw = math.degrees(math.atan((target[0] - origin[0]) / distance))
            pitch = -math.degrees(math.atan((target[1] - origin[1]) / distance))  # flip_y
            return yaw, pitch

        (ly, lp), (ry, rp) = angles(left), angles(right)
        estimate = GazeEstimate(0.0, 0.0, ly, lp, ry, rp)
        conv = projector.project_estimate("binocular-convergence", eyes, estimate, distance)
        screen = projector.project_estimate("binocular-screen", eyes, estimate, distance)
        self.assertIsNone(conv.fallback_reason)
        self.assertAlmostEqual(conv.point[0], screen.point[0], places=6)
        self.assertAlmostEqual(conv.point[1], screen.point[1], places=6)


class SharedFixtureTest(unittest.TestCase):
    def test_fixture_is_up_to_date(self) -> None:
        stored = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]
        regenerated = build_cases()
        self.assertEqual(len(stored), len(regenerated))
        for index, (old, new) in enumerate(zip(stored, regenerated)):
            self.assertEqual(old["expected_fallback"], new["expected_fallback"], index)
            for a, b in zip(old["expected_point"], new["expected_point"]):
                self.assertAlmostEqual(a, b, places=12, msg=f"case {index}; rerun gen_projection_fixtures.py")


if __name__ == "__main__":
    unittest.main()
