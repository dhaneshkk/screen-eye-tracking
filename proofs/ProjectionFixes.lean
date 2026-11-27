/-
  Formal statements for the two ScreenProjector geometry fixes.

  STATUS: written against Mathlib but NOT yet machine-checked (no Lean toolchain was available
  in the environment that produced this file). Check with any Mathlib project:
      lake env lean proofs/ProjectionFixes.lean

  Notation (screen coordinates normalised to [0, 1] on x; metres on the ray geometry):
    c : camera_screen_x,  x : unflipped normalised hit
    eyes on the plane z = 0, screen on the plane z = D (D = distance_m > 0)
-/
import Mathlib

namespace ScreenEyeTracking

/-! ## Fix 1 — mirror around the camera, not the screen centre -/

/-- The new flip: `x ↦ 2c − x` (backend.py `_normalize_hit_m`, core.ts `normalizeHitM`). -/
def flipX (c x : ℝ) : ℝ := 2 * c - x

/-- Looking straight into the lens lands on the lens, for any camera position. -/
theorem flipX_fixes_camera (c : ℝ) : flipX c c = c := by
  unfold flipX; ring

/-- The flip mirrors the camera-relative offset. -/
theorem flipX_mirrors_offset (c x : ℝ) : flipX c x - c = -(x - c) := by
  unfold flipX; ring

/-- Flipping twice is the identity. -/
theorem flipX_involutive (c x : ℝ) : flipX c (flipX c x) = x := by
  unfold flipX; ring

/-- At the default `camera_screen_x = 0.5` the new flip is the upstream `1 − x`. -/
theorem flipX_centered (x : ℝ) : flipX (1 / 2) x = 1 - x := by
  unfold flipX; ring

def clamp01 (x : ℝ) : ℝ := max 0 (min 1 x)

/-- Upstream clamped then flipped; the fix flips then clamps. At the centre these agree,
    so behaviour for `camera_screen_x = 0.5` is bit-for-bit unchanged (up to float rounding). -/
theorem clamp_flip_commute_at_center (x : ℝ) : clamp01 (1 - x) = 1 - clamp01 x := by
  unfold clamp01
  by_cases h0 : x ≤ 0
  · rw [min_eq_right (show x ≤ 1 by linarith), max_eq_left h0,
      min_eq_left (show (1 : ℝ) ≤ 1 - x by linarith),
      max_eq_right (show (0 : ℝ) ≤ 1 by norm_num)]
    norm_num
  · by_cases h1 : 1 ≤ x
    · rw [min_eq_left h1, max_eq_right (show (0 : ℝ) ≤ 1 by norm_num),
        min_eq_right (show 1 - x ≤ 1 by linarith),
        max_eq_left (show 1 - x ≤ 0 by linarith)]
      norm_num
    · rw [min_eq_right (show x ≤ 1 by linarith), max_eq_right (show 0 ≤ x by linarith),
        min_eq_right (show 1 - x ≤ 1 by linarith),
        max_eq_right (show (0 : ℝ) ≤ 1 - x by linarith)]

/-! ## Fix 2 — binocular-convergence hits the screen plane

  New hit: `h = c + (m − c) · D / (m_z − c_z)` with `c` the cyclopean eye (`c_z = 0`) and `m`
  the closest-approach midpoint of the two gaze rays. Shown per coordinate. -/

/-- The x (or y) coordinate of the new hit point. -/
noncomputable def hitCoord (cyc m mz D : ℝ) : ℝ := cyc + (m - cyc) * (D / mz)

/-- The new hit lies on the screen plane `z = D` (z-coordinate of the same formula, `c_z = 0`). -/
theorem hit_on_screen_plane (mz D : ℝ) (hz : mz ≠ 0) : hitCoord 0 mz mz D = D := by
  unfold hitCoord
  rw [zero_add, sub_zero, mul_div_assoc', mul_comm mz D, mul_div_assoc, div_self hz, mul_one]

/-- Vergence-depth error along the cyclopean ray does not move the hit: replacing `m − c`
    by `k · (m − c)` for any `k ≠ 0` gives the same point. Upstream returned `m` itself,
    which moves with `k`. -/
theorem hit_invariant_along_cyclopean_ray (cyc m mz D k : ℝ) (hk : k ≠ 0) :
    cyc + (k * (m - cyc)) * (D / (k * mz)) = hitCoord cyc m mz D := by
  unfold hitCoord
  calc cyc + (k * (m - cyc)) * (D / (k * mz))
      = cyc + (m - cyc) * (k * D / (k * mz)) := by ring
    _ = cyc + (m - cyc) * (D / mz) := by rw [mul_div_mul_left D mz hk]

/-- Exactness: if the rays meet on the screen (`m_z = D`), the new hit is that meeting point. -/
theorem hit_exact_when_rays_meet_on_screen (cyc s D : ℝ) (hD : D ≠ 0) :
    hitCoord cyc s D D = s := by
  unfold hitCoord
  rw [div_self hD, mul_one]
  ring

end ScreenEyeTracking
