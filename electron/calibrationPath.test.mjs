import { createRequire } from "node:module";
import path from "node:path";
import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const { resolveAllowedCalibrationPath } = require("./calibrationPath.cjs");

const root = path.resolve("/repo");
const configured = path.join(root, ".gaze_calibration.json");

describe("calibration IPC path guard", () => {
  it("accepts the configured file, however it is spelled", () => {
    expect(resolveAllowedCalibrationPath(configured, configured, root)).toBe(configured);
    expect(resolveAllowedCalibrationPath(".gaze_calibration.json", configured, root)).toBe(configured);
    expect(resolveAllowedCalibrationPath("sub/../.gaze_calibration.json", configured, root)).toBe(configured);
    expect(resolveAllowedCalibrationPath(undefined, configured, root)).toBe(configured);
  });

  it("accepts a relative --calibration-file", () => {
    expect(resolveAllowedCalibrationPath("cal/mine.json", "cal/mine.json", root)).toBe(path.join(root, "cal/mine.json"));
  });

  it("rejects any other path", () => {
    for (const bad of ["/etc/passwd", "../outside.json", path.join(root, "package.json"), path.join(root, ".bashrc")]) {
      expect(resolveAllowedCalibrationPath(bad, configured, root)).toBeNull();
    }
  });
});
