const path = require("path");

/**
 * Returns the absolute calibration path if `requested` resolves to exactly the configured
 * calibration file, otherwise null. Relative paths resolve against `baseDir` (the repo root).
 */
function resolveAllowedCalibrationPath(requested, configured, baseDir) {
  const allowed = path.resolve(baseDir, configured);
  const candidate = path.resolve(baseDir, requested || allowed);
  return candidate === allowed ? allowed : null;
}

module.exports = { resolveAllowedCalibrationPath };
