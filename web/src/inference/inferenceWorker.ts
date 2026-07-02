import type { BackendMessage, WebInferenceConfig } from "../global";
import { logStatus } from "../logger";
import {
  Calibration,
  DisplayGeometry,
  LIP_MOTION_MARGIN_BOTTOM,
  LIP_MOTION_MARGIN_LEFT,
  LIP_MOTION_MARGIN_RIGHT,
  LIP_MOTION_MARGIN_TOP,
  LIP_MOTION_OPEN_THRESHOLD,
  ScreenProjector,
  center,
  clamp01,
  height,
  type Detection,
  width
} from "./core";
import { createRuntimeModels, type RuntimeModels } from "./runtimes";

type MainToWorkerMessage =
  | { type: "init"; config: WebInferenceConfig }
  | {
      type: "frame";
      data: Uint8ClampedArray<ArrayBuffer>;
      width: number;
      height: number;
      captureMs: number;
      frameIntervalMs?: number;
      droppedFrames: number;
      mainThreadLongTasks: number;
    }
  | { type: "capture_calibration"; target: [number, number] }
  | { type: "stop" }
  | {
      type: "calibration_read_result";
      id: number;
      response: { ok: true; text: string | null } | { ok: false; error: string };
    }
  | {
      type: "calibration_write_result";
      id: number;
      response: { ok: true; path: string } | { ok: false; error: string };
    };

type WorkerToMainMessage =
  | { type: "ready" }
  | { type: "frame_done" }
  | { type: "backend"; payload: BackendMessage }
  | { type: "calibration_read"; id: number; path: string }
  | { type: "calibration_write"; id: number; path: string; payload: unknown };

let config: WebInferenceConfig | null = null;
let models: RuntimeModels | null = null;
let calibration: Calibration | null = null;
let projector: ScreenProjector | null = null;
let latestRaw: [number, number] | null = null;
let smoothed: [number, number] | null = null;
let lastStatus = 0;
let lastPreview = 0;
let lastProjectionWarning = 0;
let lastProcessed = 0;
let fallbackInProgress = false;
let stopped = false;
let requestId = 0;
const pendingReads = new Map<number, (value: { ok: true; text: string | null } | { ok: false; error: string }) => void>();
const pendingWrites = new Map<number, (value: { ok: true; path: string } | { ok: false; error: string }) => void>();

const post = (message: WorkerToMainMessage) => {
  self.postMessage(message);
};

const emit = (payload: BackendMessage) => {
  post({ type: "backend", payload });
};

self.onmessage = (event: MessageEvent<MainToWorkerMessage>) => {
  void handleMessage(event.data);
};

async function handleMessage(message: MainToWorkerMessage): Promise<void> {
  if (message.type === "init") {
    await initialize(message.config);
  } else if (message.type === "frame") {
    await processFrameMessage(message);
  } else if (message.type === "capture_calibration") {
    await captureCalibration(message.target);
  } else if (message.type === "stop") {
    stopped = true;
    models?.dispose();
    models = null;
  } else if (message.type === "calibration_read_result") {
    pendingReads.get(message.id)?.(message.response);
    pendingReads.delete(message.id);
  } else if (message.type === "calibration_write_result") {
    pendingWrites.get(message.id)?.(message.response);
    pendingWrites.delete(message.id);
  }
}

async function initialize(nextConfig: WebInferenceConfig): Promise<void> {
  config = nextConfig;
  stopped = false;
  latestRaw = null;
  smoothed = null;
  lastStatus = 0;
  lastPreview = 0;
  lastProjectionWarning = 0;
  lastProcessed = 0;
  const headFaceWidthRatio = config.detector === "retinaface" ? config.retinafaceHeadFaceRatio : 1;
  calibration = new Calibration(config.calibrationFile, emit, {
    read: (path) => requestCalibrationRead(path),
    write: (path, payload) => requestCalibrationWrite(path, payload)
  });
  await calibration.load();
  projector = new ScreenProjector(
    new DisplayGeometry(config.displayWidth, config.displayHeight, config.displaySizeInch),
    config.flipX,
    config.flipY,
    config.cameraScreenX,
    config.cameraScreenY,
    config.eyePositionWeightX,
    config.eyePositionWeightY,
    config.cameraFov,
    config.cameraWidth,
    config.cameraHeight
  );
  try {
    emitLiteRtWorkerEnvironmentStatus();
    try {
      models = await createRuntimeModels(config, "webgpu");
    } catch (error) {
      const payload: BackendMessage = {
        type: "status",
        level: "warning",
        message: `WebGPU model load failed in inference worker: ${error}; falling back to wasm`,
        runtime: config.runtime,
        accelerator: "webgpu",
        logged: true
      };
      logStatus(payload, error);
      emit(payload);
      models = await createRuntimeModels(config, "wasm");
    }
    emitLiteRtRuntimeLoadedStatus();
    emitModelsLoaded(headFaceWidthRatio);
    post({ type: "ready" });
  } catch (error) {
    emit({
      type: "status",
      level: "error",
      message: error instanceof Error ? error.message : String(error),
      runtime: config.runtime,
      logged: true
    });
  }
}

function emitLiteRtWorkerEnvironmentStatus(): void {
  if (!config || config.runtime !== "litert") {
    return;
  }
  emit({
    type: "status",
    level: "info",
    message: `LiteRT inference worker starting; WebGPU ${hasWorkerWebGpu() ? "available" : "unavailable"}; wasm assets ${config.liteRtWasmBaseUrl}`,
    runtime: config.runtime,
    accelerator: "webgpu",
    logged: true
  });
}

function emitLiteRtRuntimeLoadedStatus(): void {
  if (!config || !models || config.runtime !== "litert") {
    return;
  }
  emit({
    type: "status",
    level: "info",
    message: `LiteRT inference worker loaded; detector ${models.detectorProviders[0] ?? "unknown"} / gaze ${
      models.gazeProviders[0] ?? "unknown"
    }${models.lipMotionProviders.length > 0 ? ` / lip ${models.lipMotionProviders[0]}` : ""}`,
    runtime: config.runtime,
    accelerator: models.accelerator,
    detector_providers: models.detectorProviders,
    gaze_providers: models.gazeProviders,
    lip_motion_providers: models.lipMotionProviders,
    logged: true
  });
}

function hasWorkerWebGpu(): boolean {
  return typeof globalThis.navigator !== "undefined" && "gpu" in globalThis.navigator;
}

function emitModelsLoaded(headFaceWidthRatio: number): void {
  if (!config || !models) {
    return;
  }
  emit({
    type: "status",
    level: "info",
    message: "Models loaded in inference worker",
    runtime: config.runtime,
    accelerator: models.accelerator,
    detector: config.detector,
    detector_model: config.detectorModel,
    detector_providers: models.detectorProviders,
    head_face_width_ratio: headFaceWidthRatio,
    camera_fov_deg: config.cameraFov,
    camera_resolution_name: config.cameraResolutionName,
    camera_width: config.cameraWidth,
    camera_height: config.cameraHeight,
    camera_screen_x: config.cameraScreenX,
    camera_screen_y: config.cameraScreenY,
    eye_position_weight_x: config.eyePositionWeightX,
    eye_position_weight_y: config.eyePositionWeightY,
    gaze_projection_mode: config.gazeProjectionMode,
    gaze_providers: models.gazeProviders,
    lip_motion_enabled: config.enableLipMotion && config.detector === "yolo",
    lip_motion_model: config.enableLipMotion && config.detector === "yolo" ? config.lipMotionModel : undefined,
    lip_motion_providers: models.lipMotionProviders
  });
}

async function processFrameMessage(message: Extract<MainToWorkerMessage, { type: "frame" }>): Promise<void> {
  try {
    if (stopped || config === null || models === null || calibration === null || projector === null) {
      return;
    }
    const frame = new ImageData(message.data, message.width, message.height);
    const headFaceWidthRatio = config.detector === "retinaface" ? config.retinafaceHeadFaceRatio : 1;
    const previewWidthRatio = config.detector === "retinaface" ? config.retinafaceHeadFaceRatio : null;
    const previewInterval = 1000 / Math.max(0.5, config.previewFps);
    const detectStart = performance.now();
    const { head, eyes, mouth } = await models.detect(frame);
    const detectEnd = performance.now();
    const detectInferenceMs = detectEnd - detectStart;
    const now = detectEnd;
    const processedIntervalMs = lastProcessed > 0 ? now - lastProcessed : message.frameIntervalMs;
    lastProcessed = now;
    const shouldEmitPreview = !config.hidePreview && now - lastPreview >= previewInterval;
    let previewMs = 0;
    if (head === null || eyes.length < 2) {
      if (shouldEmitPreview) {
        const previewMessage = head === null ? "Head not detected" : `Eyes detected: ${eyes.length}`;
        previewMs = await emitWorkerPreview(frame, head, eyes, mouth, previewMessage, null, previewWidthRatio);
        lastPreview = now;
      }
      throw new Error(head === null ? "Head was not detected" : "Two eyes were not detected");
    }

    const gazeStart = performance.now();
    const gaze = await models.estimate(frame, head, eyes);
    const gazeEnd = performance.now();
    const gazeInferenceMs = gazeEnd - gazeStart;
    let lipMotionInferenceMs: number | undefined;
    let mouthOpenProbability: number | null | undefined;
    let mouthOpen: boolean | undefined;
    if (config.enableLipMotion && config.detector === "yolo") {
      if (mouth !== null) {
        const lipStart = performance.now();
        mouthOpenProbability = await models.estimateLipMotion(frame, mouth);
        lipMotionInferenceMs = performance.now() - lipStart;
        mouthOpen = typeof mouthOpenProbability === "number" && Number.isFinite(mouthOpenProbability) && mouthOpenProbability >= LIP_MOTION_OPEN_THRESHOLD;
      } else {
        mouthOpenProbability = null;
        mouthOpen = false;
      }
    }
    if (shouldEmitPreview) {
      previewMs = await emitWorkerPreview(frame, head, eyes, mouth, null, [gaze.yawDeg, gaze.pitchDeg], previewWidthRatio);
      lastPreview = now;
    }
    const distanceM = projector.distanceFromHead(head, headFaceWidthRatio);
    const projection = projector.projectEstimate(config.gazeProjectionMode, eyes, gaze, distanceM);
    if (projection.fallbackReason && now - lastProjectionWarning > 2000) {
      emit({
        type: "status",
        level: "warning",
        message: `${projection.fallbackReason}; falling back to legacy projection`,
        runtime: config.runtime,
        accelerator: models.accelerator,
        gaze_projection_mode: config.gazeProjectionMode,
        logged: true
      });
      lastProjectionWarning = now;
    }
    latestRaw = projection.point;
    const corrected = calibration.apply(latestRaw);
    if (smoothed === null) {
      smoothed = corrected;
    } else {
      const alphaX = Math.max(0, Math.min(0.95, config.smoothingAlpha));
      const alphaY = Math.max(0, Math.min(0.95, config.smoothingAlphaY));
      smoothed = [
        alphaX * smoothed[0] + (1 - alphaX) * corrected[0],
        alphaY * smoothed[1] + (1 - alphaY) * corrected[1]
      ];
    }
    const confidence = Math.min(head.score, (eyes[0].score + eyes[1].score) * 0.5);
    const payload: BackendMessage = {
      type: "gaze",
      x_norm: clamp01(smoothed[0]),
      y_norm: clamp01(smoothed[1]),
      raw_x_norm: latestRaw[0],
      raw_y_norm: latestRaw[1],
      confidence,
      distance_m: distanceM,
      head_face_width_ratio: headFaceWidthRatio,
      eye_position_weight_x: projector.eyePositionWeightX,
      eye_position_weight_y: projector.eyePositionWeightY,
      gaze_projection_mode: config.gazeProjectionMode,
      detect_inference_ms: detectInferenceMs,
      gaze_inference_ms: gazeInferenceMs,
      lip_motion_inference_ms: lipMotionInferenceMs,
      inference_ms: detectInferenceMs + gazeInferenceMs + (lipMotionInferenceMs ?? 0),
      capture_ms: message.captureMs,
      preview_ms: previewMs,
      frame_interval_ms: processedIntervalMs,
      processed_fps: processedIntervalMs && processedIntervalMs > 0 ? 1000 / processedIntervalMs : undefined,
      dropped_frames: message.droppedFrames,
      main_thread_long_tasks: message.mainThreadLongTasks,
      yaw_deg: gaze.yawDeg,
      pitch_deg: gaze.pitchDeg
    };
    if (config.enableLipMotion && config.detector === "yolo") {
      payload.mouth_detected = mouth !== null;
      payload.mouth_open = mouthOpen;
      payload.mouth_open_probability = mouthOpenProbability;
    }
    emit(payload);
  } catch (error) {
    await handleFrameError(error);
  } finally {
    post({ type: "frame_done" });
  }
}

async function handleFrameError(error: unknown): Promise<void> {
  if (!config || !models) {
    return;
  }
  const now = performance.now();
  if (models.accelerator === "webgpu" && !fallbackInProgress && isRecoverableInferenceError(error)) {
    fallbackInProgress = true;
    const payload: BackendMessage = {
      type: "status",
      level: "warning",
      message: `WebGPU inference failed in worker: ${error instanceof Error ? error.message : String(error)}; falling back to wasm`,
      runtime: config.runtime,
      accelerator: "webgpu",
      logged: true
    };
    logStatus(payload, error);
    emit(payload);
    try {
      models.dispose();
      models = await createRuntimeModels(config, "wasm");
      emitModelsLoaded(config.detector === "retinaface" ? config.retinafaceHeadFaceRatio : 1);
    } catch (fallbackError) {
      const fallbackPayload: BackendMessage = {
        type: "status",
        level: "error",
        message: `Failed to reload worker models with wasm: ${fallbackError instanceof Error ? fallbackError.message : String(fallbackError)}`,
        runtime: config.runtime,
        accelerator: "wasm",
        logged: true
      };
      logStatus(fallbackPayload, fallbackError);
      emit(fallbackPayload);
    } finally {
      fallbackInProgress = false;
    }
  } else if (!fallbackInProgress && now - lastStatus > 1000) {
    const payload: BackendMessage = {
      type: "status",
      level: "warning",
      message: error instanceof Error ? error.message : String(error),
      runtime: config.runtime,
      accelerator: models.accelerator,
      logged: true
    };
    logStatus(payload, error);
    emit(payload);
    lastStatus = now;
  }
}

async function captureCalibration(target: [number, number]): Promise<void> {
  if (calibration === null || config === null) {
    return;
  }
  try {
    await calibration.capture(latestRaw, target);
  } catch (error) {
    emit({
      type: "status",
      level: "error",
      message: `Calibration capture failed: ${error instanceof Error ? error.message : String(error)}`,
      runtime: config.runtime,
      logged: true
    });
  }
}

function requestCalibrationRead(path: string): Promise<{ ok: true; text: string | null } | { ok: false; error: string }> {
  const id = requestId + 1;
  requestId = id;
  post({ type: "calibration_read", id, path });
  return new Promise((resolve) => pendingReads.set(id, resolve));
}

function requestCalibrationWrite(path: string, payload: unknown): Promise<{ ok: true; path: string } | { ok: false; error: string }> {
  const id = requestId + 1;
  requestId = id;
  post({ type: "calibration_write", id, path, payload });
  return new Promise((resolve) => pendingWrites.set(id, resolve));
}

async function emitWorkerPreview(
  frame: ImageData,
  head: Detection | null,
  eyes: Detection[],
  mouth: Detection | null,
  message: string | null,
  gazeAngles: [number, number] | null,
  widthRatio: number | null
): Promise<number> {
  const start = performance.now();
  const image = await createPreviewImageInWorker(frame, head, eyes, mouth, message, widthRatio, gazeAngles ?? undefined);
  if (image) {
    emit({
      type: "preview",
      image,
      head_detected: head !== null,
      eye_count: eyes.length,
      width_ratio: widthRatio
    });
  }
  return performance.now() - start;
}

async function createPreviewImageInWorker(
  frame: ImageData,
  head: Detection | null,
  eyes: Detection[],
  mouth: Detection | null,
  message: string | null,
  widthRatio: number | null,
  gazeAngles?: [number, number]
): Promise<string> {
  if (typeof OffscreenCanvas === "undefined") {
    return "";
  }
  const source = new OffscreenCanvas(frame.width, frame.height);
  const sourceCtx = source.getContext("2d");
  if (!sourceCtx) {
    return "";
  }
  sourceCtx.putImageData(frame, 0, 0);
  if (head) {
    drawBox(sourceCtx, head, "#14dc5a", "Head");
  }
  for (const eye of eyes) {
    drawBox(sourceCtx, eye, "#00d2ff", "Eye");
    const [cx, cy] = center(eye);
    sourceCtx.fillStyle = "#00d2ff";
    sourceCtx.beginPath();
    sourceCtx.arc(cx, cy, 4, 0, Math.PI * 2);
    sourceCtx.fill();
  }
  if (mouth) {
    drawBox(sourceCtx, mouthDetectionWithMargin(frame, mouth), "#ff8c00", "Mouth");
  }
  if (gazeAngles && eyes.length >= 2) {
    drawGazeLines(sourceCtx, eyes, gazeAngles[0], gazeAngles[1]);
  }
  if (message) {
    strokeText(sourceCtx, message, 12, 28, 22);
  }
  if (widthRatio !== null) {
    strokeText(sourceCtx, `Head/Face ${widthRatio.toFixed(3)}x`, 12, frame.height - 14, 18);
  }
  const preview = new OffscreenCanvas(320, Math.max(1, Math.round((frame.height * 320) / Math.max(1, frame.width))));
  const previewCtx = preview.getContext("2d");
  if (!previewCtx) {
    return "";
  }
  previewCtx.drawImage(source, 0, 0, preview.width, preview.height);
  const blob = await preview.convertToBlob({ type: "image/jpeg", quality: 0.72 });
  return blobToDataUrl(blob);
}

async function blobToDataUrl(blob: Blob): Promise<string> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  const chunkSize = 0x8000;
  for (let index = 0; index < bytes.length; index += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
  }
  return `data:${blob.type};base64,${btoa(binary)}`;
}

function drawBox(ctx: OffscreenCanvasRenderingContext2D, det: Detection, color: string, label: string): void {
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  ctx.strokeRect(det.x1, det.y1, width(det), height(det));
  strokeText(ctx, `${label} ${det.score.toFixed(2)}`, det.x1, Math.max(18, det.y1 - 6), 16, color);
}

function strokeText(
  ctx: OffscreenCanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  size: number,
  fill = "#ffffff"
): void {
  ctx.font = `${size}px Inter, sans-serif`;
  ctx.lineWidth = 4;
  ctx.strokeStyle = "#000000";
  ctx.fillStyle = fill;
  ctx.strokeText(text, x, y);
  ctx.fillText(text, x, y);
}

function drawGazeLines(ctx: OffscreenCanvasRenderingContext2D, eyes: Detection[], yawDeg: number, pitchDeg: number): void {
  const length = 180;
  const dx = length * Math.sin((yawDeg * Math.PI) / 180);
  const dy = length * Math.sin((-pitchDeg * Math.PI) / 180);
  for (const eye of eyes.slice(0, 2)) {
    const [cx, cy] = center(eye);
    ctx.lineCap = "round";
    ctx.lineWidth = 7;
    ctx.strokeStyle = "#000000";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + dx, cy + dy);
    ctx.stroke();
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#00d084";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + dx, cy + dy);
    ctx.stroke();
  }
}

function mouthDetectionWithMargin(frame: ImageData, mouth: Detection): Detection {
  const x1 = Math.max(0, Math.min(frame.width - 1, Math.floor(mouth.x1) - LIP_MOTION_MARGIN_LEFT));
  const y1 = Math.max(0, Math.min(frame.height - 1, Math.floor(mouth.y1) - LIP_MOTION_MARGIN_TOP));
  const x2 = Math.max(x1 + 1, Math.min(frame.width, Math.ceil(mouth.x2) + LIP_MOTION_MARGIN_RIGHT));
  const y2 = Math.max(y1 + 1, Math.min(frame.height, Math.ceil(mouth.y2) + LIP_MOTION_MARGIN_BOTTOM));
  return { ...mouth, x1, y1, x2, y2 };
}

function isRecoverableInferenceError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return (
    message.includes("OrtRun") ||
    message.includes("bad_array_new_length") ||
    message.includes("failed to call") ||
    message.includes("ERROR_CODE")
  );
}
