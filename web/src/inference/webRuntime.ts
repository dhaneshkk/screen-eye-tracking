import type { BackendMessage, WebInferenceConfig } from "../global";
import { logStatus } from "../logger";
import {
  Calibration,
  DisplayGeometry,
  ScreenProjector,
  clamp01,
  createPreviewImage,
  LIP_MOTION_OPEN_THRESHOLD,
  type Detection
} from "./core";
import InferenceWorker from "./inferenceWorker.ts?worker";
import { createRuntimeModels, type RuntimeModels } from "./runtimes";

export interface WebRuntimeController {
  captureCalibration(target: [number, number]): Promise<void>;
  stop(): void;
}

export async function startWebRuntime(
  config: WebInferenceConfig,
  emit: (payload: BackendMessage) => void
): Promise<WebRuntimeController> {
  if (config.webInferenceWorker === "dedicated") {
    return startDedicatedWorkerRuntime(config, emit);
  }
  if (config.webInferenceWorker === "ort-proxy" && config.runtime !== "onnxweb") {
    emit({
      type: "status",
      level: "warning",
      message: "--web-inference-worker ort-proxy is only supported by onnxweb; using normal renderer inference",
      runtime: config.runtime,
      logged: true
    });
  }
  let stopped = false;
  let latestRaw: [number, number] | null = null;
  let smoothed: [number, number] | null = null;
  let stream: MediaStream | null = null;
  let models: RuntimeModels | null = null;
  let lastStatus = 0;
  let lastPreview = 0;
  let lastProcessed = 0;
  let lastProjectionWarning = 0;
  let fallbackInProgress = false;
  const longTasks = createLongTaskCounter();

  const calibration = new Calibration(config.calibrationFile, emit);
  await calibration.load();

  const headFaceWidthRatio = config.detector === "retinaface" ? config.retinafaceHeadFaceRatio : 1;
  const previewWidthRatio = config.detector === "retinaface" ? config.retinafaceHeadFaceRatio : null;

  try {
    const useOrtProxy = config.webInferenceWorker === "ort-proxy" && config.runtime === "onnxweb";
    try {
      models = useOrtProxy ? await createRuntimeModels(config, "wasm") : await createRuntimeModels(config, "webgpu");
    } catch (error) {
      if (useOrtProxy) {
        throw error;
      }
      const payload: BackendMessage = {
        type: "status",
        level: "warning",
        message: `WebGPU model load failed: ${error}; falling back to wasm`,
        runtime: config.runtime,
        accelerator: "webgpu",
        logged: true
      };
      logStatus(payload, error);
      emit(payload);
      models = await createRuntimeModels(config, "wasm");
    }
    emit({
      type: "status",
      level: "info",
      message: "Models loaded",
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

    const camera = await openCamera(config.camera, config.cameraWidth, config.cameraHeight);
    stream = camera.stream;
    const video = camera.video;
    const canvas = document.createElement("canvas");
    canvas.width = config.cameraWidth;
    canvas.height = config.cameraHeight;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      throw new Error("2D canvas context is not available");
    }

    const projector = new ScreenProjector(
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
    const previewInterval = 1000 / Math.max(0.5, config.previewFps);

    const loop = async () => {
      if (stopped || models === null) {
        return;
      }
      try {
        const captureStart = performance.now();
        ctx.drawImage(video, 0, 0, config.cameraWidth, config.cameraHeight);
        const frame = ctx.getImageData(0, 0, config.cameraWidth, config.cameraHeight);
        const captureMs = performance.now() - captureStart;
        const detectStart = performance.now();
        const { head, eyes, mouth } = await models.detect(frame);
        const detectEnd = performance.now();
        const detectInferenceMs = detectEnd - detectStart;
        const now = detectEnd;
        const frameIntervalMs = lastProcessed > 0 ? now - lastProcessed : undefined;
        lastProcessed = now;
        const processedFps = frameIntervalMs && frameIntervalMs > 0 ? 1000 / frameIntervalMs : undefined;
        const shouldEmitPreview = !config.hidePreview && now - lastPreview >= previewInterval;
        let previewMs = 0;
        if (head === null || eyes.length < 2) {
          if (shouldEmitPreview) {
            const message = head === null ? "Head not detected" : `Eyes detected: ${eyes.length}`;
            previewMs = emitPreview(frame, head, eyes, mouth, message, null, previewWidthRatio, emit);
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
          previewMs = emitPreview(frame, head, eyes, mouth, null, [gaze.yawDeg, gaze.pitchDeg], previewWidthRatio, emit);
          lastPreview = now;
        }
        const distanceM = projector.distanceFromHead(head, headFaceWidthRatio);
        const projection = projector.projectEstimate(config.gazeProjectionMode, eyes, gaze, distanceM);
        if (projection.fallbackReason && now - lastProjectionWarning > 2000) {
          const payload: BackendMessage = {
            type: "status",
            level: "warning",
            message: `${projection.fallbackReason}; falling back to legacy projection`,
            runtime: config.runtime,
            accelerator: models.accelerator,
            gaze_projection_mode: config.gazeProjectionMode,
            logged: true
          };
          logStatus(payload);
          emit(payload);
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
          capture_ms: captureMs,
          preview_ms: previewMs,
          frame_interval_ms: frameIntervalMs,
          processed_fps: processedFps,
          dropped_frames: 0,
          main_thread_long_tasks: longTasks.count(),
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
        const now = performance.now();
        if (models?.accelerator === "webgpu" && !fallbackInProgress && isRecoverableInferenceError(error)) {
          fallbackInProgress = true;
          const payload: BackendMessage = {
            type: "status",
            level: "warning",
            message: `WebGPU inference failed: ${error instanceof Error ? error.message : String(error)}; falling back to wasm`,
            runtime: config.runtime,
            accelerator: "webgpu",
            logged: true
          };
          logStatus(payload, error);
          emit(payload);
          try {
            models.dispose();
            models = await createRuntimeModels(config, "wasm");
            emit({
              type: "status",
              level: "info",
              message: "Models reloaded with wasm",
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
          } catch (fallbackError) {
            const fallbackPayload: BackendMessage = {
              type: "status",
              level: "error",
              message: `Failed to reload models with wasm: ${fallbackError instanceof Error ? fallbackError.message : String(fallbackError)}`,
              runtime: config.runtime,
              accelerator: "wasm",
              logged: true
            };
            logStatus(fallbackPayload, fallbackError);
            emit(fallbackPayload);
          } finally {
            fallbackInProgress = false;
          }
        } else if (fallbackInProgress) {
          // Suppress repeated frame errors while runtime fallback is rebuilding sessions.
        } else {
          if (now - lastStatus > 1000) {
            const payload: BackendMessage = {
              type: "status",
              level: "warning",
              message: error instanceof Error ? error.message : String(error),
              runtime: config.runtime,
              accelerator: models?.accelerator,
              logged: true
            };
            logStatus(payload, error);
            emit(payload);
            lastStatus = now;
          }
        }
      } finally {
        if (!stopped) {
          window.setTimeout(loop, 33);
        }
      }
    };
    void loop();
  } catch (error) {
    const payload: BackendMessage = {
      type: "status",
      level: "error",
      message: error instanceof Error ? error.message : String(error),
      runtime: config.runtime,
      accelerator: models?.accelerator,
      logged: true
    };
    logStatus(payload, error);
    emit(payload);
  }

  return {
    captureCalibration: (target) => captureCalibration(calibration, latestRaw, target, emit, config),
    stop: () => {
      stopped = true;
      longTasks.stop();
      models?.dispose();
      for (const track of stream?.getTracks() ?? []) {
        track.stop();
      }
    }
  };
}

function createLongTaskCounter(): { count(): number; stop(): void } {
  let count = 0;
  let observer: PerformanceObserver | null = null;
  try {
    observer = new PerformanceObserver((list) => {
      count += list.getEntries().length;
    });
    observer.observe({ entryTypes: ["longtask"] });
  } catch {
    observer = null;
  }
  return {
    count: () => count,
    stop: () => observer?.disconnect()
  };
}

type WorkerToMainMessage =
  | { type: "ready" }
  | { type: "frame_done" }
  | { type: "backend"; payload: BackendMessage }
  | { type: "calibration_read"; id: number; path: string }
  | { type: "calibration_write"; id: number; path: string; payload: unknown };

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

async function startDedicatedWorkerRuntime(
  config: WebInferenceConfig,
  emit: (payload: BackendMessage) => void
): Promise<WebRuntimeController> {
  let stopped = false;
  let stream: MediaStream | null = null;
  let inFlight = false;
  let droppedFrames = 0;
  let lastCapture = 0;
  let captureTimer: number | null = null;
  const longTasks = createLongTaskCounter();
  const worker = new InferenceWorker();

  const post = (message: MainToWorkerMessage, transfer?: Transferable[]) => {
    worker.postMessage(message, transfer ?? []);
  };

  worker.onmessage = (event: MessageEvent<WorkerToMainMessage>) => {
    const message = event.data;
    if (message.type === "backend") {
      emit(message.payload);
    } else if (message.type === "frame_done") {
      inFlight = false;
    } else if (message.type === "calibration_read") {
      window.gazeBridge.readCalibration(message.path).then((response) => {
        post({ type: "calibration_read_result", id: message.id, response });
      });
    } else if (message.type === "calibration_write") {
      window.gazeBridge.writeCalibration(message.path, message.payload).then((response) => {
        post({ type: "calibration_write_result", id: message.id, response });
      });
    }
  };
  worker.onerror = (event) => {
    emit({
      type: "status",
      level: "error",
      message: `Inference worker error: ${event.message}`,
      runtime: config.runtime,
      logged: true
    });
  };

  post({ type: "init", config: { ...config, assetBaseUrl: window.location.href } });

  try {
    const camera = await openCamera(config.camera, config.cameraWidth, config.cameraHeight);
    stream = camera.stream;
    const video = camera.video;
    const canvas = document.createElement("canvas");
    canvas.width = config.cameraWidth;
    canvas.height = config.cameraHeight;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      throw new Error("2D canvas context is not available");
    }

    const captureLoop = () => {
      if (stopped) {
        return;
      }
      if (inFlight) {
        droppedFrames += 1;
      } else {
        const captureStart = performance.now();
        ctx.drawImage(video, 0, 0, config.cameraWidth, config.cameraHeight);
        const frame = ctx.getImageData(0, 0, config.cameraWidth, config.cameraHeight);
        const now = performance.now();
        const frameIntervalMs = lastCapture > 0 ? now - lastCapture : undefined;
        lastCapture = now;
        inFlight = true;
        post(
          {
            type: "frame",
            data: frame.data,
            width: frame.width,
            height: frame.height,
            captureMs: now - captureStart,
            frameIntervalMs,
            droppedFrames,
            mainThreadLongTasks: longTasks.count()
          },
          [frame.data.buffer]
        );
      }
      captureTimer = window.setTimeout(captureLoop, 33);
    };
    captureLoop();
  } catch (error) {
    emit({
      type: "status",
      level: "error",
      message: error instanceof Error ? error.message : String(error),
      runtime: config.runtime,
      logged: true
    });
  }

  return {
    captureCalibration: async (target) => {
      post({ type: "capture_calibration", target });
    },
    stop: () => {
      stopped = true;
      if (captureTimer !== null) {
        window.clearTimeout(captureTimer);
      }
      longTasks.stop();
      post({ type: "stop" });
      worker.terminate();
      for (const track of stream?.getTracks() ?? []) {
        track.stop();
      }
    }
  };
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

async function captureCalibration(
  calibration: Calibration,
  raw: [number, number] | null,
  target: [number, number],
  emit: (payload: BackendMessage) => void,
  config: WebInferenceConfig
): Promise<void> {
  console.info("Capturing calibration sample", JSON.stringify({ raw, target }));
  try {
    await calibration.capture(raw, target);
  } catch (error) {
    const payload: BackendMessage = {
      type: "status",
      level: "error",
      message: `Calibration capture failed: ${error instanceof Error ? error.message : String(error)}`,
      runtime: config.runtime,
      logged: true
    };
    logStatus(payload, error);
    emit(payload);
  }
}

async function openCamera(
  camera: string,
  expectedWidth: number,
  expectedHeight: number
): Promise<{ stream: MediaStream; video: HTMLVideoElement }> {
  const constraints = await cameraConstraints(camera);
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      ...constraints,
      width: { exact: expectedWidth },
      height: { exact: expectedHeight }
    },
    audio: false
  });
  const video = document.createElement("video");
  video.autoplay = true;
  video.muted = true;
  video.playsInline = true;
  video.srcObject = stream;
  await video.play();
  await waitForVideo(video);
  const settings = stream.getVideoTracks()[0]?.getSettings();
  const actualWidth = settings?.width ?? video.videoWidth;
  const actualHeight = settings?.height ?? video.videoHeight;
  if (actualWidth !== expectedWidth || actualHeight !== expectedHeight) {
    for (const track of stream.getTracks()) {
      track.stop();
    }
    throw new Error(`Camera returned ${actualWidth}x${actualHeight}, expected ${expectedWidth}x${expectedHeight}`);
  }
  return { stream, video };
}

async function cameraConstraints(camera: string): Promise<MediaTrackConstraints> {
  const index = Number.parseInt(camera, 10);
  if (Number.isFinite(index) && String(index) === camera.trim()) {
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === "videoinput");
    const selected = devices[index];
    return selected?.deviceId ? { deviceId: { exact: selected.deviceId } } : {};
  }
  if (camera) {
    return { deviceId: { exact: camera } };
  }
  return {};
}

function waitForVideo(video: HTMLVideoElement): Promise<void> {
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error("Camera video did not become ready"));
    }, 5000);
    const onReady = () => {
      cleanup();
      resolve();
    };
    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener("loadeddata", onReady);
    };
    video.addEventListener("loadeddata", onReady, { once: true });
  });
}

function emitPreview(
  frame: ImageData,
  head: Detection | null,
  eyes: Detection[],
  mouth: Detection | null,
  message: string | null,
  gazeAngles: [number, number] | null,
  widthRatio: number | null,
  emit: (payload: BackendMessage) => void
): number {
  const start = performance.now();
  const image = createPreviewImage(frame, head, eyes, mouth, message, widthRatio, gazeAngles ?? undefined);
  if (!image) {
    return performance.now() - start;
  }
  emit({
    type: "preview",
    image,
    head_detected: head !== null,
    eye_count: eyes.length,
    width_ratio: widthRatio
  });
  return performance.now() - start;
}
