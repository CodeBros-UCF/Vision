"""
GazeTrack Engine - High Performance Multi-Threaded Eye Tracking
================================================================
Architecture:
  Thread 1 (CaptureThread)   - Grabs webcam frames as fast as possible
  Thread 2 (InferenceThread) - Runs Mediapipe FaceMesh on every Nth frame
  Thread 3 (RenderThread)    - Composites UI overlay + updates mouse position

Adaptive GPU Tiers:
  LIGHTWEIGHT  - Integrated / No GPU  → 320x240, 20 FPS inference, basic smoothing
  BALANCED     - Mid-tier GPU          → 640x480, 30 FPS inference, Kalman filter
  ENHANCED     - High-end GPU          → 1280x720, 60 FPS inference, predictive model
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import threading
import time
import collections
import platform
import subprocess
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

# ─── Disable pyautogui failsafe for smoother operation ───────────────────────
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # Remove pyautogui's built-in 0.1s delay

# ─── GPU Tier Definitions ─────────────────────────────────────────────────────
class GPUTier(Enum):
    LIGHTWEIGHT = "lightweight"
    BALANCED    = "balanced"
    ENHANCED    = "enhanced"

@dataclass
class TierConfig:
    name: str
    description: str
    cam_width: int
    cam_height: int
    target_inference_fps: int
    smoothing_alpha: float       # EMA alpha for low-end
    use_kalman: bool
    use_prediction: bool
    display_fps: int             # Target render FPS
    color: str                   # Hex color for UI badge

TIER_CONFIGS = {
    GPUTier.LIGHTWEIGHT: TierConfig(
        name="Lightweight Mode",
        description="Integrated / No GPU — optimized for minimal CPU load",
        cam_width=320, cam_height=240,
        target_inference_fps=20,
        smoothing_alpha=0.5,
        use_kalman=False,
        use_prediction=False,
        display_fps=30,
        color="#f59e0b",
    ),
    GPUTier.BALANCED: TierConfig(
        name="Balanced Mode",
        description="Mid-tier GPU (GTX 1050–1660) — smooth tracking with Kalman filter",
        cam_width=640, cam_height=480,
        target_inference_fps=30,
        smoothing_alpha=0.3,
        use_kalman=True,
        use_prediction=False,
        display_fps=60,
        color="#06b6d4",
    ),
    GPUTier.ENHANCED: TierConfig(
        name="Enhanced AI Mode",
        description="High-end GPU (RTX 2070+) — full precision + predictive gaze model",
        cam_width=1280, cam_height=720,
        target_inference_fps=60,
        smoothing_alpha=0.2,
        use_kalman=True,
        use_prediction=True,
        display_fps=120,
        color="#a855f7",
    ),
}

# ─── GPU Benchmarking ─────────────────────────────────────────────────────────
@dataclass
class GPUInfo:
    name: str = "Unknown"
    vram_mb: int = 0
    tier: GPUTier = GPUTier.LIGHTWEIGHT
    driver_version: str = "N/A"

def benchmark_gpu() -> GPUInfo:
    """Query GPU info using nvidia-smi or GPUtil. Falls back gracefully."""
    info = GPUInfo()
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                info.name           = parts[0].strip()
                info.vram_mb        = int(parts[1].strip())
                info.driver_version = parts[2].strip() if len(parts) > 2 else "N/A"

                # Tier classification based on VRAM heuristic
                vram_gb = info.vram_mb / 1024
                name_upper = info.name.upper()

                # High-end check (RTX 2070+ / 8GB+)
                high_end_keywords = ["RTX 3", "RTX 4", "RTX 2070", "RTX 2080",
                                     "A4000", "A5000", "A6000", "TITAN", "V100", "A100"]
                mid_tier_keywords = ["GTX 1650", "GTX 1660", "RTX 2060",
                                     "GTX 970", "GTX 980", "GTX 1070", "GTX 1080",
                                     "GTX 1050", "RTX 2050", "MX"]

                if any(k in name_upper for k in high_end_keywords) or vram_gb >= 8:
                    info.tier = GPUTier.ENHANCED
                elif any(k in name_upper for k in mid_tier_keywords) or vram_gb >= 4:
                    info.tier = GPUTier.BALANCED
                else:
                    info.tier = GPUTier.LIGHTWEIGHT
                return info
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    # Try GPUtil as fallback
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            g = gpus[0]
            info.name    = g.name
            info.vram_mb = int(g.memoryTotal)
            vram_gb      = info.vram_mb / 1024
            name_upper   = g.name.upper()
            if "RTX" in name_upper and vram_gb >= 8:
                info.tier = GPUTier.ENHANCED
            elif vram_gb >= 4:
                info.tier = GPUTier.BALANCED
            else:
                info.tier = GPUTier.LIGHTWEIGHT
            return info
    except ImportError:
        pass

    info.name = "CPU / Integrated Graphics"
    info.tier = GPUTier.LIGHTWEIGHT
    return info

# ─── Kalman Filter for gaze smoothing ────────────────────────────────────────
class GazeKalmanFilter:
    """
    2D Kalman Filter tracking (x, y) with velocity prediction.
    State vector: [x, y, vx, vy]
    """
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        dt = 1.0
        # Transition matrix (constant velocity)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)
        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-4
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self.initialized = False

    def update(self, x: float, y: float):
        measurement = np.array([[x], [y]], dtype=np.float32)
        if not self.initialized:
            self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.initialized  = True
        self.kf.correct(measurement)
        predicted = self.kf.predict()
        return float(predicted[0]), float(predicted[1])

    def reset(self):
        self.initialized = False

# ─── Thread-safe ring buffer ──────────────────────────────────────────────────
class LatencyBuffer:
    """Stores the last N latency samples for computing rolling averages."""
    def __init__(self, maxlen=60):
        self._lock   = threading.Lock()
        self._buffer = collections.deque(maxlen=maxlen)

    def push(self, value_ms: float):
        with self._lock:
            self._buffer.append(value_ms)

    def avg(self) -> float:
        with self._lock:
            if not self._buffer:
                return 0.0
            return sum(self._buffer) / len(self._buffer)

    def last(self) -> float:
        with self._lock:
            return self._buffer[-1] if self._buffer else 0.0

# ─── Shared State (lock-protected) ───────────────────────────────────────────
class SharedState:
    def __init__(self):
        self._lock           = threading.Lock()
        self.latest_frame    = None   # BGR ndarray from capture thread
        self.latest_result   = None   # Dict from inference thread
        self.capture_ts      = 0.0    # Timestamp of latest captured frame
        self.inference_ts    = 0.0    # Timestamp of latest inference result
        self.running         = True
        self.paused          = False

    def write_frame(self, frame, ts):
        with self._lock:
            self.latest_frame = frame
            self.capture_ts   = ts

    def read_frame(self):
        with self._lock:
            return self.latest_frame, self.capture_ts

    def write_result(self, result, ts):
        with self._lock:
            self.latest_result = result
            self.inference_ts  = ts

    def read_result(self):
        with self._lock:
            return self.latest_result, self.inference_ts

    def is_running(self):
        with self._lock:
            return self.running

    def is_paused(self):
        with self._lock:
            return self.paused

    def stop(self):
        with self._lock:
            self.running = False

    def set_paused(self, val: bool):
        with self._lock:
            self.paused = val

# ─── Metrics ──────────────────────────────────────────────────────────────────
@dataclass
class Metrics:
    capture_fps:   float = 0.0
    inference_fps: float = 0.0
    render_fps:    float = 0.0
    cap_to_inf_latency_ms: float = 0.0   # How stale the frame is when inference runs
    inf_to_render_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    vram_used_mb: int = 0
    cpu_bottleneck: bool = False          # True if capture is significantly faster than inference
    gaze_x: int = 0
    gaze_y: int = 0
    blink_detected: bool = False

class MetricsStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._m    = Metrics()

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self._m, k, v)

    def snapshot(self) -> Metrics:
        with self._lock:
            import copy
            return copy.copy(self._m)

# ─── Capture Thread ───────────────────────────────────────────────────────────
class CaptureThread(threading.Thread):
    def __init__(self, shared: SharedState, metrics: MetricsStore, config: TierConfig):
        super().__init__(daemon=True, name="CaptureThread")
        self.shared  = shared
        self.metrics = metrics
        self.config  = config
        self._fps_counter = collections.deque(maxlen=30)

    def run(self):
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW faster on Windows
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  self.config.cam_width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.cam_height)
        cam.set(cv2.CAP_PROP_FPS, 60)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag

        while self.shared.is_running():
            ret, frame = cam.read()
            if not ret:
                time.sleep(0.005)
                continue

            ts = time.perf_counter()
            frame = cv2.flip(frame, 1)
            self.shared.write_frame(frame, ts)

            # FPS tracking
            self._fps_counter.append(ts)
            if len(self._fps_counter) >= 2:
                elapsed = self._fps_counter[-1] - self._fps_counter[0]
                fps     = (len(self._fps_counter) - 1) / elapsed if elapsed > 0 else 0
                self.metrics.update(capture_fps=fps)

        cam.release()

# ─── Inference Thread ─────────────────────────────────────────────────────────
class InferenceThread(threading.Thread):
    def __init__(self, shared: SharedState, metrics: MetricsStore, config: TierConfig):
        super().__init__(daemon=True, name="InferenceThread")
        self.shared  = shared
        self.metrics = metrics
        self.config  = config
        self._kalman = GazeKalmanFilter() if config.use_kalman else None
        self._ema_x  = None
        self._ema_y  = None
        self._fps_counter = collections.deque(maxlen=30)
        self._lat_buf     = LatencyBuffer()
        self._screen_w, self._screen_h = pyautogui.size()

    def _ema(self, prev, current, alpha):
        if prev is None:
            return current
        return alpha * current + (1 - alpha) * prev

    def run(self):
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        interval = 1.0 / self.config.target_inference_fps
        last_frame_ts = -1.0

        while self.shared.is_running():
            t_start = time.perf_counter()

            if self.shared.is_paused():
                time.sleep(0.01)
                continue

            frame, cap_ts = self.shared.read_frame()
            if frame is None or cap_ts == last_frame_ts:
                time.sleep(0.001)
                continue

            last_frame_ts = cap_ts
            cap_latency   = (t_start - cap_ts) * 1000  # ms

            # Run Mediapipe
            h, w = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            proc  = face_mesh.process(rgb)

            result = {
                "landmarks":      None,
                "gaze_screen_x":  None,
                "gaze_screen_y":  None,
                "left_eye":       None,
                "blink":          False,
                "iris_landmarks": None,
            }

            if proc.multi_face_landmarks:
                lm  = proc.multi_face_landmarks[0].landmark
                result["landmarks"] = lm

                # Left eye blink detection (landmarks 145=lower, 159=upper)
                left_eye = [lm[145], lm[159]]
                result["left_eye"] = left_eye
                blink = (left_eye[0].y - left_eye[1].y) < 0.01
                result["blink"] = blink

                # Iris landmarks 474-477
                iris = [lm[474], lm[475], lm[476], lm[477]]
                result["iris_landmarks"] = iris

                # Raw gaze position from landmark 475 (center-ish of iris)
                raw_x = int(iris[1].x * w)
                raw_y = int(iris[1].y * h)
                sx    = int(self._screen_w / w * raw_x)
                sy    = int(self._screen_h / h * raw_y)

                if self._kalman:
                    sx, sy = self._kalman.update(sx, sy)
                else:
                    self._ema_x = self._ema(self._ema_x, sx, self.config.smoothing_alpha)
                    self._ema_y = self._ema(self._ema_y, sy, self.config.smoothing_alpha)
                    sx, sy = self._ema_x, self._ema_y

                result["gaze_screen_x"] = int(sx)
                result["gaze_screen_y"] = int(sy)

            inf_ts = time.perf_counter()
            inf_latency = (inf_ts - t_start) * 1000

            self.shared.write_result(result, inf_ts)
            self._lat_buf.push(cap_latency)

            self.metrics.update(
                cap_to_inf_latency_ms=self._lat_buf.avg(),
                blink_detected=result["blink"],
                gaze_x=result.get("gaze_screen_x") or 0,
                gaze_y=result.get("gaze_screen_y") or 0,
            )

            # FPS tracking
            self._fps_counter.append(inf_ts)
            if len(self._fps_counter) >= 2:
                elapsed = self._fps_counter[-1] - self._fps_counter[0]
                fps     = (len(self._fps_counter) - 1) / elapsed if elapsed > 0 else 0
                self.metrics.update(inference_fps=fps)

            # Sleep to hit target inference FPS
            elapsed_total = time.perf_counter() - t_start
            sleep_time    = interval - elapsed_total
            if sleep_time > 0:
                time.sleep(sleep_time)

        face_mesh.close()

# ─── Render Thread ────────────────────────────────────────────────────────────
class RenderThread(threading.Thread):
    """
    Reads the latest frame + inference result, draws the overlay, shows it,
    and moves the mouse cursor.
    """
    def __init__(self, shared: SharedState, metrics: MetricsStore,
                 config: TierConfig, gpu_info: GPUInfo,
                 on_metrics_update: Optional[Callable] = None):
        super().__init__(daemon=True, name="RenderThread")
        self.shared            = shared
        self.metrics           = metrics
        self.config            = config
        self.gpu_info          = gpu_info
        self.on_metrics_update = on_metrics_update

        self._fps_counter    = collections.deque(maxlen=30)
        self._lat_buf        = LatencyBuffer()
        self._click_held     = False

        # GPU VRAM polling
        self._vram_poll_interval = 2.0  # seconds
        self._last_vram_poll     = 0.0

    def _get_vram_used(self) -> int:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass
        return 0

    def _draw_overlay(self, frame, result, m: Metrics):
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent dark bar at top
        cv2.rectangle(overlay, (0, 0), (w, 60), (10, 10, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Metrics text
        fps_txt     = f"Capture: {m.capture_fps:.0f}fps  Inf: {m.inference_fps:.0f}fps  Render: {m.render_fps:.0f}fps"
        latency_txt = f"Latency  Cap→Inf: {m.cap_to_inf_latency_ms:.1f}ms  Total: {m.total_latency_ms:.1f}ms"

        cv2.putText(frame, fps_txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, latency_txt, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (100, 220, 255), 1, cv2.LINE_AA)

        # GPU tier badge top-right
        tier_label = self.config.name.split(" ")[0]
        cv2.putText(frame, tier_label, (w - 120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 160, 255), 1, cv2.LINE_AA)

        # Draw iris landmarks
        if result and result["iris_landmarks"]:
            for lm in result["iris_landmarks"]:
                px = int(lm.x * w)
                py = int(lm.y * h)
                cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)

        # Draw left eye landmarks
        if result and result["left_eye"]:
            for lm in result["left_eye"]:
                px = int(lm.x * w)
                py = int(lm.y * h)
                cv2.circle(frame, (px, py), 3, (0, 255, 100), -1)

        # Blink indicator
        if result and result["blink"]:
            cv2.putText(frame, "BLINK", (w // 2 - 40, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2, cv2.LINE_AA)

        return frame

    def run(self):
        interval = 1.0 / self.config.display_fps

        while self.shared.is_running():
            t_start = time.perf_counter()

            if self.shared.is_paused():
                time.sleep(0.01)
                continue

            frame, _      = self.shared.read_frame()
            result, inf_ts = self.shared.read_result()
            m             = self.metrics.snapshot()

            if frame is None:
                time.sleep(0.005)
                continue

            render_ts  = time.perf_counter()
            total_lat  = (render_ts - inf_ts) * 1000 if inf_ts > 0 else 0

            # VRAM polling (infrequent)
            if render_ts - self._last_vram_poll > self._vram_poll_interval:
                vram = self._get_vram_used()
                self.metrics.update(vram_used_mb=vram)
                self._last_vram_poll = render_ts

            # CPU bottleneck detection
            cpu_bottleneck = m.capture_fps < m.inference_fps * 0.8
            self.metrics.update(
                total_latency_ms=total_lat,
                cpu_bottleneck=cpu_bottleneck,
            )

            # Mouse movement
            if result and result["gaze_screen_x"] is not None:
                pyautogui.moveTo(result["gaze_screen_x"], result["gaze_screen_y"])
                if result["blink"]:
                    if not self._click_held:
                        pyautogui.mouseDown()
                        self._click_held = True
                else:
                    if self._click_held:
                        pyautogui.mouseUp()
                        self._click_held = False

            # Draw and show frame
            display = self._draw_overlay(frame.copy(), result, m)
            cv2.imshow("GazeTrack — High-Performance Eye Tracker", display)

            # FPS tracking
            self._fps_counter.append(render_ts)
            if len(self._fps_counter) >= 2:
                elapsed = self._fps_counter[-1] - self._fps_counter[0]
                fps     = (len(self._fps_counter) - 1) / elapsed if elapsed > 0 else 0
                self.metrics.update(render_fps=fps)

            # Callback for GUI update
            if self.on_metrics_update:
                self.on_metrics_update(self.metrics.snapshot())

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                self.shared.stop()
                break

            elapsed_total = time.perf_counter() - t_start
            sleep_time    = interval - elapsed_total
            if sleep_time > 0:
                time.sleep(sleep_time)

        if self._click_held:
            pyautogui.mouseUp()
        cv2.destroyAllWindows()

# ─── Main Engine Orchestrator ─────────────────────────────────────────────────
class GazeTrackEngine:
    def __init__(self, tier_override: Optional[GPUTier] = None,
                 on_metrics_update: Optional[Callable] = None,
                 on_ready: Optional[Callable] = None):
        self.on_metrics_update = on_metrics_update
        self.on_ready          = on_ready

        # 1. Benchmark GPU
        self.gpu_info = benchmark_gpu()

        # 2. Apply tier
        self.tier   = tier_override or self.gpu_info.tier
        self.config = TIER_CONFIGS[self.tier]

        # 3. Shared state + metrics
        self.shared  = SharedState()
        self.metrics = MetricsStore()

        # 4. Build threads
        self.capture_thread   = CaptureThread(self.shared, self.metrics, self.config)
        self.inference_thread = InferenceThread(self.shared, self.metrics, self.config)
        self.render_thread    = RenderThread(
            self.shared, self.metrics, self.config, self.gpu_info,
            on_metrics_update=on_metrics_update,
        )

    def start(self):
        self.capture_thread.start()
        self.inference_thread.start()
        # Give capture a moment to fill the buffer
        time.sleep(0.15)
        self.render_thread.start()
        if self.on_ready:
            self.on_ready(self.gpu_info, self.tier, self.config)

    def stop(self):
        self.shared.stop()

    def pause(self):
        self.shared.set_paused(True)

    def resume(self):
        self.shared.set_paused(False)

    def wait(self):
        for t in [self.capture_thread, self.inference_thread, self.render_thread]:
            t.join(timeout=5.0)

    def switch_tier(self, new_tier: GPUTier):
        """Restart engine with a different tier."""
        self.stop()
        self.wait()
        self.__init__(
            tier_override=new_tier,
            on_metrics_update=self.on_metrics_update,
            on_ready=self.on_ready,
        )
        self.start()
