"""
GazeTrack Latency Benchmark Tool
=================================
Standalone latency profiler — runs the capture + inference pipeline and
outputs a detailed breakdown of where time is being spent.

Usage:
    python latency_test.py
    python latency_test.py --frames 300 --tier balanced
"""

import argparse
import time
import statistics
import threading
import sys
import os

# Add parent dir to path so we can import the engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gazetrack_engine import (
    GazeTrackEngine, GPUTier, TIER_CONFIGS, benchmark_gpu,
    SharedState, MetricsStore, CaptureThread, InferenceThread
)

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

SEPARATOR = "-" * 68

def parse_args():
    p = argparse.ArgumentParser(description="GazeTrack Latency Benchmark")
    p.add_argument("--frames",  type=int, default=200,
                   help="Number of inference frames to sample (default: 200)")
    p.add_argument("--tier",    choices=["lightweight", "balanced", "enhanced"],
                   default=None, help="Force performance tier")
    p.add_argument("--warmup",  type=int, default=30,
                   help="Frames to discard as warm-up (default: 30)")
    return p.parse_args()


class BenchmarkInferenceThread(InferenceThread):
    """Extends InferenceThread to capture per-frame latencies."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latencies_ms   = []          # cap→inf per frame
        self.frame_times_ms = []          # inference-only duration
        self.warmup_frames  = 30
        self._frame_idx     = 0
        self.done_event     = threading.Event()
        self._target_frames = 200

    def set_target(self, warmup: int, frames: int):
        self.warmup_frames  = warmup
        self._target_frames = frames

    def run(self):
        """
        Override to collect raw latency samples instead of sleeping.
        WARNING: This is a fragile override. It completely replaces InferenceThread.run()
        without calling super(). If gazetrack_engine.InferenceThread.__init__ or run()
        are modified (e.g., adding new state variables), this benchmark might break
        and need to be manually synchronized.
        """
        model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        import pyautogui
        screen_w, screen_h = pyautogui.size()
        last_frame_ts = -1.0

        while self.shared.is_running():
            if len(self.latencies_ms) >= self._target_frames:
                self.done_event.set()
                break

            frame, cap_ts = self.shared.read_frame()
            if frame is None or cap_ts == last_frame_ts:
                time.sleep(0.001)
                continue
            last_frame_ts = cap_ts

            t_inf_start = time.perf_counter()
            cap_latency = (t_inf_start - cap_ts) * 1000

            # Run inference
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection = face_landmarker.detect(mp_image)

            t_inf_end   = time.perf_counter()
            inf_duration = (t_inf_end - t_inf_start) * 1000

            self._frame_idx += 1
            if self._frame_idx > self.warmup_frames:
                self.latencies_ms.append(cap_latency)
                self.frame_times_ms.append(inf_duration)

            self.shared.write_result({}, t_inf_end)

        face_landmarker.close()


def run_benchmark(target_frames: int = 200, warmup: int = 30,
                  tier_str: str = None):
    tier_map = {
        "lightweight": GPUTier.LIGHTWEIGHT,
        "balanced":    GPUTier.BALANCED,
        "enhanced":    GPUTier.ENHANCED,
    }

    print(f"\n{SEPARATOR}")
    print("  GazeTrack Latency Benchmark")
    print(SEPARATOR)

    # GPU detection
    print("  Benchmarking GPU…", end="", flush=True)
    gpu = benchmark_gpu()
    tier = tier_map.get(tier_str) if tier_str else gpu.tier
    config = TIER_CONFIGS[tier]
    print(f" done.")
    print(f"  GPU    : {gpu.name}")
    print(f"  VRAM   : {gpu.vram_mb:,} MB")
    print(f"  Tier   : {config.name}")
    print(f"  Cam Res: {config.cam_width}×{config.cam_height}")
    print(f"  Target : {config.target_inference_fps} inference FPS")
    print(f"  Kalman : {'ON' if config.use_kalman else 'OFF'}")
    print(SEPARATOR)
    print(f"  Collecting {target_frames} frames (discarding {warmup} warm-up frames)…")
    print(f"  Press ESC in the preview window to abort.\n")

    shared  = SharedState()
    metrics = MetricsStore()

    capture   = CaptureThread(shared, metrics, config)
    inference = BenchmarkInferenceThread(shared, metrics, config)
    inference.set_target(warmup, target_frames)

    capture.start()
    time.sleep(0.2)  # Let capture fill
    inference.start()

    # Show a live preview while collecting
    frame_count = 0
    t_bench_start = time.time()
    while not inference.done_event.is_set():
        frame, _ = shared.read_frame()
        if frame is not None:
            sampled = len(inference.latencies_ms)
            pct     = sampled / target_frames * 100
            display = frame.copy()
            bar_w   = int(display.shape[1] * pct / 100)
            cv2.rectangle(display, (0, 0), (bar_w, 8), (99, 102, 241), -1)
            cv2.putText(display, f"Benchmarking…  {sampled}/{target_frames} frames ({pct:.0f}%)",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("GazeTrack — Benchmark", display)
        key = cv2.waitKey(1)
        if key == 27:
            shared.stop()
            break
        time.sleep(0.01)

    cv2.destroyAllWindows()
    shared.stop()
    t_bench_total = time.time() - t_bench_start

    lats  = inference.latencies_ms
    infs  = inference.frame_times_ms

    if len(lats) < 10:
        print("  Not enough data collected — ensure webcam is connected and try again.")
        return

    print(f"\n{SEPARATOR}")
    print("  RESULTS")
    print(SEPARATOR)

    def stat_row(label, data, unit="ms"):
        if not data:
            return
        avg  = statistics.mean(data)
        minn = min(data)
        maxx = max(data)
        p50  = statistics.median(data)
        p95  = sorted(data)[int(len(data) * 0.95)]
        print(f"  {label:<30} avg={avg:6.2f}{unit}  min={minn:6.2f}{unit}  "
              f"p50={p50:6.2f}{unit}  p95={p95:6.2f}{unit}  max={maxx:6.2f}{unit}")

    stat_row("Capture -> Inference Latency", lats)
    stat_row("Inference Duration (Mediapipe)", infs)
    combined = [l + i for l, i in zip(lats, infs)]
    stat_row("Total (Capture + Inference)", combined)

    avg_inf_fps = 1000 / statistics.mean(infs) if statistics.mean(infs) > 0 else 0
    print(f"\n  Effective Inference FPS      : {avg_inf_fps:.1f}")
    print(f"  Total benchmark duration     : {t_bench_total:.1f}s")
    print(f"  Frames sampled               : {len(lats)}")

    # Bottleneck analysis
    avg_cap_lat = statistics.mean(lats)
    avg_inf_dur = statistics.mean(infs)
    print(f"\n{SEPARATOR}")
    print("  BOTTLENECK ANALYSIS")
    print(SEPARATOR)
    if avg_cap_lat > avg_inf_dur * 0.5:
        print("  ⚠  Camera I/O is a significant bottleneck.")
        print("     → Consider lowering capture resolution or using CAP_DSHOW.")
    if avg_inf_dur > 30:
        print("  ⚠  Mediapipe inference is slow (>30ms per frame).")
        print("     → GPU acceleration not active or hardware is limited.")
    if avg_inf_dur < 10 and avg_cap_lat < 5:
        print("  ✓  Pipeline is well-optimized. Low bottlenecks detected.")
    print(SEPARATOR)
    print()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        target_frames=args.frames,
        warmup=args.warmup,
        tier_str=args.tier,
    )
