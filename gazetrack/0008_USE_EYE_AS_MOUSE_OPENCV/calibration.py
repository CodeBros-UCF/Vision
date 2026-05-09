"""
GazeTrack Calibration Engine
=============================
9-point gaze calibration that maps raw iris coordinates
to actual screen positions using polynomial regression.

Flow:
  1. Fullscreen dark window appears with 9 target dots (3x3 grid)
  2. Each dot lights up one at a time — user holds gaze on it
  3. A circular progress fill shows when enough samples are collected
  4. After all 9 points, a homography/polynomial map is computed
  5. The CalibrationMap is stored and used by InferenceThread to
     remap raw iris coords to corrected screen coords.
"""

import cv2
import numpy as np
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple


# ─── Calibration Data ─────────────────────────────────────────────────────────
@dataclass
class CalibrationMap:
    """Stores the polynomial coefficients mapping raw→screen gaze."""
    is_valid: bool = False
    # We store two degree-2 polynomial models: one for X, one for Y
    # Each is trained on [raw_x, raw_y, raw_x², raw_y², raw_x*raw_y] → screen_x/y
    coeff_x: Optional[np.ndarray] = None
    coeff_y: Optional[np.ndarray] = None
    # Fallback scale factors if polynomial fails
    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    def apply(self, raw_x: float, raw_y: float) -> Tuple[int, int]:
        """Map raw iris screen coord → calibrated screen coord."""
        if not self.is_valid or self.coeff_x is None:
            return int(raw_x), int(raw_y)
            
        feat = np.array([1, raw_x, raw_y], dtype=np.float64)
        cx = float(feat @ self.coeff_x)
        cy = float(feat @ self.coeff_y)
        
        if np.isnan(cx) or np.isnan(cy) or np.isinf(cx) or np.isinf(cy):
            return int(raw_x), int(raw_y)
            
        return int(cx), int(cy)


def build_calibration_map(
    raw_points: List[Tuple[float, float]],
    screen_points: List[Tuple[float, float]]
) -> CalibrationMap:
    """
    Fit a stable linear/affine mapping from raw gaze ratios to screen coords.
    Requires at least 4 point pairs.
    """
    cmap = CalibrationMap()
    if len(raw_points) < 4:
        return cmap

    raw    = np.array(raw_points,    dtype=np.float64)
    screen = np.array(screen_points, dtype=np.float64)

    # Build affine feature matrix [1, x, y]
    ones = np.ones((len(raw), 1))
    A = np.hstack([
        ones,
        raw[:, 0:1],
        raw[:, 1:2]
    ])  # shape: (N, 3)

    # Solve least-squares for X and Y independently
    try:
        result_x, _, _, _ = np.linalg.lstsq(A, screen[:, 0], rcond=None)
        result_y, _, _, _ = np.linalg.lstsq(A, screen[:, 1], rcond=None)
    except Exception:
        return cmap

    cmap.coeff_x  = result_x
    cmap.coeff_y  = result_y
    cmap.is_valid = True
    return cmap


# ─── Calibration Point Definition ─────────────────────────────────────────────
def get_calibration_points(screen_w: int, screen_h: int) -> List[Tuple[int, int]]:
    """Return 9 screen-space calibration target positions (3x3 grid)."""
    margin_x = int(screen_w * 0.08)
    margin_y = int(screen_h * 0.10)
    cx, cy   = screen_w // 2, screen_h // 2

    return [
        (margin_x,             margin_y),               # top-left
        (cx,                   margin_y),               # top-center
        (screen_w - margin_x,  margin_y),               # top-right
        (margin_x,             cy),                     # mid-left
        (cx,                   cy),                     # center
        (screen_w - margin_x,  cy),                     # mid-right
        (margin_x,             screen_h - margin_y),    # bottom-left
        (cx,                   screen_h - margin_y),    # bottom-center
        (screen_w - margin_x,  screen_h - margin_y),   # bottom-right
    ]


# ─── Calibration Window (runs in its own thread) ──────────────────────────────
class CalibrationWindow(threading.Thread):
    """
    Fullscreen OpenCV window that guides user through 9-point calibration.
    Communicates with the InferenceThread via SharedState to collect
    raw iris readings at each target position.
    """

    SAMPLES_NEEDED  = 40    # frames to average per point
    SAMPLE_DELAY    = 0.5   # seconds to wait before collecting (let gaze settle)
    POINT_HOLD_MS   = 2500  # total ms to show each point (including settle time)

    def __init__(
        self,
        screen_w: int, screen_h: int,
        get_raw_gaze_fn: Callable[[], Optional[Tuple[float, float]]],
        on_complete: Callable[[CalibrationMap], None],
        on_cancel:   Callable[[], None],
    ):
        super().__init__(daemon=True, name="CalibrationThread")
        self.screen_w        = screen_w
        self.screen_h        = screen_h
        self.get_raw_gaze    = get_raw_gaze_fn    # Returns (raw_x, raw_y) or None
        self.on_complete     = on_complete
        self.on_cancel       = on_cancel

        self._points         = get_calibration_points(screen_w, screen_h)
        self._raw_collected: List[Tuple[float, float]]    = []
        self._screen_targets: List[Tuple[float, float]]   = []

    def run(self):
        WIN = "GazeTrack — Calibration"
        cv2.namedWindow(WIN, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        total   = len(self._points)
        success = True

        # ── Intro screen (1.5 seconds) ────────────────────────────────────
        intro = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        cv2.putText(intro, "GAZE CALIBRATION",
                    (self.screen_w // 2 - 220, self.screen_h // 2 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(intro, "Look at each green dot and hold your gaze steady.",
                    (self.screen_w // 2 - 300, self.screen_h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 140, 160), 1, cv2.LINE_AA)
        cv2.putText(intro, "Press ESC at any time to skip calibration.",
                    (self.screen_w // 2 - 250, self.screen_h // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 120), 1, cv2.LINE_AA)
        cv2.imshow(WIN, intro)
        key = cv2.waitKey(1800)
        if key == 27:
            cv2.destroyWindow(WIN)
            self.on_cancel()
            return

        for idx, (tx, ty) in enumerate(self._points):
            # ── Shrink canvas to screen size
            canvas = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)

            # Draw instructions
            step_text = f"Point {idx + 1} / {total}   —   Look at the GREEN dot and hold still"
            cv2.putText(canvas, step_text,
                        (self.screen_w // 2 - 300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 180, 160), 1, cv2.LINE_AA)

            esc_text = "Press ESC to skip calibration"
            cv2.putText(canvas, esc_text,
                        (self.screen_w - 340, self.screen_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 100), 1, cv2.LINE_AA)

            # Draw all previous dots (dim green = done)
            for i, (px, py) in enumerate(self._points[:idx]):
                cv2.circle(canvas, (px, py), 10, (30, 60, 30), -1)
                cv2.circle(canvas, (px, py), 6, (40, 100, 40), -1)

            # Draw upcoming dots (very dim)
            for i, (px, py) in enumerate(self._points[idx + 1:]):
                cv2.circle(canvas, (px, py), 10, (25, 25, 40), -1)

            # ── Settle phase: show pulsing dot, don't collect yet
            t_start = time.perf_counter()
            settled = False
            samples: List[Tuple[float, float]] = []

            while True:
                elapsed = time.perf_counter() - t_start
                progress = min(elapsed / (self.POINT_HOLD_MS / 1000), 1.0)

                frame = canvas.copy()

                # Outer glow ring (pulses)
                glow_radius = 32 + int(4 * abs(np.sin(elapsed * 4)))
                cv2.circle(frame, (tx, ty), glow_radius, (40, 100, 60), 2)

                # Progress arc
                if elapsed > self.SAMPLE_DELAY:
                    sweep = int(360 * (elapsed - self.SAMPLE_DELAY) /
                                ((self.POINT_HOLD_MS / 1000) - self.SAMPLE_DELAY))
                    sweep = min(sweep, 360)
                    cv2.ellipse(frame, (tx, ty), (28, 28), -90, 0, sweep,
                                (99, 210, 130), 3)

                # Center dot — bright green when active
                dot_color = (80, 240, 80) if not settled else (100, 255, 150)
                cv2.circle(frame, (tx, ty), 12, dot_color, -1)
                cv2.circle(frame, (tx, ty), 13, (255, 255, 255), 1)

                # Show "collecting" text when sampling
                if settled:
                    cv2.putText(frame, f"Collecting... ({len(samples)} samples)",
                                (tx - 80, ty + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 180, 120), 1, cv2.LINE_AA)

                # Sample collection after settle delay
                if elapsed > self.SAMPLE_DELAY:
                    settled = True
                    raw = self.get_raw_gaze()
                    if raw is not None:
                        samples.append(raw)

                # Done with this point
                if elapsed >= (self.POINT_HOLD_MS / 1000):
                    break

                cv2.imshow(WIN, frame)
                key = cv2.waitKey(16)  # ~60 FPS render
                if key == 27:  # ESC
                    success = False
                    break

            if not success:
                break

            # Average collected samples for this point
            if len(samples) >= 5:
                avg_x = np.mean([s[0] for s in samples])
                avg_y = np.mean([s[1] for s in samples])
                self._raw_collected.append((avg_x, avg_y))
                self._screen_targets.append((float(tx), float(ty)))
            else:
                # Not enough gaze data — skip this point
                pass

            if not success:
                break

        # ── Completion screen ──────────────────────────────────────────────────
        if success and len(self._raw_collected) >= 6:
            done_screen = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
            cv2.putText(done_screen, "CALIBRATION COMPLETE",
                        (self.screen_w // 2 - 250, self.screen_h // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 255, 130), 2, cv2.LINE_AA)
            cv2.putText(done_screen, f"Successfully calibrated with {len(self._raw_collected)} points.",
                        (self.screen_w // 2 - 240, self.screen_h // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 160, 140), 1, cv2.LINE_AA)
            cv2.imshow(WIN, done_screen)
            cv2.waitKey(1200)

        cv2.destroyWindow(WIN)

        if success and len(self._raw_collected) >= 6:
            cmap = build_calibration_map(self._raw_collected, self._screen_targets)
            self.on_complete(cmap)
        else:
            self.on_cancel()
