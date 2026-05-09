"""
GazeTrack — Entry Point
========================
Run this file to launch the dashboard:
    python main.py

Or to run headless (no GUI, just the OpenCV window):
    python main.py --headless

Or to force a specific tier:
    python main.py --tier lightweight
    python main.py --tier balanced
    python main.py --tier enhanced
"""

import argparse
import sys

def parse_args():
    p = argparse.ArgumentParser(
        description="GazeTrack Adaptive Eye Control System"
    )
    p.add_argument(
        "--headless", action="store_true",
        help="Run without the GUI dashboard (OpenCV window only)"
    )
    p.add_argument(
        "--tier", choices=["lightweight", "balanced", "enhanced"],
        default=None,
        help="Force a specific performance tier (default: auto-detect)"
    )
    return p.parse_args()


def run_headless(tier_str):
    from gazetrack_engine import GazeTrackEngine, GPUTier, TIER_CONFIGS

    tier_map = {
        "lightweight": GPUTier.LIGHTWEIGHT,
        "balanced":    GPUTier.BALANCED,
        "enhanced":    GPUTier.ENHANCED,
    }
    tier_override = tier_map.get(tier_str) if tier_str else None

    def on_ready(gpu_info, tier, cfg):
        print(f"\n{'='*60}")
        print(f"  GazeTrack — Headless Mode")
        print(f"{'='*60}")
        print(f"  GPU    : {gpu_info.name}")
        print(f"  VRAM   : {gpu_info.vram_mb:,} MB")
        print(f"  Tier   : {cfg.name}")
        print(f"  Res    : {cfg.cam_width}×{cfg.cam_height}")
        print(f"  Inf FPS: {cfg.target_inference_fps}")
        print(f"  Kalman : {'ON' if cfg.use_kalman else 'OFF'}")
        print(f"{'='*60}\n")
        print("Press ESC in the OpenCV window to quit.\n")

    engine = GazeTrackEngine(
        tier_override=tier_override,
        on_ready=on_ready,
    )
    engine.start()
    engine.wait()


if __name__ == "__main__":
    args = parse_args()

    if args.headless:
        run_headless(args.tier)
    else:
        from gui import run_gui
        run_gui()
