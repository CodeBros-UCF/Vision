"""
GazeTrack Professional Dashboard
=================================
A modern, dark-mode control panel built with customtkinter.
Displays real-time latency, FPS, VRAM, and GPU tier information.
"""

import customtkinter as ctk
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from typing import Optional

from gazetrack_engine import (
    GazeTrackEngine, GPUTier, GPUInfo, TierConfig, Metrics,
    TIER_CONFIGS, benchmark_gpu
)

# ─── Theme ────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

TIER_COLORS = {
    GPUTier.LIGHTWEIGHT: "#f59e0b",
    GPUTier.BALANCED:    "#06b6d4",
    GPUTier.ENHANCED:    "#a855f7",
}

TIER_ICONS = {
    GPUTier.LIGHTWEIGHT: "⚡",
    GPUTier.BALANCED:    "🎯",
    GPUTier.ENHANCED:    "🚀",
}

BG_DARK    = "#0a0a0f"
BG_CARD    = "#111118"
BG_CARD2   = "#16161f"
ACCENT     = "#6366f1"
TEXT_DIM   = "#6b7280"
TEXT_MAIN  = "#e2e8f0"
GREEN_MET  = "#10b981"
RED_MET    = "#ef4444"
YELLOW_MET = "#f59e0b"

# ─── Animated Metric Card ─────────────────────────────────────────────────────
class MetricCard(ctk.CTkFrame):
    def __init__(self, parent, title: str, unit: str = "", width: int = 180,
                 accent_color: str = ACCENT, **kwargs):
        super().__init__(parent, width=width, height=110,
                         fg_color=BG_CARD, corner_radius=16, **kwargs)
        self.grid_propagate(False)
        self._accent = accent_color
        self._unit   = unit

        # Subtle accent top-border line
        self._border = ctk.CTkFrame(self, height=3, fg_color=accent_color, corner_radius=2)
        self._border.pack(fill="x", padx=0, pady=(0, 0))

        self._title_lbl = ctk.CTkLabel(
            self, text=title.upper(), text_color=TEXT_DIM,
            font=ctk.CTkFont("Inter", 10, weight="normal")
        )
        self._title_lbl.pack(anchor="w", padx=14, pady=(10, 0))

        self._value_var = tk.StringVar(value="—")
        self._value_lbl = ctk.CTkLabel(
            self, textvariable=self._value_var,
            text_color=TEXT_MAIN,
            font=ctk.CTkFont("Inter", 30, weight="bold")
        )
        self._value_lbl.pack(anchor="w", padx=14, pady=(0, 2))

        self._unit_lbl = ctk.CTkLabel(
            self, text=unit, text_color=TEXT_DIM,
            font=ctk.CTkFont("Inter", 11)
        )
        self._unit_lbl.pack(anchor="w", padx=14)

    def set_value(self, val: str, color: Optional[str] = None):
        self._value_var.set(val)
        if color:
            self._value_lbl.configure(text_color=color)

# ─── Mini Latency Bar ─────────────────────────────────────────────────────────
class LatencyBar(ctk.CTkFrame):
    """Visual bar showing latency from 0→max_ms."""
    def __init__(self, parent, label: str, max_ms: float = 50.0, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self._max   = max_ms
        self._label = label

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x")
        ctk.CTkLabel(top, text=label, text_color=TEXT_DIM,
                     font=ctk.CTkFont("Inter", 11)).pack(side="left")
        self._val_lbl = ctk.CTkLabel(top, text="0 ms", text_color=TEXT_MAIN,
                                      font=ctk.CTkFont("Inter", 11, weight="bold"))
        self._val_lbl.pack(side="right")

        self._bar = ctk.CTkProgressBar(self, height=6, corner_radius=4,
                                        fg_color="#1e1e2e", progress_color=ACCENT)
        self._bar.set(0)
        self._bar.pack(fill="x", pady=(3, 0))

    def set_ms(self, val: float):
        pct   = min(val / self._max, 1.0)
        color = GREEN_MET if val < self._max * 0.3 else \
                YELLOW_MET if val < self._max * 0.7 else RED_MET
        self._bar.configure(progress_color=color)
        self._bar.set(pct)
        self._val_lbl.configure(text=f"{val:.1f} ms")

# ─── Tier Badge ───────────────────────────────────────────────────────────────
class TierBadge(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=BG_CARD, corner_radius=20, **kwargs)
        self._icon  = ctk.CTkLabel(self, text="⚡", font=ctk.CTkFont("Segoe UI Emoji", 32))
        self._icon.pack(pady=(16, 4))
        self._name  = ctk.CTkLabel(self, text="Detecting…", text_color=TEXT_MAIN,
                                    font=ctk.CTkFont("Inter", 15, weight="bold"))
        self._name.pack()
        self._desc  = ctk.CTkLabel(self, text="", text_color=TEXT_DIM,
                                    font=ctk.CTkFont("Inter", 10), wraplength=220,
                                    justify="center")
        self._desc.pack(pady=(4, 12), padx=16)

    def set_tier(self, tier: GPUTier, config: TierConfig):
        self._icon.configure(text=TIER_ICONS[tier])
        self._name.configure(text=config.name, text_color=TIER_COLORS[tier])
        self._desc.configure(text=config.description)

# ─── Main Application Window ──────────────────────────────────────────────────
class GazeTrackApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("GazeTrack  —  Adaptive Eye Control")
        self.geometry("980x680")
        self.minsize(900, 620)
        self.configure(fg_color=BG_DARK)

        self._engine: Optional[GazeTrackEngine] = None
        self._running = False
        self._gpu_info: Optional[GPUInfo] = None
        self._selected_tier: Optional[GPUTier] = None
        self._build_ui()

        # Run GPU benchmark in background so UI appears instantly
        threading.Thread(target=self._do_gpu_benchmark, daemon=True).start()

    # ── UI Construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header bar ───────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color=BG_CARD, height=64, corner_radius=0)
        header.pack(fill="x")
        header.pack_propagate(False)

        logo_frame = ctk.CTkFrame(header, fg_color="transparent")
        logo_frame.pack(side="left", padx=24, pady=12)
        ctk.CTkLabel(logo_frame, text="👁", font=ctk.CTkFont("Segoe UI Emoji", 26)
                     ).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(logo_frame, text="GazeTrack",
                     font=ctk.CTkFont("Inter", 20, weight="bold"),
                     text_color=TEXT_MAIN).pack(side="left")
        ctk.CTkLabel(logo_frame, text="  Adaptive Eye Control System",
                     font=ctk.CTkFont("Inter", 11),
                     text_color=TEXT_DIM).pack(side="left", pady=(6, 0))

        # Status dot
        self._status_dot = ctk.CTkLabel(header, text="● OFFLINE",
                                         font=ctk.CTkFont("Inter", 11, weight="bold"),
                                         text_color=RED_MET)
        self._status_dot.pack(side="right", padx=24)

        # ── Body: Left sidebar | Right main ──────────────────────────────────
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20, pady=16)

        # LEFT SIDEBAR (220px)
        sidebar = ctk.CTkFrame(body, width=240, fg_color="transparent")
        sidebar.pack(side="left", fill="y", padx=(0, 16))
        sidebar.pack_propagate(False)

        # Tier badge
        self._tier_badge = TierBadge(sidebar)
        self._tier_badge.pack(fill="x", pady=(0, 12))

        # GPU Info card
        gpu_card = ctk.CTkFrame(sidebar, fg_color=BG_CARD, corner_radius=16)
        gpu_card.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(gpu_card, text="GPU HARDWARE", text_color=TEXT_DIM,
                     font=ctk.CTkFont("Inter", 10)).pack(anchor="w", padx=14, pady=(12, 4))
        self._gpu_name_lbl = ctk.CTkLabel(gpu_card, text="Detecting…",
                                           text_color=TEXT_MAIN,
                                           font=ctk.CTkFont("Inter", 12, weight="bold"),
                                           wraplength=200, justify="left")
        self._gpu_name_lbl.pack(anchor="w", padx=14)
        self._vram_lbl = ctk.CTkLabel(gpu_card, text="VRAM: —",
                                       text_color=TEXT_DIM,
                                       font=ctk.CTkFont("Inter", 11))
        self._vram_lbl.pack(anchor="w", padx=14, pady=(2, 12))

        # Manual Tier Override
        override_card = ctk.CTkFrame(sidebar, fg_color=BG_CARD, corner_radius=16)
        override_card.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(override_card, text="MANUAL OVERRIDE", text_color=TEXT_DIM,
                     font=ctk.CTkFont("Inter", 10)).pack(anchor="w", padx=14, pady=(12, 6))

        self._tier_var = ctk.StringVar(value="Auto")
        tier_options   = ["Auto", "Lightweight", "Balanced", "Enhanced"]
        seg = ctk.CTkSegmentedButton(override_card, values=tier_options,
                                      variable=self._tier_var,
                                      command=self._on_tier_changed,
                                      fg_color="#1e1e2e",
                                      selected_color=ACCENT,
                                      selected_hover_color="#4f52d4",
                                      font=ctk.CTkFont("Inter", 11))
        seg.pack(fill="x", padx=14, pady=(0, 12))

        # CPU Bottleneck warning
        self._bottleneck_lbl = ctk.CTkLabel(sidebar, text="", text_color=YELLOW_MET,
                                             font=ctk.CTkFont("Inter", 10),
                                             wraplength=210, justify="left")
        self._bottleneck_lbl.pack(anchor="w")

        # ── Main panel ───────────────────────────────────────────────────────
        main = ctk.CTkFrame(body, fg_color="transparent")
        main.pack(side="left", fill="both", expand=True)

        # ── Metrics row ───────────────────────────────────────────────────────
        metrics_row = ctk.CTkFrame(main, fg_color="transparent")
        metrics_row.pack(fill="x", pady=(0, 12))
        metrics_row.columnconfigure([0,1,2,3], weight=1, uniform="met")

        self._card_cap = MetricCard(metrics_row, "Capture FPS", "fps",
                                     accent_color=GREEN_MET)
        self._card_cap.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self._card_inf = MetricCard(metrics_row, "Inference FPS", "fps",
                                     accent_color=ACCENT)
        self._card_inf.grid(row=0, column=1, sticky="nsew", padx=(0, 8))

        self._card_ren = MetricCard(metrics_row, "Render FPS", "fps",
                                     accent_color="#06b6d4")
        self._card_ren.grid(row=0, column=2, sticky="nsew", padx=(0, 8))

        self._card_vram = MetricCard(metrics_row, "VRAM Used", "MB",
                                      accent_color="#a855f7")
        self._card_vram.grid(row=0, column=3, sticky="nsew")

        # ── Latency panel ─────────────────────────────────────────────────────
        lat_frame = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=16)
        lat_frame.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(lat_frame, text="LATENCY BREAKDOWN", text_color=TEXT_DIM,
                     font=ctk.CTkFont("Inter", 10)).pack(anchor="w", padx=18, pady=(14, 6))

        lat_inner = ctk.CTkFrame(lat_frame, fg_color="transparent")
        lat_inner.pack(fill="x", padx=18, pady=(0, 14))
        lat_inner.columnconfigure([0,1], weight=1)

        self._lat_cap = LatencyBar(lat_inner, "Capture → Inference", max_ms=50)
        self._lat_cap.grid(row=0, column=0, sticky="ew", padx=(0, 20), pady=4)

        self._lat_tot = LatencyBar(lat_inner, "Total Pipeline Latency", max_ms=100)
        self._lat_tot.grid(row=0, column=1, sticky="ew", pady=4)

        # ── Gaze activity panel ───────────────────────────────────────────────
        gaze_frame = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=16)
        gaze_frame.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(gaze_frame, text="GAZE ACTIVITY", text_color=TEXT_DIM,
                     font=ctk.CTkFont("Inter", 10)).pack(anchor="w", padx=18, pady=(14, 6))

        gaze_inner = ctk.CTkFrame(gaze_frame, fg_color="transparent")
        gaze_inner.pack(fill="x", padx=18, pady=(0, 14))

        self._gaze_xy = ctk.CTkLabel(gaze_inner, text="Gaze: (—, —)",
                                      text_color=TEXT_MAIN,
                                      font=ctk.CTkFont("Inter", 13))
        self._gaze_xy.pack(side="left")

        self._blink_lbl = ctk.CTkLabel(gaze_inner, text="", text_color=RED_MET,
                                        font=ctk.CTkFont("Inter", 13, weight="bold"))
        self._blink_lbl.pack(side="left", padx=24)

        # ── Calibration panel ─────────────────────────────────────────────────
        cal_frame = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=16)
        cal_frame.pack(fill="x", pady=(0, 12))

        cal_header = ctk.CTkFrame(cal_frame, fg_color="transparent")
        cal_header.pack(fill="x", padx=18, pady=(14, 6))
        ctk.CTkLabel(cal_header, text="GAZE CALIBRATION", text_color=TEXT_DIM,
                     font=ctk.CTkFont("Inter", 10)).pack(side="left")

        self._cal_badge = ctk.CTkLabel(cal_header, text="● NOT CALIBRATED",
                                        text_color=YELLOW_MET,
                                        font=ctk.CTkFont("Inter", 10, weight="bold"))
        self._cal_badge.pack(side="right")

        cal_body = ctk.CTkFrame(cal_frame, fg_color="transparent")
        cal_body.pack(fill="x", padx=18, pady=(0, 8))

        ctk.CTkLabel(cal_body,
                     text="Run 9-point calibration to map your gaze to screen coordinates accurately.",
                     text_color=TEXT_DIM,
                     font=ctk.CTkFont("Inter", 11),
                     wraplength=400, justify="left").pack(side="left", fill="x", expand=True)

        self._cal_btn = ctk.CTkButton(
            cal_body, text="⊕  Calibrate",
            font=ctk.CTkFont("Inter", 12, weight="bold"),
            fg_color="#1e1e2e", hover_color="#2d2d3e",
            border_width=1, border_color=ACCENT,
            text_color=ACCENT,
            height=36, width=130, corner_radius=10,
            state="disabled",
            command=self._on_calibrate
        )
        self._cal_btn.pack(side="right", padx=(12, 0))

        # Calibrate-on-start checkbox
        cal_opt = ctk.CTkFrame(cal_frame, fg_color="transparent")
        cal_opt.pack(fill="x", padx=18, pady=(0, 14))

        self._cal_on_start_var = ctk.BooleanVar(value=True)
        self._cal_on_start_cb = ctk.CTkCheckBox(
            cal_opt, text="Calibrate automatically when tracking starts",
            variable=self._cal_on_start_var,
            font=ctk.CTkFont("Inter", 11),
            text_color=TEXT_DIM,
            fg_color=ACCENT, hover_color="#4f52d4",
            border_color="#3d3d5e",
            corner_radius=4, height=24,
        )
        self._cal_on_start_cb.pack(side="left")

        # ── Config summary ────────────────────────────────────────────────────
        self._config_frame = ctk.CTkFrame(main, fg_color=BG_CARD2, corner_radius=16)
        self._config_frame.pack(fill="x", pady=(0, 16))
        ctk.CTkLabel(self._config_frame, text="ACTIVE PIPELINE CONFIG",
                     text_color=TEXT_DIM,
                     font=ctk.CTkFont("Inter", 10)).pack(anchor="w", padx=18, pady=(12, 6))
        self._config_lbl = ctk.CTkLabel(self._config_frame, text="Awaiting benchmark…",
                                         text_color=TEXT_DIM,
                                         font=ctk.CTkFont("Inter", 11),
                                         justify="left")
        self._config_lbl.pack(anchor="w", padx=18, pady=(0, 12))

        # ── Control buttons ───────────────────────────────────────────────────
        btn_row = ctk.CTkFrame(main, fg_color="transparent")
        btn_row.pack(fill="x")

        self._start_btn = ctk.CTkButton(
            btn_row, text="▶  Start Tracking",
            font=ctk.CTkFont("Inter", 14, weight="bold"),
            fg_color=ACCENT, hover_color="#4f52d4",
            height=48, corner_radius=12,
            command=self._on_start
        )
        self._start_btn.pack(side="left", padx=(0, 10))

        self._stop_btn = ctk.CTkButton(
            btn_row, text="■  Stop",
            font=ctk.CTkFont("Inter", 14),
            fg_color="#1e1e2e", hover_color="#2d2d3e",
            border_width=1, border_color="#3d3d5e",
            height=48, corner_radius=12,
            state="disabled",
            command=self._on_stop
        )
        self._stop_btn.pack(side="left", padx=(0, 10))

        self._pause_btn = ctk.CTkButton(
            btn_row, text="⏸  Pause",
            font=ctk.CTkFont("Inter", 14),
            fg_color="#1e1e2e", hover_color="#2d2d3e",
            border_width=1, border_color="#3d3d5e",
            height=48, corner_radius=12,
            state="disabled",
            command=self._on_pause
        )
        self._pause_btn.pack(side="left")

    # ── GPU Benchmark (background) ────────────────────────────────────────────
    def _do_gpu_benchmark(self):
        info = benchmark_gpu()
        self._gpu_info = info
        self.after(0, self._on_benchmark_done, info)

    def _on_benchmark_done(self, info: GPUInfo):
        self._gpu_name_lbl.configure(text=info.name)
        vram_str = f"VRAM: {info.vram_mb:,} MB" if info.vram_mb else "VRAM: N/A"
        self._vram_lbl.configure(text=vram_str)
        cfg = TIER_CONFIGS[info.tier]
        self._tier_badge.set_tier(info.tier, cfg)
        self._selected_tier = info.tier
        self._update_config_label(info.tier)

    def _update_config_label(self, tier: GPUTier):
        cfg  = TIER_CONFIGS[tier]
        text = (
            f"Resolution: {cfg.cam_width}×{cfg.cam_height}  |  "
            f"Inference target: {cfg.target_inference_fps} FPS  |  "
            f"Render target: {cfg.display_fps} FPS  |  "
            f"Kalman filter: {'ON' if cfg.use_kalman else 'OFF'}  |  "
            f"Prediction: {'ON' if cfg.use_prediction else 'OFF'}"
        )
        self._config_lbl.configure(text=text, text_color=TIER_COLORS[tier])

    # ── Tier Override ─────────────────────────────────────────────────────────
    def _on_tier_changed(self, val: str):
        mapping = {
            "Lightweight": GPUTier.LIGHTWEIGHT,
            "Balanced":    GPUTier.BALANCED,
            "Enhanced":    GPUTier.ENHANCED,
        }
        if val == "Auto" and self._gpu_info:
            self._selected_tier = self._gpu_info.tier
        else:
            self._selected_tier = mapping.get(val, self._gpu_info.tier if self._gpu_info else GPUTier.LIGHTWEIGHT)

        cfg = TIER_CONFIGS[self._selected_tier]
        self._tier_badge.set_tier(self._selected_tier, cfg)
        self._update_config_label(self._selected_tier)

        # If already running, hot-switch
        if self._running and self._engine:
            self._engine.switch_tier(self._selected_tier)

    # ── Controls ──────────────────────────────────────────────────────────────
    def _on_start(self):
        if self._running:
            return
        tier = self._selected_tier or (self._gpu_info.tier if self._gpu_info else GPUTier.LIGHTWEIGHT)
        self._engine = GazeTrackEngine(
            tier_override=tier,
            on_metrics_update=self._on_metrics,
            on_ready=self._on_engine_ready,
        )
        self._engine.start()
        self._running = True
        self._status_dot.configure(text="● LIVE", text_color=GREEN_MET)
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._pause_btn.configure(state="normal")
        self._cal_btn.configure(state="normal")

        # Auto-calibrate if checkbox is checked
        if self._cal_on_start_var.get():
            # Give inference thread a moment to start producing gaze data
            self.after(800, self._on_calibrate)

    def _on_stop(self):
        if self._engine:
            self._engine.stop()
        self._running = False
        self._engine  = None
        self._status_dot.configure(text="● OFFLINE", text_color=RED_MET)
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._pause_btn.configure(state="disabled", text="⏸  Pause")
        self._cal_btn.configure(state="disabled")
        self._cal_badge.configure(text="● NOT CALIBRATED", text_color=YELLOW_MET)

    def _on_pause(self):
        if not self._engine:
            return
        if self._pause_btn.cget("text").startswith("⏸"):
            self._engine.pause()
            self._pause_btn.configure(text="▶  Resume")
            self._status_dot.configure(text="● PAUSED", text_color=YELLOW_MET)
        else:
            self._engine.resume()
            self._pause_btn.configure(text="⏸  Pause")
            self._status_dot.configure(text="● LIVE", text_color=GREEN_MET)

    def _on_calibrate(self):
        if not self._engine or not self._running:
            return
        self._cal_btn.configure(state="disabled", text="⏳  Calibrating…")
        self._cal_badge.configure(text="● CALIBRATING", text_color=YELLOW_MET)

        def on_complete(cmap):
            self.after(0, self._on_calibration_done, cmap)

        def on_cancel():
            self.after(0, self._on_calibration_cancelled)

        self._engine.start_calibration(
            on_complete=on_complete,
            on_cancel=on_cancel,
        )

    def _on_calibration_done(self, cmap):
        if cmap.is_valid:
            self._cal_badge.configure(text="● CALIBRATED ✓", text_color=GREEN_MET)
        else:
            self._cal_badge.configure(text="● CALIBRATION FAILED", text_color=RED_MET)
        self._cal_btn.configure(state="normal", text="↺  Re-Calibrate")

    def _on_calibration_cancelled(self):
        self._cal_badge.configure(text="● NOT CALIBRATED", text_color=YELLOW_MET)
        self._cal_btn.configure(state="normal", text="⊕  Calibrate")

    def _on_engine_ready(self, gpu_info: GPUInfo, tier: GPUTier, cfg: TierConfig):
        self.after(0, lambda: self._tier_badge.set_tier(tier, cfg))
        self.after(0, lambda: self._update_config_label(tier))

    # ── Metrics Update (called from RenderThread) ─────────────────────────────
    def _on_metrics(self, m: Metrics):
        """Schedule a GUI update on the main thread."""
        self.after(0, self._apply_metrics, m)

    def _apply_metrics(self, m: Metrics):
        def fps_color(v):
            if v >= 55: return GREEN_MET
            if v >= 25: return YELLOW_MET
            return RED_MET

        self._card_cap.set_value(f"{m.capture_fps:.0f}", fps_color(m.capture_fps))
        self._card_inf.set_value(f"{m.inference_fps:.0f}", fps_color(m.inference_fps))
        self._card_ren.set_value(f"{m.render_fps:.0f}", fps_color(m.render_fps))
        self._card_vram.set_value(f"{m.vram_used_mb:,}" if m.vram_used_mb else "—")

        self._lat_cap.set_ms(m.cap_to_inf_latency_ms)
        self._lat_tot.set_ms(m.total_latency_ms)

        self._gaze_xy.configure(text=f"Gaze: ({m.gaze_x}, {m.gaze_y})")
        if m.blink_detected:
            self._blink_lbl.configure(text="● BLINK CLICK")
        else:
            self._blink_lbl.configure(text="")

        if m.cpu_bottleneck:
            self._bottleneck_lbl.configure(
                text="⚠ CPU bottleneck detected — capture is significantly slower than inference"
            )
        else:
            self._bottleneck_lbl.configure(text="")

    def on_closing(self):
        if self._engine:
            self._engine.stop()
        self.destroy()


def run_gui():
    app = GazeTrackApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    run_gui()
