# Vision: Advanced Gaze Tracking & Eye-Controlled Canvas

![Vision Banner](https://github.com/pranavsaigandikota/Vision/blob/main/banner.jpg)

## Overview
Vision is a high-performance, multi-threaded eye-controlled mouse and drawing application. It leverages Mediapipe's Face Mesh for precise iris tracking and direct Win32 API calls for ultra-low latency cursor control.

## Key Features
*   **High-Performance Engine**: Utilizes a three-threaded architecture (Capture, Inference, Render) to ensure smooth interaction and zero-lag feedback.
*   **Ultra-Low Latency Cursor Control**: Bypasses traditional automation libraries like `pyautogui` in favor of direct Windows API `SetCursorPos` calls, reducing movement overhead to ~0.1ms.
*   **Adaptive GPU Tiers**:
    *   **Lightweight Mode**: Optimized for CPU/Integrated graphics (640x480).
    *   **Balanced Mode**: HD tracking with Kalman filtering for jitter reduction (1280x720).
    *   **Enhanced AI Mode**: Full HD precision with predictive gaze modeling for high-end systems (1920x1080).
*   **Intelligent Interaction Logic**:
    *   **Single Blink**: Standard mouse click for quick actions.
    *   **Double Blink**: Toggles **"Click-and-Drag"** mode, allowing users to draw or move windows seamlessly.
*   **9-Point Calibration**: Comprehensive fullscreen calibration using polynomial mapping to adjust for individual eye geometry and display characteristics.
*   **Live Latency Profiling**: Real-time dashboard showing Capture-to-Inference and End-to-End latency metrics.

## Latency Calculation
We achieve ultra-low latency by profiling every stage of the vision pipeline:
1.  **Capture Timestamp**: Recorded immediately when the frame is grabbed from the camera buffer.
2.  **Inference Latency**: Measured from frame arrival to the completion of the 478-point face mesh.
3.  **End-to-End Latency**: The total time from the physical eye movement to the Win32 cursor update.
*All metrics are displayed in real-time on the GazeTrack dashboard.*

## How to Run
1.  Install dependencies: `pip install mediapipe opencv-python numpy pyautogui customtkinter`
2.  Run the dashboard: `python main.py`
3.  Click **"Start Tracking"** and complete the **9-point calibration**.
4.  Launch your favorite drawing app (e.g., MS Paint) and use your eyes to create!

## Team
**Made with ❤️ by:**  
Pranavsai Gandikota, Arwa Arshad Ali, David Navarrete, Peter-Karl Jackson

---
**Repository:** [https://github.com/CodeBros-UCF/Vision](https://github.com/CodeBros-UCF/Vision)
