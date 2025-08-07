\# Face Tracker



\## Overview

This simple Python project uses OpenCV to detect faces and eyes in real-time via webcam and controls the mouse cursor based on face position. It features configurable boundaries for mouse movement, smoothing to prevent jittery cursor, and a CLI interface for customization. The project is designed as a professional portfolio piece, demonstrating computer vision, hardware interaction (mouse control), and modular code structure.



\## Features

\- \*\*Face and Eye Detection\*\*: Uses Haar Cascade classifiers for robust detection.

\- \*\*Mouse Control\*\*: Moves cursor based on face position relative to screen boundaries.

\- \*\*Smoothing\*\*: Applies exponential smoothing to mouse movements for fluid control.

\- \*\*CLI Interface\*\*: Arguments for adjusting detection parameters, boundaries, and disabling video feed.

\- \*\*Error Handling\*\*: Checks for camera availability and handles detection failures gracefully.

\- \*\*Logging\*\*: Detailed logs for debugging and performance monitoring.

\- \*\*Cross-Platform\*\*: Works on Windows, macOS, and Linux (via pyautogui).



\## Requirements

\- Python 3.8+

\- Libraries: `opencv-python`, `pyautogui`



Install dependencies:

```bash

pip install opencv-python pyautogui





How to Run



Run with default settings:

bashpython face\_tracker.py



Custom boundaries and smoothing:

bashpython face\_tracker.py --boundary\_left 300 --boundary\_right 1000 --smoothing\_factor 0.7



Run without video display (headless mode):

bashpython face\_tracker.py --no\_video







Press 'q' to quit (if video is shown).

Mouse resets to (0,0) on exit.



Usage



Move your face left/right/up/down beyond the on-screen boundaries to control the mouse.

Eyes are detected and highlighted (up to 2) for verification.

Adjust --scale\_factor and --min\_neighbors for detection sensitivity.

Use --no\_video for background running (e.g., accessibility tool).



Example Output



Logs: "Starting face tracking. Press 'q' to quit."

Video feed shows face/eye rectangles and movement boundary.

Mouse moves smoothly based on face position.



Improvements and Future Work



Add hand gesture control using MediaPipe.

Implement calibration mode for user-specific boundaries.

Support multiple faces (e.g., select primary face).

Add GUI with Tkinter for real-time parameter tuning.

Unit tests with pytest for detection and mouse control functions.

