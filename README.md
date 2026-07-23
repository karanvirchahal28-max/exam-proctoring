# Head Pose Intelligence Engine — AI Exam Proctoring System

An AI-powered real-time proctoring system that detects suspicious head movement during exams using computer vision and adaptive statistical modeling — not hardcoded thresholds.

## Features
- **Kalman Filtering** — smooths noisy head-yaw signal frame-to-frame, extracts velocity as a free byproduct
- **Adaptive Calibration** — learns each student's natural head position using Welford's online algorithm instead of a fixed threshold
- **Kinematic Analysis** — rolling-window velocity & acceleration tracking to distinguish quick glances from deliberate turns
- **Temporal Pattern Detection** — flags based on frequency, duration, and away-time ratio, not single events
- **Session Memory Scoring** — cumulative suspicion score that persists across the exam, so past behavior isn't forgotten
- **Live Dashboard** — real-time OpenCV overlay showing suspicion score, signal breakdown, and teacher verdict

## Tech Stack
Python · OpenCV · MediaPipe FaceMesh · NumPy

## How It Works
1. Detects facial landmarks via MediaPipe and extracts yaw angle using `solvePnP`
2. Smooths the signal with a 1D Kalman filter (position + velocity)
3. Calibrates a personalized baseline per student (150-frame warmup)
4. Computes 5 weighted signals — position anomaly, angular velocity, event frequency, away-time ratio, and session memory
5. Fuses them into a single 0–100% suspicion score with EMA smoothing
6. Renders a live dashboard with a teacher-facing verdict (Normal / Monitor / Alert / Flagged)

## Demo
*(add a screenshot or short GIF of the dashboard here)*

## Setup
\`\`\`bash
pip install opencv-python mediapipe numpy
python module_headpose_intelligence.py
\`\`\`

## Project Structure
\`\`\`
module_headpose_intelligence.py   # Main engine + dashboard
\`\`\`

## Author
Karanvir Singh — [LinkedIn](https://linkedin.com/in/karanvir28) · [GitHub](https://github.com/karanvirchahal28-max)
