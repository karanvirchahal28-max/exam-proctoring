"""
╔══════════════════════════════════════════════════════════════════════════╗
║     HEAD POSE INTELLIGENCE ENGINE  — Capstone CPG 127                    ║
║     AI-Based Classroom Exam Proctoring System                            ║
║                                                                          ║
║     Module: module_headpose_intelligence.py  |  Version 5.0              ║
║                                                                          ║
║     Architecture:                                                        ║
║       KalmanFilter1D         → Smooth noisy yaw signal                   ║
║       CalibrationEngine      → Welford online statistics                 ║
║       KinematicAnalyzer      → Rolling-window velocity & acceleration    ║
║       TemporalPatternEngine  → Event frequency, away-ratio, duration     ║
║       SuspicionFusion        → Weighted multi-signal EMA scoring         ║
╚══════════════════════════════════════════════════════════════════════════╝

Why these design choices:
- Kalman filter: Raw MediaPipe yaw is noisy frame-to-frame. A 1D Kalman
  gives us smooth position AND velocity as a free byproduct.
- Welford algorithm: Numerically stable, single-pass running mean/variance.
  No need to store all samples. Correct for online calibration.
- Adaptive threshold: threshold = k * std_dev, NOT a hardcoded number.
  A student who naturally tilts 5° has a different "normal" than one who
  sits perfectly straight.
- Adaptive baseline drift: EMA update ONLY when z-score is low (student
  is clearly looking forward). Prevents cheating from "poisoning" the
  baseline.
- Rolling-window velocity: average over 5 frames, not frame-diff.
  Frame-diff velocity is O(noise), not O(signal).
- Temporal analysis: Single events are NOT evidence. Patterns are.
  Frequency + ratio + duration in a 20s window is what matters.
- EMA smoothing on suspicion: Prevents flickering alerts that mean nothing.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ═══════════════════════════════════════════════════════════════════
# CONFIG — All tunables in one place
# ═══════════════════════════════════════════════════════════════════


class Config:
    # Calibration
    CALIBRATION_FRAMES = 150  # ~5 seconds at 30fps
    ADAPTIVE_ALPHA = 0.0015  # Baseline drift speed (very slow)
    MIN_STD_DEG = 3.5
    MAX_STD_DEG = 5.0  # Cap — prevents huge threshold from wide calibration movement

    # Adaptive threshold
    # Direction = "Away" if |yaw - baseline| > K_SIGMA * std_dev
    K_SIGMA = 2.3  # Was 2.8 → 2.3. Threshold now ≈ 8° (was ≈ 10°).
    # 8° is realistic: looking at neighbour's paper = ~15°,
    # natural thinking glance = ~5°. This catches real cheating.

    # Kinematic
    VELOCITY_WINDOW = 7
    VELOCITY_CAP = 120.0
    ACCEL_CAP = 400.0

    # Temporal
    TEMPORAL_WINDOW_SEC = 20
    MIN_EVENT_DURATION = 1.2

    # Suspicion fusion weights (must sum to 1.0)
    W_POSITION = 0.15  # Reduced — position alone is weak evidence
    W_VELOCITY = 0.10  # Unchanged
    W_FREQUENCY = 0.20  # Reduced slightly — window-based, can be gamed
    W_DURATION = 0.25  # Unchanged — away time is honest signal
    W_SESSION = 0.30  # Increased from 0.20 → 0.30.
    # Session memory now has most weight.
    # Past cheating STAYS on record and matters most.

    # Smoothing
    SUSPICION_EMA_ALPHA = 0.08  # Was 0.035 → 0.08. Score reacts faster.
    # Previous was so slow it felt dead.

    # Alert thresholds — lowered so they're actually reachable
    LEVEL_LOW = 18  # Was 20
    LEVEL_MILD = 35  # Was 38
    LEVEL_MODERATE = 50  # Was 58 — most important change
    LEVEL_HIGH = 65  # Was 75 — now reachable after sustained cheating

    # Confirmed event accumulation
    CONFIRMED_THRESHOLD = 75.0  # % above which time accumulates


# ═══════════════════════════════════════════════════════════════════
# KALMAN FILTER — 1D, tracks position + velocity
# ═══════════════════════════════════════════════════════════════════


class KalmanFilter1D:
    """
    State vector: [yaw_position, yaw_velocity]
    Gives us smooth yaw AND velocity as a byproduct of filtering —
    no need for a separate noisy frame-diff calculation.
    """

    def __init__(self, process_noise: float = 0.8, measurement_noise: float = 8.0):
        self.Q = process_noise  # Process noise covariance
        self.R = measurement_noise  # Measurement noise covariance
        self.x = np.array([0.0, 0.0])  # State: [position, velocity]
        self.P = np.eye(2) * 10.0  # Error covariance (start uncertain)
        self.initialized = False

    def update(self, measurement: float, dt: float) -> Tuple[float, float]:
        dt = max(dt, 0.001)
        F = np.array([[1.0, dt], [0.0, 1.0]])  # State transition
        H = np.array([[1.0, 0.0]])  # Measurement matrix
        Q = np.array(
            [
                [self.Q * dt**3 / 3, self.Q * dt**2 / 2],
                [self.Q * dt**2 / 2, self.Q * dt],
            ]
        )

        if not self.initialized:
            self.x[0] = measurement
            self.initialized = True
            return float(self.x[0]), float(self.x[1])

        # Predict
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        # Update (Kalman gain)
        S = H @ P_pred @ H.T + self.R
        K = (P_pred @ H.T) / S[0, 0]
        innovation = float(measurement - (H @ x_pred)[0])
        self.x = x_pred + K.flatten() * innovation
        self.P = (np.eye(2) - np.outer(K, H)) @ P_pred

        return float(self.x[0]), float(self.x[1])


# ═══════════════════════════════════════════════════════════════════
# CALIBRATION ENGINE — Welford online mean/variance
# ═══════════════════════════════════════════════════════════════════


class CalibrationEngine:
    """
    Phase 1 (Hard calibration): Welford's algorithm — numerically stable,
    single-pass, no sample storage. Runs for CALIBRATION_FRAMES frames.

    Phase 2 (Adaptive drift): EMA update of baseline, only when the
    student is clearly looking forward (z < 1.0). This way the baseline
    slowly adapts to natural posture shifts without being gamed.
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Running sum of squared deviations
        self.std = Config.MIN_STD_DEG
        self.is_calibrated = False
        self.baseline = 0.0  # Adaptive mean (post-calibration)

    def feed(self, yaw: float) -> bool:
        """Feed a yaw sample during calibration. Returns True when done."""
        self.n += 1
        delta = yaw - self.mean
        self.mean += delta / self.n
        delta2 = yaw - self.mean
        self.M2 += delta * delta2

        if self.n >= Config.CALIBRATION_FRAMES:
            variance = self.M2 / (self.n - 1) if self.n > 1 else 1.0
            raw_std = math.sqrt(variance)
            self.std = max(Config.MIN_STD_DEG, min(raw_std, Config.MAX_STD_DEG))
            self.baseline = self.mean
            self.is_calibrated = True

        return self.is_calibrated

    def adapt(self, yaw: float, z: float):
        """Slowly drift baseline — only when clearly forward.
        Was z < 1.0 → changed to z < 0.5 so the baseline does NOT
        drift when student is even slightly away. Prevents the system
        from slowly 'accepting' a turned head as normal."""
        if z < 0.5:
            self.baseline = (
                1 - Config.ADAPTIVE_ALPHA
            ) * self.baseline + Config.ADAPTIVE_ALPHA * yaw

    def z_score(self, yaw: float) -> float:
        """Signed z-score: positive = right of baseline, negative = left."""
        return (yaw - self.baseline) / self.std

    def is_away(self, yaw: float) -> Tuple[bool, str]:
        """True if yaw exceeds the adaptive threshold."""
        z = self.z_score(yaw)
        if abs(z) > Config.K_SIGMA:
            return True, ("Right" if z > 0 else "Left")
        return False, "Forward"

    @property
    def progress(self) -> float:
        return min(self.n / Config.CALIBRATION_FRAMES, 1.0)


# ═══════════════════════════════════════════════════════════════════
# KINEMATIC ANALYZER — Rolling-window velocity + acceleration
# ═══════════════════════════════════════════════════════════════════


class KinematicAnalyzer:
    """
    Uses a rolling window of (yaw, time) pairs to compute:
    - velocity  = mean of finite differences over window
    - acceleration = change in velocity over window

    Frame-diff velocity is O(noise). A rolling mean is much more stable.
    """

    def __init__(self):
        self._yaw = deque(maxlen=Config.VELOCITY_WINDOW)
        self._time = deque(maxlen=Config.VELOCITY_WINDOW)
        self.velocity = 0.0
        self.acceleration = 0.0
        self._prev_vel = 0.0
        self.max_velocity = 0.0

    def update(self, yaw: float, t: float):
        self._yaw.append(yaw)
        self._time.append(t)

        if len(self._yaw) >= 3:
            # Compute all finite differences in the window
            diffs = []
            for i in range(1, len(self._yaw)):
                dt = self._time[i] - self._time[i - 1]
                if dt > 1e-4:
                    diffs.append(abs((self._yaw[i] - self._yaw[i - 1]) / dt))
            if diffs:
                self.velocity = float(np.mean(diffs))
                dt_total = self._time[-1] - self._time[0]
                if dt_total > 1e-4:
                    self.acceleration = abs(self.velocity - self._prev_vel) / dt_total
                self._prev_vel = self.velocity
                self.max_velocity = max(self.max_velocity, self.velocity)

    @property
    def velocity_norm(self) -> float:
        return min(self.velocity / Config.VELOCITY_CAP, 1.0)

    @property
    def accel_norm(self) -> float:
        return min(self.acceleration / Config.ACCEL_CAP, 1.0)


# ═══════════════════════════════════════════════════════════════════
# TEMPORAL PATTERN ENGINE — Rolling window behavioral statistics
# ═══════════════════════════════════════════════════════════════════


@dataclass
class DeviationEvent:
    timestamp: float
    duration: float
    direction: str
    z_score: float


class TemporalPatternEngine:
    """
    Tracks deviation events in a rolling time window and computes:
    - frequency_score : events per minute, normalized
    - away_ratio      : fraction of session time spent looking away
    - duration_score  : average event duration, normalized

    Single events are noise. Patterns are signal.
    """

    def __init__(self):
        self.events: List[DeviationEvent] = []
        self._away_start: Optional[float] = None
        self._away_dir: str = "Forward"
        self._away_z: float = 0.0
        self.total_away_s: float = 0.0
        self._session_start = time.time()

    # ── State transitions ──────────────────────────────────────────

    def mark_away(self, t: float, direction: str, z: float):
        if self._away_start is None:
            self._away_start = t
            self._away_dir = direction
            self._away_z = z
        else:
            # Update direction/z if still away
            self._away_z = max(self._away_z, z)

    def mark_forward(self, t: float):
        if self._away_start is not None:
            dur = t - self._away_start
            if dur >= Config.MIN_EVENT_DURATION:
                self.events.append(
                    DeviationEvent(
                        timestamp=t,
                        duration=dur,
                        direction=self._away_dir,
                        z_score=self._away_z,
                    )
                )
                self.total_away_s += dur
            self._away_start = None

    def is_currently_away(self) -> bool:
        return self._away_start is not None

    # ── Scoring ───────────────────────────────────────────────────

    def _recent(self, t: float) -> List[DeviationEvent]:
        cutoff = t - Config.TEMPORAL_WINDOW_SEC
        return [e for e in self.events if e.timestamp > cutoff]

    def frequency_score(self, t: float) -> float:
        """
        First event is free — only repeated looks build score.
        0=0%, 1=0%, 2=25%, 3=50%, 5+=100%.
        """
        n = len(self._recent(t))
        return min(max(n - 1, 0) / 4.0, 1.0)

    def away_ratio(self, t: float) -> float:
        """
        NON-LINEAR away ratio score.
        Old: 20% away time → 20% score (linear, too forgiving)
        New: 20% away time → 67% score using x^0.5 curve
             This means spending 1/5 of exam looking away is already HIGH evidence.
        Thresholds that feel right:
          5% away  → 32% score  (minor)
          15% away → 58% score  (concerning)
          25% away → 75% score  (high)
          40%+ away → 95%+ score (extreme)
        """
        total = t - self._session_start
        if total < 1.0:
            return 0.0
        away = self.total_away_s
        if self._away_start:
            away += t - self._away_start
        raw_ratio = min(away / total, 1.0)
        # Square-root curve: score = sqrt(raw_ratio / 0.40)
        # At 40% away time → score = 1.0 (100%)
        return min(math.sqrt(raw_ratio / 0.40), 1.0)

    def session_score(self, t: float) -> float:
        """
        CUMULATIVE SESSION MEMORY — never forgets past behavior.
        This is the fix for the core problem: after looking straight
        for 2-3 minutes, all other scores decay back to zero.
        This score accumulates based on total events and total away
        time over the ENTIRE session, not just recent activity.

        Formula:
          - Each completed event adds (duration / 4.0) points
            capped at 1.0 per event (a 4s event = max contribution)
          - Score = sum of all event contributions / 8  (8 = max expected)
          - Saturates at 1.0 (100%)

        Result: 19 events like in Image 2 → this score stays HIGH
        even after 2-3 min of straight looking.
        """
        if not self.events:
            return 0.0
        total_score = sum(min(e.duration / 4.0, 1.0) for e in self.events)
        return min(total_score / 5.0, 1.0)  # Was /8 → /5: memory builds faster

    def duration_score(self, t: float) -> float:
        """Mean event duration in window, normalized. 6s avg = 1.0"""
        recent = self._recent(t)
        if not recent:
            return 0.0
        avg = float(np.mean([e.duration for e in recent]))
        return min(avg / 6.0, 1.0)

    def recent_count(self, t: float) -> int:
        return len(self._recent(t))

    def total_count(self) -> int:
        return len(self.events)


# ═══════════════════════════════════════════════════════════════════
# SUSPICION FUSION — Weighted multi-signal EMA scoring
# ═══════════════════════════════════════════════════════════════════


@dataclass
class SuspicionState:
    probability: float = 0.0
    max_probability: float = 0.0
    confirmed_secs: float = 0.0  # Time above HIGH threshold
    # Component breakdown (for dashboard bars)
    pos_component: float = 0.0
    vel_component: float = 0.0
    freq_component: float = 0.0
    dur_component: float = 0.0
    session_component: float = 0.0  # Cumulative memory score


class SuspicionFusion:
    """
    Fuses 4 normalized (0–1) signals into a single 0–100% probability.

    Formula:
        raw = W_pos * pos + W_vel * vel + W_freq * freq + W_dur * dur
        prob = EMA(raw * 100, alpha)

    EMA smoothing prevents the score from flickering with every frame.
    Confirmed seconds accumulate only when probability > HIGH threshold.
    """

    def __init__(self):
        self.state = SuspicionState()
        self._smoothed = 0.0

    def update(
        self,
        pos_score: float,
        vel_score: float,
        freq_score: float,
        dur_score: float,
        session_score: float,
        dt: float,
    ) -> SuspicionState:

        raw = (
            Config.W_POSITION * pos_score
            + Config.W_VELOCITY * vel_score
            + Config.W_FREQUENCY * freq_score
            + Config.W_DURATION * dur_score
            + Config.W_SESSION * session_score
        ) * 100.0

        # EMA
        a = Config.SUSPICION_EMA_ALPHA
        self._smoothed = a * raw + (1 - a) * self._smoothed
        prob = max(0.0, min(self._smoothed, 100.0))

        # Accumulate confirmed time
        if prob > Config.CONFIRMED_THRESHOLD:
            self.state.confirmed_secs += dt

        # Store
        self.state.probability = prob
        self.state.max_probability = max(self.state.max_probability, prob)
        self.state.pos_component = pos_score * 100
        self.state.vel_component = vel_score * 100
        self.state.freq_component = freq_score * 100
        self.state.dur_component = dur_score * 100
        self.state.session_component = session_score * 100

        return self.state

    @staticmethod
    def get_level(prob: float) -> Tuple[str, Tuple[int, int, int]]:
        if prob >= Config.LEVEL_HIGH:
            return "HIGH RISK", (0, 0, 255)
        elif prob >= Config.LEVEL_MODERATE:
            return "MODERATE", (0, 100, 255)
        elif prob >= Config.LEVEL_MILD:
            return "MILD", (0, 200, 200)
        elif prob >= Config.LEVEL_LOW:
            return "LOW", (0, 220, 100)
        else:
            return "NORMAL", (0, 255, 60)


# ═══════════════════════════════════════════════════════════════════
# HEAD POSE EXTRACTION — SolvePnP from MediaPipe landmarks
# ═══════════════════════════════════════════════════════════════════

_LANDMARK_IDS = [1, 33, 263, 199, 61, 291]
_BBOX_IDS = [10, 152, 234, 454]


def extract_yaw(landmarks, w: int, h: int) -> Tuple[Optional[float], bool, float]:
    """Returns (yaw_deg, is_profile, face_aspect_ratio)."""
    bbox_pts = [landmarks.landmark[i] for i in _BBOX_IDS]
    fx = [p.x * w for p in bbox_pts]
    fy = [p.y * h for p in bbox_pts]
    face_w = max(fx) - min(fx)
    face_h = max(fy) - min(fy)
    far = face_w / face_h if face_h > 1 else 1.0
    is_profile = far < 0.52

    face_2d, face_3d = [], []
    for idx in _LANDMARK_IDS:
        pt = landmarks.landmark[idx]
        x, y = int(pt.x * w), int(pt.y * h)
        face_2d.append([x, y])
        face_3d.append([x, y, pt.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    cam = np.array([[w, 0, h / 2], [0, w, w / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))

    success, rot_vec, _ = cv2.solvePnP(
        face_3d, face_2d, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None, is_profile, far

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pnp_yaw = float(angles[1] * 180.0)

    if is_profile:
        nose_x = landmarks.landmark[1].x
        left_eye_x = landmarks.landmark[33].x
        right_eye_x = landmarks.landmark[263].x
        eye_mid_x = (left_eye_x + right_eye_x) / 2.0
        dir_sign = 1.0 if nose_x < eye_mid_x else -1.0
        return dir_sign * 32.0, True, far

    return pnp_yaw, False, far


# ═══════════════════════════════════════════════════════════════════
# DASHBOARD UI — Professional OpenCV overlay
# ═══════════════════════════════════════════════════════════════════


def _draw_bar(
    frame,
    x: int,
    y: int,
    value: float,
    color: tuple,
    label: str,
    bar_w: int = 220,
    bar_h: int = 13,
):
    pct = max(0.0, min(value / 100.0, 1.0))
    cv2.putText(
        frame,
        label,
        (x, y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (140, 140, 140),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (35, 35, 35), -1)
    if pct > 0:
        cv2.rectangle(frame, (x, y), (x + int(bar_w * pct), y + bar_h), color, -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (70, 70, 70), 1)
    cv2.putText(
        frame,
        f"{value:.0f}%",
        (x + bar_w + 6, y + bar_h - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_dashboard(
    frame,
    w: int,
    h: int,
    calib: CalibrationEngine,
    kinematic: KinematicAnalyzer,
    temporal: TemporalPatternEngine,
    fusion_state: SuspicionState,
    yaw_smooth: float,
    t: float,
    is_profile: bool = False,
    face_ar: float = 1.0,
):

    prob = fusion_state.probability
    level, lc = SuspicionFusion.get_level(prob)

    # ── Header bar ────────────────────────────────────────────────
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 52), (10, 12, 18), -1)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
    cv2.putText(
        frame,
        "HEAD POSE INTELLIGENCE  |  CPG-127",
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    # Right-align level text — measure its pixel width to avoid overlap
    (lw, _), _ = cv2.getTextSize(level, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
    cv2.putText(
        frame,
        level,
        (w - lw - 12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        lc,
        2,
        cv2.LINE_AA,
    )

    # ── Right panel background ────────────────────────────────────
    px = w - 300
    ov2 = frame.copy()
    cv2.rectangle(ov2, (px - 8, 55), (w, h), (8, 10, 16), -1)
    cv2.addWeighted(ov2, 0.78, frame, 0.22, 0, frame)
    cv2.line(frame, (px - 8, 55), (px - 8, h), (40, 40, 50), 1)

    def txt(text, y, color=(180, 180, 180), scale=0.46, bold=1):
        cv2.putText(
            frame,
            text,
            (px, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            bold,
            cv2.LINE_AA,
        )

    y = 82
    txt("SUSPICION SCORE", y, (220, 220, 220), 0.52, 2)
    y += 26
    _draw_bar(frame, px, y, prob, lc, "Overall Probability", 255, 16)
    y += 36

    txt("SIGNAL BREAKDOWN", y, (160, 160, 160), 0.40)
    y += 18
    _draw_bar(
        frame,
        px,
        y,
        fusion_state.pos_component,
        (100, 200, 255),
        "Position Anomaly  (20%)",
    )
    y += 26
    _draw_bar(
        frame,
        px,
        y,
        fusion_state.vel_component,
        (255, 165, 60),
        "Angular Velocity  (10%)",
    )
    y += 26
    _draw_bar(
        frame,
        px,
        y,
        fusion_state.freq_component,
        (180, 80, 255),
        "Event Frequency   (25%)",
    )
    y += 26
    _draw_bar(
        frame,
        px,
        y,
        fusion_state.dur_component,
        (255, 80, 80),
        "Away-Time Ratio   (25%)",
    )
    y += 26
    _draw_bar(
        frame,
        px,
        y,
        fusion_state.session_component,
        (255, 220, 0),
        "Session Memory    (20%)",
    )
    y += 32

    txt("KINEMATICS", y, (160, 160, 160), 0.40)
    y += 18
    txt(f"Yaw (smooth):  {yaw_smooth:+6.1f} deg", y)
    y += 20
    if is_profile:
        cv2.rectangle(frame, (px, y - 13), (px + 255, y + 5), (0, 50, 160), -1)
        cv2.putText(
            frame,
            f"PROFILE DETECTED  FAR={face_ar:.2f}",
            (px + 4, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        txt(f"Face Ratio:    {face_ar:.2f}  (profile<0.52)", y, (120, 120, 120), 0.40)
    y += 20
    txt(f"Baseline:      {calib.baseline:+6.1f} deg", y)
    y += 20
    z = calib.z_score(yaw_smooth)
    zcol = (
        (0, 255, 60)
        if abs(z) < 1.0
        else (0, 100, 255) if abs(z) < Config.K_SIGMA else (0, 0, 255)
    )
    txt(f"Z-Score:       {z:+6.2f}  std={calib.std:.2f}", y, zcol)
    y += 20
    txt(f"Velocity:     {kinematic.velocity:6.1f} deg/s", y)
    y += 20
    txt(f"Max Velocity: {kinematic.max_velocity:6.1f} deg/s", y)
    y += 30

    txt("SESSION STATS", y, (160, 160, 160), 0.40)
    y += 18
    rc = temporal.recent_count(t)
    txt(
        f"Events (20s window): {rc}",
        y,
        (0, 0, 255) if rc >= 5 else (0, 200, 200) if rc >= 2 else (180, 180, 180),
    )
    y += 20
    txt(f"Total events:        {temporal.total_count()}", y)
    y += 20
    txt(f"Away ratio:          {temporal.away_ratio(t) * 100:.1f}%", y)
    y += 20
    txt(f"Peak suspicion:      {fusion_state.max_probability:.1f}%", y, lc)
    y += 20
    conf = fusion_state.confirmed_secs
    txt(
        f"Confirmed time:      {conf:.1f}s",
        y,
        (0, 0, 255) if conf > 3 else (180, 180, 180),
    )
    y += 26

    # ── TEACHER VERDICT ───────────────────────────────────────────
    rc = temporal.recent_count(t)
    ar = temporal.away_ratio(t)  # already non-linear score 0-1
    tc = temporal.total_count()

    ev_freq = 1 if rc >= 3 else 0
    ev_dur = 1 if ar > 0.55 else 0  # ~15% raw away time → 0.58 score
    ev_conf = 1 if conf > 5.0 else 0
    ev_peak = 1 if fusion_state.max_probability >= Config.LEVEL_HIGH else 0
    evidence = ev_freq + ev_dur + ev_conf + ev_peak

    if evidence >= 3:
        v_text, v_color, v_bg = "!! FLAGGED FOR REVIEW !!", (0, 0, 255), (60, 0, 0)
    elif evidence == 2:
        v_text, v_color, v_bg = "ALERT INVIGILATOR", (0, 100, 255), (40, 20, 0)
    elif evidence == 1:
        v_text, v_color, v_bg = "MONITOR CLOSELY", (0, 200, 200), (0, 30, 30)
    else:
        v_text, v_color, v_bg = "NO CONCERN", (80, 200, 80), (0, 20, 0)

    cv2.rectangle(frame, (px - 4, y - 2), (w - 4, y + 52), v_bg, -1)
    cv2.rectangle(frame, (px - 4, y - 2), (w - 4, y + 52), v_color, 1)
    txt("TEACHER VERDICT", y + 12, (160, 160, 160), 0.38)
    txt(v_text, y + 34, v_color, 0.48, 2)
    y += 60

    def ev_dot(label, passed, yy):
        dot_c = (0, 220, 80) if passed else (60, 60, 60)
        cv2.circle(frame, (px + 7, yy - 4), 5, dot_c, -1, cv2.LINE_AA)
        txt(label, yy, dot_c if passed else (90, 90, 90), 0.36)

    raw_ar = temporal.total_away_s / max(t - temporal._session_start, 1)
    ev_dot(f"Repeated looks  ({rc} in 20s, need 3)", ev_freq, y)
    y += 17
    ev_dot(f"Away time       ({raw_ar*100:.1f}%, need 15%)", ev_dur, y)
    y += 17
    ev_dot(f"High suspicion  ({conf:.1f}s, need 5s)", ev_conf, y)
    y += 17
    ev_dot(f"Peak HIGH       ({fusion_state.max_probability:.0f}%)", ev_peak, y)

    # ── Currently away indicator ───────────────────────────────────
    if temporal.is_currently_away():
        away_dur = t - temporal._away_start if temporal._away_start else 0
        pulse = int(127 + 127 * math.sin(t * 4))
        cv2.putText(
            frame,
            f"LOOKING AWAY  {away_dur:.1f}s",
            (px, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, pulse // 2, pulse),
            2,
            cv2.LINE_AA,
        )

    # ── Alert border (pulsing at HIGH) ────────────────────────────
    if prob >= Config.LEVEL_HIGH:
        pulse_t = int(4 + 5 * abs(math.sin(t * 3.5)))
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), pulse_t)
    elif prob >= Config.LEVEL_MODERATE:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 100, 255), 3)

    # ── Calibration overlay ───────────────────────────────────────
    if not calib.is_calibrated:
        ov3 = frame.copy()
        cv2.rectangle(ov3, (0, h // 2 - 70), (w, h // 2 + 70), (5, 5, 10), -1)
        cv2.addWeighted(ov3, 0.82, frame, 0.18, 0, frame)

        pulse_c = int(180 + 75 * abs(math.sin(t * 2)))
        cv2.putText(
            frame,
            "AI CALIBRATING — LOOK FORWARD & STAY STILL",
            (w // 2 - 330, h // 2 - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, pulse_c, pulse_c),
            2,
            cv2.LINE_AA,
        )

        prog = calib.progress
        bw, bx, by = 480, w // 2 - 240, h // 2 + 12
        cv2.rectangle(frame, (bx, by), (bx + bw, by + 28), (30, 30, 30), -1)
        cv2.rectangle(
            frame, (bx, by), (bx + int(bw * prog), by + 28), (0, 200, 140), -1
        )
        cv2.rectangle(frame, (bx, by), (bx + bw, by + 28), (80, 80, 80), 1)
        cv2.putText(
            frame,
            f"{int(prog * 100)}%  ({calib.n}/{Config.CALIBRATION_FRAMES} frames)",
            (bx + bw // 2 - 60, by + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


# ═══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════


def main():
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] HEAD POSE INTELLIGENCE ENGINE STARTED")
    print(f"[INFO] Calibration: {Config.CALIBRATION_FRAMES} frames")
    print(f"[INFO] Adaptive threshold: ±{Config.K_SIGMA}σ")
    print("[INFO] Press ESC to exit.\n")

    # ── Engine instances ──────────────────────────────────────────
    kalman = KalmanFilter1D(process_noise=0.5, measurement_noise=6.0)
    calib = CalibrationEngine()
    kinematic = KinematicAnalyzer()
    temporal = TemporalPatternEngine()
    fusion = SuspicionFusion()

    prev_t = time.time()
    yaw_smooth = 0.0
    fs = fusion.state  # live reference

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        t = time.time()
        dt = max(t - prev_t, 1e-3)
        prev_t = t

        # ── Face mesh ─────────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        yaw_raw = 0.0
        face_present = False
        is_profile = False
        face_ar = 1.0

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            face_present = True
            yaw_raw, is_profile, face_ar = extract_yaw(lm, w, h)
            if yaw_raw is None:
                yaw_raw = 0.0
            for idx in _LANDMARK_IDS:
                pt = lm.landmark[idx]
                cx, cy = int(pt.x * w), int(pt.y * h)
                dot_color = (0, 80, 255) if is_profile else (0, 230, 230)
                cv2.circle(frame, (cx, cy), 2, dot_color, -1, cv2.LINE_AA)

        # ── Kalman smooth ─────────────────────────────────────────
        yaw_smooth, _ = kalman.update(yaw_raw, dt)

        # ── Calibration phase ─────────────────────────────────────
        if face_present:
            calib.feed(yaw_smooth)

        # ── Monitoring phase (only after calibration) ─────────────
        if calib.is_calibrated and face_present:

            kinematic.update(yaw_smooth, t)

            z = calib.z_score(yaw_smooth)
            away, direction = calib.is_away(yaw_smooth)
            calib.adapt(yaw_smooth, abs(z))

            if away:
                temporal.mark_away(t, direction, abs(z))
            else:
                temporal.mark_forward(t)

            # ── Component scores (normalized 0–1) ─────────────────
            pos_score = min(abs(z) / (Config.K_SIGMA * 2.5), 1.0)
            vel_score = kinematic.velocity_norm
            freq_score = temporal.frequency_score(t)
            dur_score = temporal.away_ratio(t)  # non-linear
            sess_score = temporal.session_score(t)  # cumulative memory

            fs = fusion.update(
                pos_score, vel_score, freq_score, dur_score, sess_score, dt
            )

        # ── Draw ──────────────────────────────────────────────────
        draw_dashboard(
            frame,
            w,
            h,
            calib,
            kinematic,
            temporal,
            fs,
            yaw_smooth,
            t,
            is_profile=is_profile,
            face_ar=face_ar,
        )

        cv2.imshow("Head Pose Intelligence - CPG 127", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Session ended.")
    print(f"[STATS] Total events:     {temporal.total_count()}")
    print(f"[STATS] Peak suspicion:   {fs.max_probability:.1f}%")
    print(f"[STATS] Confirmed time:   {fs.confirmed_secs:.1f}s")
    print(f"[STATS] Learned baseline: {calib.baseline:.2f}°  std={calib.std:.2f}°")


if __name__ == "__main__":
    main()
