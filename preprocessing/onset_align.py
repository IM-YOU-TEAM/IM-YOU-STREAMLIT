"""Align OpenSim motion-capture angles to phone IMU time via stillness → motion onset.

Phone and OpenSim clocks are independent. For trials that begin with quiet standing
then walking, detect each stream's motion onset and shift OpenSim so the onsets coincide.
"""
from __future__ import annotations

import io
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

OPENSIM_JOINT_COLS = ("hip_flexion_r", "knee_angle_r", "ankle_angle_r")


def parse_opensim_motion(file_bytes: bytes, joint_cols: Sequence[str] = OPENSIM_JOINT_COLS) -> Tuple[pd.DataFrame, dict]:
    """Parse OpenSim Coordinates .mot/.csv (header ends at `endheader`).

    Returns a dataframe with `time` plus requested joint columns in degrees, and meta.
    """
    text = file_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    header_end = None
    meta: dict = {"inDegrees": True}
    for i, line in enumerate(lines):
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("indegrees"):
            meta["inDegrees"] = "yes" in lower
        if lower == "endheader":
            header_end = i
            break

    if header_end is None:
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_csv(io.StringIO("\n".join(lines[header_end + 1 :])))

    df.columns = [str(c).strip() for c in df.columns]
    required = ["time", *joint_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OpenSim file missing columns: {missing}")

    out = df[list(required)].apply(pd.to_numeric, errors="coerce").dropna()
    if out.empty:
        raise ValueError("OpenSim file has no numeric joint rows.")

    angles = out[list(joint_cols)].to_numpy(dtype=float)
    if not meta["inDegrees"]:
        angles = np.rad2deg(angles)

    result = pd.DataFrame({"time": out["time"].to_numpy(dtype=float)})
    for i, name in enumerate(joint_cols):
        result[name] = angles[:, i]
    return result, meta


def _smooth(signal: np.ndarray, fs: float, window_s: float = 0.05) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    kernel = max(int(round(window_s * fs)), 1)
    if kernel <= 1 or len(signal) < kernel:
        return signal
    if kernel % 2 == 0:
        kernel += 1
    w = np.ones(kernel) / kernel
    return np.convolve(signal, w, mode="same")


def detect_onset_from_activity(
    activity: np.ndarray,
    fs: float,
    *,
    quiet_s: float = 0.5,
    k_std: float = 5.0,
    min_above_s: float = 0.15,
) -> float:
    """Return onset time (seconds from activity[0]) when activity leaves initial stillness.

    Uses the first quiet window as baseline. Onset is the first sample after that window
    where activity stays above mean + k_std*std for at least `min_above_s`.
    """
    activity = np.asarray(activity, dtype=float)
    n = len(activity)
    if n < 5:
        return 0.0

    n_quiet = int(round(quiet_s * fs))
    n_quiet = max(5, min(n_quiet, max(n // 4, 5)))
    baseline = activity[:n_quiet]
    thr = float(np.mean(baseline) + k_std * (np.std(baseline) + 1e-9))
    min_run = max(int(round(min_above_s * fs)), 1)

    above = activity[n_quiet:] > thr
    run = 0
    for i, flag in enumerate(above):
        run = run + 1 if flag else 0
        if run >= min_run:
            return (n_quiet + i - min_run + 1) / fs

    # Fallback: strongest rise after the quiet window.
    d = np.abs(np.diff(activity, prepend=activity[0]))
    d[:n_quiet] = 0.0
    return int(np.argmax(d)) / fs


def imu_activity_from_gyro(imu: np.ndarray, fs: float) -> np.ndarray:
    """Scalar activity proxy for phone IMU: smoothed gyro magnitude."""
    imu = np.asarray(imu, dtype=float)
    if imu.ndim != 2 or imu.shape[1] < 6:
        raise ValueError(f"Expected IMU [N,6+], got {imu.shape}")
    gyro_mag = np.linalg.norm(imu[:, 3:6], axis=1)
    return _smooth(gyro_mag, fs=fs, window_s=0.08)


def opensim_activity_from_joints(
    time_s: np.ndarray,
    angles_deg: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Scalar activity proxy for OpenSim: smoothed joint angular-speed magnitude.

    Returns (activity_on_sample_grid, nominal_fs).
    """
    time_s = np.asarray(time_s, dtype=float)
    angles_deg = np.asarray(angles_deg, dtype=float)
    if len(time_s) != len(angles_deg):
        raise ValueError("OpenSim time/angle length mismatch")
    if len(time_s) < 3:
        return np.zeros(len(time_s)), 60.0

    dt = np.diff(time_s)
    dt = dt[dt > 0]
    fs = 1.0 / float(np.median(dt)) if len(dt) else 60.0
    # Gradient w.r.t. time for each joint, then magnitude.
    dadt = np.gradient(angles_deg, time_s, axis=0)
    speed = np.linalg.norm(dadt, axis=1)
    return _smooth(speed, fs=fs, window_s=0.08), fs


def detect_imu_onset(imu: np.ndarray, fs: float = 100.0, **kwargs) -> float:
    activity = imu_activity_from_gyro(imu, fs=fs)
    return detect_onset_from_activity(activity, fs=fs, **kwargs)


def detect_opensim_onset(time_s: np.ndarray, angles_deg: np.ndarray, **kwargs) -> float:
    """Onset in the OpenSim file's own time axis (not shifted)."""
    activity, fs = opensim_activity_from_joints(time_s, angles_deg)
    rel = detect_onset_from_activity(activity, fs=fs, **kwargs)
    return float(time_s[0] + rel)


def match_onsets(
    opensim_time_s: np.ndarray,
    opensim_angles_deg: np.ndarray,
    imu: np.ndarray,
    *,
    imu_fs: float = 100.0,
    manual_offset_s: Optional[float] = None,
    refine_with_knee_xcorr: bool = False,
    pred_time_s: Optional[np.ndarray] = None,
    pred_angles_deg: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Shift OpenSim times onto the IMU/model relative clock (0 = IMU series start).

    If `manual_offset_s` is set, uses `opensim_time + manual_offset_s`.
    Otherwise aligns OpenSim onset to IMU onset:
        t_plot = opensim_time - opensim_onset + imu_onset

    Optional knee cross-correlation fine-tunes the shift using predictions.
    """
    opensim_time_s = np.asarray(opensim_time_s, dtype=float)
    opensim_angles_deg = np.asarray(opensim_angles_deg, dtype=float)
    info: Dict[str, float] = {}

    if manual_offset_s is not None:
        t_plot = opensim_time_s + float(manual_offset_s)
        info["offset_s"] = float(manual_offset_s)
        info["method_code"] = 0.0
        return t_plot, info

    imu_onset = detect_imu_onset(imu, fs=imu_fs)
    osim_onset = detect_opensim_onset(opensim_time_s, opensim_angles_deg)
    offset = -osim_onset + imu_onset
    t_plot = opensim_time_s + offset
    info.update(
        {
            "imu_onset_s": float(imu_onset),
            "opensim_onset_s": float(osim_onset),
            "offset_s": float(offset),
            "method_code": 1.0,
        }
    )

    if (
        refine_with_knee_xcorr
        and pred_time_s is not None
        and pred_angles_deg is not None
        and len(pred_time_s) > 20
        and opensim_angles_deg.shape[1] >= 2
    ):
        fine = _knee_xcorr_shift(
            pred_time_s,
            pred_angles_deg[:, 1],
            t_plot,
            opensim_angles_deg[:, 1],
            fs=imu_fs,
        )
        t_plot = t_plot + fine
        info["xcorr_shift_s"] = float(fine)
        info["offset_s"] = float(info["offset_s"] + fine)
        info["method_code"] = 2.0

    return t_plot, info


def _knee_xcorr_shift(
    pred_t: np.ndarray,
    pred_knee: np.ndarray,
    osim_t: np.ndarray,
    osim_knee: np.ndarray,
    fs: float,
    max_lag_s: float = 1.0,
) -> float:
    pred_t = np.asarray(pred_t, dtype=float)
    osim_t = np.asarray(osim_t, dtype=float)
    t0 = max(pred_t[0], osim_t[0])
    t1 = min(pred_t[-1], osim_t[-1])
    if t1 <= t0 + 0.5:
        return 0.0

    grid = np.arange(t0, t1, 1.0 / fs)
    if len(grid) < 20:
        return 0.0
    a = np.interp(grid, pred_t, pred_knee)
    b = np.interp(grid, osim_t, osim_knee)
    a = a - np.mean(a)
    b = b - np.mean(b)
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-len(b) + 1, len(a))
    max_lag = int(round(max_lag_s * fs))
    mask = np.abs(lags) <= max_lag
    if not np.any(mask):
        return 0.0
    best = int(lags[mask][int(np.argmax(corr[mask]))])
    return best / fs
