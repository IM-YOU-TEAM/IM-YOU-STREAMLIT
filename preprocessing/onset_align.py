"""Align OpenSim motion-capture angles to model prediction time via hip peaks.

Phone and OpenSim clocks are independent. Two strategies:
  - match hip flexion peaks
  - coarse stillness→walk onset, then refine with hip flexion peaks
"""
from __future__ import annotations

import io
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

OPENSIM_JOINT_COLS = ("hip_flexion_r", "knee_angle_r", "ankle_angle_r")
JOINT_INDEX = {"hip": 0, "knee": 1, "ankle": 2}


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


def _detect_onset_from_activity(
    activity: np.ndarray,
    fs: float,
    *,
    quiet_s: float = 0.5,
    k_std: float = 5.0,
    min_above_s: float = 0.15,
) -> float:
    """Onset time (s from activity[0]) when activity leaves initial stillness."""
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

    d = np.abs(np.diff(activity, prepend=activity[0]))
    d[:n_quiet] = 0.0
    return int(np.argmax(d)) / fs


def _imu_onset(imu: np.ndarray, fs: float = 100.0) -> float:
    imu = np.asarray(imu, dtype=float)
    if imu.ndim != 2 or imu.shape[1] < 6:
        raise ValueError(f"Expected IMU [N,6+], got {imu.shape}")
    gyro_mag = _smooth(np.linalg.norm(imu[:, 3:6], axis=1), fs=fs, window_s=0.08)
    return _detect_onset_from_activity(gyro_mag, fs=fs)


def _opensim_onset(time_s: np.ndarray, angles_deg: np.ndarray) -> float:
    time_s = np.asarray(time_s, dtype=float)
    angles_deg = np.asarray(angles_deg, dtype=float)
    if len(time_s) < 3:
        return float(time_s[0]) if len(time_s) else 0.0
    dt = np.diff(time_s)
    dt = dt[dt > 0]
    fs = 1.0 / float(np.median(dt)) if len(dt) else 60.0
    speed = np.linalg.norm(np.gradient(angles_deg, time_s, axis=0), axis=1)
    speed = _smooth(speed, fs=fs, window_s=0.08)
    return float(time_s[0] + _detect_onset_from_activity(speed, fs=fs))


def _coarse_onset_shift(
    opensim_time_s: np.ndarray,
    opensim_angles_deg: np.ndarray,
    imu: np.ndarray,
    imu_fs: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Shift OpenSim so stillness→walk onset matches IMU gyro onset."""
    imu_onset = _imu_onset(imu, fs=imu_fs)
    osim_onset = _opensim_onset(opensim_time_s, opensim_angles_deg)
    offset = -osim_onset + imu_onset
    return opensim_time_s + offset, {
        "imu_onset_s": float(imu_onset),
        "opensim_onset_s": float(osim_onset),
        "coarse_offset_s": float(offset),
    }


def _uniform_series(time_s: np.ndarray, values: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(time_s) < 2:
        return time_s, values
    t0, t1 = float(time_s[0]), float(time_s[-1])
    grid = np.arange(t0, t1 + 0.5 / fs, 1.0 / fs)
    return grid, np.interp(grid, time_s, values)


def find_joint_peak_times(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    *,
    fs: float = 100.0,
    min_period_s: float = 0.45,
    prominence_frac: float = 0.12,
    use_valleys: bool = False,
) -> np.ndarray:
    """Times (s) of prominent peaks (or valleys) in a joint-angle series."""
    t_u, y_u = _uniform_series(time_s, angle_deg, fs=fs)
    if len(y_u) < 5:
        return np.asarray([], dtype=float)

    y = -y_u if use_valleys else y_u
    y_range = float(np.ptp(y))
    prominence = max(prominence_frac * y_range, 1.0) if y_range > 0 else 1.0
    distance = max(int(round(min_period_s * fs)), 1)
    peaks, _props = find_peaks(y, distance=distance, prominence=prominence)
    if len(peaks) == 0:
        peaks, _props = find_peaks(y, distance=distance, prominence=max(prominence * 0.5, 0.5))
    return t_u[peaks]


def _median_peak_offset(pred_peaks: np.ndarray, osim_peaks: np.ndarray) -> Optional[float]:
    """Offset to add to OpenSim times so its peaks line up with prediction peaks."""
    if len(pred_peaks) == 0 or len(osim_peaks) == 0:
        return None
    n = min(len(pred_peaks), len(osim_peaks))
    return float(np.median(pred_peaks[:n] - osim_peaks[:n]))


def match_joint_peaks(
    opensim_time_s: np.ndarray,
    opensim_angles_deg: np.ndarray,
    pred_time_s: np.ndarray,
    pred_angles_deg: np.ndarray,
    *,
    joint: str = "hip",
    fs: float = 100.0,
    coarse_onset_imu: Optional[np.ndarray] = None,
    imu_fs: float = 100.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Shift OpenSim times so joint-angle peaks align with model prediction peaks.

    Default joint is hip flexion. If `coarse_onset_imu` is provided, first aligns
    stillness→walk onsets, then refines with peak matching.

    Returns (opensim_time_on_pred_axis, info).
    """
    opensim_time_s = np.asarray(opensim_time_s, dtype=float)
    opensim_angles_deg = np.asarray(opensim_angles_deg, dtype=float)
    pred_time_s = np.asarray(pred_time_s, dtype=float)
    pred_angles_deg = np.asarray(pred_angles_deg, dtype=float)

    if joint not in JOINT_INDEX:
        raise ValueError(f"joint must be one of {list(JOINT_INDEX)}, got {joint!r}")
    j = JOINT_INDEX[joint]
    if opensim_angles_deg.ndim != 2 or opensim_angles_deg.shape[1] <= j:
        raise ValueError("opensim_angles_deg must be [N, >=joint]")
    if pred_angles_deg.ndim != 2 or pred_angles_deg.shape[1] <= j:
        raise ValueError("pred_angles_deg must be [N, >=joint]")

    info: Dict[str, float] = {"joint_index": float(j)}
    t_plot = opensim_time_s.copy()

    if coarse_onset_imu is not None:
        t_plot, coarse = _coarse_onset_shift(
            opensim_time_s, opensim_angles_deg, coarse_onset_imu, imu_fs=imu_fs
        )
        info.update(coarse)
        info["offset_s"] = float(coarse["coarse_offset_s"])
    else:
        info["offset_s"] = 0.0

    pred_peaks = find_joint_peak_times(pred_time_s, pred_angles_deg[:, j], fs=fs)
    osim_peaks = find_joint_peak_times(t_plot, opensim_angles_deg[:, j], fs=fs)

    if len(pred_peaks) >= 2 and len(osim_peaks) < max(len(pred_peaks) // 2, 1):
        osim_alt = find_joint_peak_times(t_plot, opensim_angles_deg[:, j], fs=fs, use_valleys=True)
        if len(osim_alt) > len(osim_peaks):
            osim_peaks = osim_alt
            info["used_valleys"] = 1.0

    peak_offset = _median_peak_offset(pred_peaks, osim_peaks)
    if peak_offset is None:
        pred_y = pred_angles_deg[:, j]
        osim_y = np.interp(pred_time_s, t_plot, opensim_angles_deg[:, j], left=np.nan, right=np.nan)
        if np.isfinite(osim_y).sum() > 10:
            peak_offset = float(pred_time_s[int(np.nanargmax(pred_y))] - pred_time_s[int(np.nanargmax(osim_y))])
        else:
            peak_offset = 0.0
        info["peak_fallback"] = 1.0

    t_plot = t_plot + float(peak_offset)
    info["peak_offset_s"] = float(peak_offset)
    info["offset_s"] = float(info.get("offset_s", 0.0) + peak_offset)
    info["n_pred_peaks"] = float(len(pred_peaks))
    info["n_opensim_peaks"] = float(len(osim_peaks))
    return t_plot, info
