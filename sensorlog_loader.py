"""Load Sensor Logger export folders into BioMAT phone-session CSVs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_SENSORLOG_ROOT = Path(__file__).resolve().parent / "sensorlogData"


@dataclass
class SensorLogRecording:
    path: Path
    activity: str
    phone: int  # 1 = thigh, 2 = shank
    trial: int
    stamp: str
    label: str


@dataclass
class SensorLogPair:
    key: str
    activity: str
    trial: int
    thigh: SensorLogRecording
    shank: SensorLogRecording


def _normalize_folder_name(name: str) -> str:
    s = name.lower().strip()
    s = s.replace("wailking", "walking")
    s = s.replace("desctn", "descent")
    s = s.replace("phoje", "phone")
    s = s.replace("trail", "trial")
    s = s.replace("trials", "trial")
    s = re.sub(r"\s+\d+$", "", s)  # trailing " 2" duplicates
    return s


def parse_recording_folder(path: Path) -> Optional[SensorLogRecording]:
    if not path.is_dir():
        return None
    if not (path / "Accelerometer.csv").is_file() or not (path / "Gyroscope.csv").is_file():
        return None
    name = _normalize_folder_name(path.name)
    # activity_phone_N_trial_M-YYYY-MM-DD_HH-MM-SS
    m = re.match(
        r"^(?P<activity>.+?)_phone_(?P<phone>[12])_trial_(?P<trial>\d+)-(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$",
        name,
    )
    if not m:
        return None
    activity = m.group("activity").replace("_", " ")
    return SensorLogRecording(
        path=path,
        activity=activity,
        phone=int(m.group("phone")),
        trial=int(m.group("trial")),
        stamp=m.group("stamp"),
        label=f"{activity} · trial {m.group('trial')} · {m.group('stamp')}",
    )


def discover_recordings(root: Path = DEFAULT_SENSORLOG_ROOT) -> List[SensorLogRecording]:
    if not root.is_dir():
        return []
    out: List[SensorLogRecording] = []
    for path in sorted(root.iterdir()):
        rec = parse_recording_folder(path)
        if rec is not None:
            out.append(rec)
    return out


def discover_pairs(root: Path = DEFAULT_SENSORLOG_ROOT) -> List[SensorLogPair]:
    by_key: Dict[Tuple[str, int], Dict[int, SensorLogRecording]] = {}
    for rec in discover_recordings(root):
        key = (rec.activity, rec.trial)
        by_key.setdefault(key, {})[rec.phone] = rec
    pairs: List[SensorLogPair] = []
    for (activity, trial), phones in sorted(by_key.items()):
        if 1 in phones and 2 in phones:
            thigh, shank = phones[1], phones[2]
            pairs.append(
                SensorLogPair(
                    key=f"{activity} | trial {trial}",
                    activity=activity,
                    trial=trial,
                    thigh=thigh,
                    shank=shank,
                )
            )
    return pairs


def _xyz_from_sensorlogger(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Sensor Logger columns are time, seconds_elapsed, z, y, x."""
    t = df["seconds_elapsed"].to_numpy(dtype=float)
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)
    return t, np.column_stack([x, y, z])


def load_sensorlog_phone_df(folder: Path) -> Tuple[pd.DataFrame, dict]:
    """
    Build one phone schema DataFrame from a Sensor Logger folder.

    iOS Sensor Logger Accelerometer is typically linear acceleration (no gravity).
    BioMAT expects acceleration including gravity, so Gravity.csv is added when present.
    Gyro is treated as rad/s.
    """
    acc = pd.read_csv(folder / "Accelerometer.csv")
    gyro = pd.read_csv(folder / "Gyroscope.csv")
    t_acc, a = _xyz_from_sensorlogger(acc)
    t_gyro, w = _xyz_from_sensorlogger(gyro)

    # Interpolate gyro onto accelerometer time base.
    w_i = np.column_stack([np.interp(t_acc, t_gyro, w[:, i]) for i in range(3)])

    gravity_added = False
    grav_path = folder / "Gravity.csv"
    if grav_path.is_file():
        grav = pd.read_csv(grav_path)
        t_g, g = _xyz_from_sensorlogger(grav)
        g_i = np.column_stack([np.interp(t_acc, t_g, g[:, i]) for i in range(3)])
        # If accel is already ~1g at rest, skip; else add gravity.
        still = np.linalg.norm(a, axis=1) < 1.5
        mean_norm = float(np.mean(np.linalg.norm(a[still], axis=1))) if still.any() else float(np.mean(np.linalg.norm(a, axis=1)))
        if mean_norm < 5.0:
            a = a + g_i
            gravity_added = True

    keep = np.isfinite(t_acc) & np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(w_i), axis=1)
    t_acc, a, w_i = t_acc[keep], a[keep], w_i[keep]
    order = np.argsort(t_acc)
    t_acc, a, w_i = t_acc[order], a[order], w_i[order]
    uniq = np.concatenate([[True], np.diff(t_acc) > 0])
    t_acc, a, w_i = t_acc[uniq], a[uniq], w_i[uniq]

    out = pd.DataFrame(
        {
            "common_timestamp": t_acc,
            "ax": a[:, 0],
            "ay": a[:, 1],
            "az": a[:, 2],
            "gx": w_i[:, 0],
            "gy": w_i[:, 1],
            "gz": w_i[:, 2],
            "accel_units": "m/s^2",
            "gyro_units": "rad/s",
        }
    )
    hz = float((len(t_acc) - 1) / (t_acc[-1] - t_acc[0])) if len(t_acc) > 1 else 0.0
    info = {
        "folder": str(folder),
        "samples": len(out),
        "native_hz": hz,
        "gravity_added": gravity_added,
        "accel_units": "m/s^2",
        "gyro_units": "rad/s",
    }
    return out, info
