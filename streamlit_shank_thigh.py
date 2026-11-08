"""
Streamlit UI for the standalone shank+thigh BioMAT packages (60 Hz and 100 Hz).

Run from the biomat folder (venv activated):
  streamlit run streamlit_shank_thigh.py

Expects sibling folders:
  ../biomat_shank_thigh/biomat_shank_thigh
  ../biomat_shank_thigh_100hz/biomat_shank_thigh_100hz
"""

from __future__ import annotations

import importlib
import io
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import torch
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

from sensorlog_loader import DEFAULT_SENSORLOG_ROOT, discover_pairs, load_sensorlog_phone_df

ROOT = Path(__file__).resolve().parent
TESTING_MODEL = ROOT.parent

PACKAGE_DIRS = {
    "60 Hz (canonical60_v1) — NEW shank+thigh weights": TESTING_MODEL
    / "biomat_shank_thigh"
    / "biomat_shank_thigh",
    "100 Hz (canonical100_v1) — NEW shank+thigh weights": TESTING_MODEL
    / "biomat_shank_thigh_100hz"
    / "biomat_shank_thigh_100hz",
}

TARGETS = ["hip_flexion_r", "knee_angle_r", "ankle_angle_r"]
JOINT_LABELS = ["Hip flexion", "Knee", "Ankle"]


def _norm(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(col).lower())


def _find_column(columns: List[str], patterns: List[str]) -> Optional[str]:
    normalized = {_norm(c): c for c in columns}
    for pattern in patterns:
        key = _norm(pattern)
        if key in normalized:
            return normalized[key]
        for ncol, original in normalized.items():
            if key in ncol and "absolute" not in ncol:
                return original
    return None


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    return {
        "time": _find_column(cols, ["common_timestamp", "Time (s)", "t_s", "session_time_s", "time", "t"]),
        "acc_x": _find_column(cols, ["Acceleration x", "ax", "accx"]),
        "acc_y": _find_column(cols, ["Acceleration y", "ay", "accy"]),
        "acc_z": _find_column(cols, ["Acceleration z", "az", "accz"]),
        "gyro_x": _find_column(cols, ["Gyroscope x", "gx_alpha", "gyrox", "gx"]),
        "gyro_y": _find_column(cols, ["Gyroscope y", "gy_beta", "gyroy", "gy"]),
        "gyro_z": _find_column(cols, ["Gyroscope z", "gz_gamma", "gyroz", "gz"]),
        "accel_units": _find_column(cols, ["accel_units"]),
        "gyro_units": _find_column(cols, ["gyro_units"]),
        "device_id": _find_column(cols, ["device_id"]),
    }


def load_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    buffer = io.BytesIO(file_bytes)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    try:
        return pd.read_csv(buffer)
    except Exception:
        buffer.seek(0)
        return pd.read_csv(buffer, comment="#")


def estimate_rate_hz(time: np.ndarray) -> float:
    if len(time) < 2:
        return 0.0
    return float((len(time) - 1) / (time[-1] - time[0]))


def dataframe_to_phone_schema(
    df: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
    accel_units: str,
    gyro_units: str,
    target_hz: Optional[float] = None,
) -> Tuple[pd.DataFrame, dict]:
    missing = [k for k in ("time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z") if not mapping.get(k)]
    if missing:
        raise ValueError(f"Missing required columns for: {', '.join(missing)}")

    time = df[mapping["time"]].to_numpy(dtype=float)
    imu = np.column_stack(
        [
            df[mapping["acc_x"]].to_numpy(dtype=float),
            df[mapping["acc_y"]].to_numpy(dtype=float),
            df[mapping["acc_z"]].to_numpy(dtype=float),
            df[mapping["gyro_x"]].to_numpy(dtype=float),
            df[mapping["gyro_y"]].to_numpy(dtype=float),
            df[mapping["gyro_z"]].to_numpy(dtype=float),
        ]
    )
    valid = np.isfinite(time) & np.all(np.isfinite(imu), axis=1)
    time, imu = time[valid], imu[valid]
    order = np.argsort(time)
    time, imu = time[order], imu[order]
    keep = np.concatenate([[True], np.diff(time) > 0])
    time, imu = time[keep], imu[keep]
    if len(time) < 2:
        raise ValueError("Need at least 2 valid IMU samples")

    native_hz = estimate_rate_hz(time)
    resampled = False
    if target_hz is not None:
        lo, hi = 0.9 * target_hz, 1.1 * target_hz
        if not (lo <= native_hz <= hi):
            n_out = max(int(np.floor((time[-1] - time[0]) * target_hz)) + 1, 2)
            t_new = time[0] + np.arange(n_out) / target_hz
            imu = np.column_stack([np.interp(t_new, time, imu[:, i]) for i in range(6)])
            time = t_new
            resampled = True

    out = pd.DataFrame(
        {
            "common_timestamp": time,
            "ax": imu[:, 0],
            "ay": imu[:, 1],
            "az": imu[:, 2],
            "gx": imu[:, 3],
            "gy": imu[:, 4],
            "gz": imu[:, 5],
            "accel_units": accel_units,
            "gyro_units": gyro_units,
        }
    )
    return out, {
        "native_hz": native_hz,
        "output_hz": estimate_rate_hz(time),
        "resampled": resampled,
        "samples": len(out),
    }


def split_combined(df: pd.DataFrame, device_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ids = df[device_col].astype(str).str.lower()
    thigh = df[ids.str.contains("phone_1") | ids.str.contains("thigh")]
    shank = df[ids.str.contains("phone_2") | ids.str.contains("shank")]
    if thigh.empty or shank.empty:
        uniq = sorted(df[device_col].astype(str).unique().tolist())
        raise ValueError(
            "Combined file needs both thigh (phone_1) and shank (phone_2) rows. "
            f"Found device_id values: {uniq}"
        )
    return thigh.copy(), shank.copy()


def write_session(
    session_dir: Path,
    thigh_df: pd.DataFrame,
    shank_df: pd.DataFrame,
    start: Optional[float] = None,
) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    thigh_df.to_csv(session_dir / "phone_1.csv", index=False)
    shank_df.to_csv(session_dir / "phone_2.csv", index=False)
    if start is None:
        start = float(max(thigh_df["common_timestamp"].iloc[0], shank_df["common_timestamp"].iloc[0]))
    (session_dir / "metadata.json").write_text(json.dumps({"start": start}, indent=2) + "\n")
    return session_dir


def ensure_package_imports(package_dir: Path) -> Dict[str, object]:
    """Import package modules with package-local resolution; avoid 60/100 Hz collisions."""
    inserted = False
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
        inserted = True
    try:
        for name in ("model", "classifier", "preprocessing", "predict"):
            sys.modules.pop(name, None)
        modules: Dict[str, object] = {"model": importlib.import_module("model")}
        if (package_dir / "preprocessing.py").is_file():
            modules["preprocessing"] = importlib.import_module("preprocessing")
        if (package_dir / "classifier.py").is_file():
            modules["classifier"] = importlib.import_module("classifier")
        modules["predict"] = importlib.import_module("predict")
        return modules
    finally:
        if inserted and sys.path and sys.path[0] == str(package_dir):
            sys.path.pop(0)


@st.cache_resource
def load_60hz_bundle(package_dir_str: str):
    package_dir = Path(package_dir_str)
    mods = ensure_package_imports(package_dir)
    predict_mod = mods["predict"]
    BioMAT = mods["model"].BioMAT
    saved = torch.load(package_dir / "model.pt", map_location="cpu", weights_only=True)
    model = BioMAT(seq_len=predict_mod.LENGTH, features=12)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return {
        "hz": 60,
        "length": predict_mod.LENGTH,
        "prepare_phone": predict_mod.prepare_phone,
        "predict_fn": predict_mod.predict,
        "model": model,
        "mean": saved["mean"].numpy(),
        "scale": saved["scale"].numpy(),
        "targets": list(predict_mod.TARGETS),
        "has_classifier": False,
    }


@st.cache_resource
def load_100hz_bundle(package_dir_str: str):
    package_dir = Path(package_dir_str)
    mods = ensure_package_imports(package_dir)
    predict_mod = mods["predict"]
    prep = mods["preprocessing"]
    return {
        "hz": 100,
        "length": prep.LENGTH,
        "prepare_phone": prep.prepare_phone,
        "predict_fn": prep.predict,
        "models": predict_mod.load_models(),
        "targets": list(prep.TARGETS),
        "classes": list(prep.CLASSES),
        "has_classifier": True,
    }


def run_inference(bundle: dict, session_dir: Path) -> dict:
    torch.set_num_threads(4)
    x, starts, timeline, usable, info = bundle["prepare_phone"](session_dir)
    if bundle["hz"] == 60:
        predictions = bundle["predict_fn"](
            bundle["model"], x, bundle["mean"], bundle["scale"], "cpu", "regressor"
        )
        activities = None
    else:
        outputs = {
            task: bundle["predict_fn"](model, x, mean, scale, "cpu", task)
            for task, (model, mean, scale) in bundle["models"].items()
        }
        predictions = outputs["regressor"]
        probabilities = outputs["classifier"]
        labels = probabilities.argmax(1)
        activities = pd.DataFrame(
            {
                "start_session_s": timeline[starts],
                "end_session_s": timeline[starts] + (bundle["length"] - 1) / bundle["hz"],
                "usable_motion_window": usable,
                "candidate_activity": [
                    bundle["classes"][i] if ok else "not_evaluated_low_motion_or_unstable_axis"
                    for i, ok in zip(labels, usable)
                ],
                "uncalibrated_model_score": np.where(usable, probabilities.max(1), np.nan),
            }
        )
        for i, name in enumerate(bundle["classes"]):
            activities[name + "_score"] = np.where(usable, probabilities[:, i], np.nan)

    length = bundle["length"]
    sums = np.zeros((len(timeline), 3))
    counts = np.zeros(len(timeline), dtype=int)
    for k, start in enumerate(starts):
        if usable[k]:
            sums[start : start + length] += predictions[k]
            counts[start : start + length] += 1
    average = np.full_like(sums, np.nan)
    np.divide(sums, counts[:, None], out=average, where=counts[:, None] > 0)
    angles = pd.DataFrame({"session_time_s": timeline, "contributing_windows": counts})
    for j, name in enumerate(bundle["targets"]):
        angles[name + "_degrees_unvalidated"] = average[:, j]

    return {
        "angles": angles,
        "activities": activities,
        "usable": usable,
        "windows": len(x),
        "accepted": int(usable.sum()),
        "info": info,
    }


def plot_angles(angles: pd.DataFrame, flip_knee: bool) -> go.Figure:
    t = angles["session_time_s"].to_numpy()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=JOINT_LABELS, vertical_spacing=0.06)
    for i, name in enumerate(TARGETS):
        y = angles[name + "_degrees_unvalidated"].to_numpy(dtype=float)
        if flip_knee and name == "knee_angle_r":
            y = -y
        fig.add_trace(
            go.Scatter(x=t, y=y, mode="lines", name=JOINT_LABELS[i], showlegend=False),
            row=i + 1,
            col=1,
        )
        fig.update_yaxes(title_text="deg", row=i + 1, col=1)
    fig.update_xaxes(title_text="session_time_s", row=3, col=1)
    fig.update_layout(
        height=720,
        margin=dict(l=40, r=20, t=40, b=40),
        title="Joint angles (accepted windows only)",
    )
    return fig


IMU_AXIS_COLORS = {"x": "#1f77b4", "y": "#ff7f0e", "z": "#2ca02c"}


def _phone_time_axis(df: pd.DataFrame) -> np.ndarray:
    t = df["common_timestamp"].to_numpy(dtype=float)
    return t - float(t[0])


def plot_imu_site(df: pd.DataFrame, site_label: str, accel_units: str, gyro_units: str) -> go.Figure:
    """Plot accel + gyro XYZ for one phone (schema: common_timestamp, ax..gz)."""
    t = _phone_time_axis(df)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(f"{site_label} acceleration", f"{site_label} gyroscope"),
        vertical_spacing=0.08,
    )
    for axis in ("x", "y", "z"):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=df[f"a{axis}"].to_numpy(dtype=float),
                mode="lines",
                name=f"a{axis}",
                legendgroup=axis,
                line=dict(color=IMU_AXIS_COLORS[axis], width=1),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=df[f"g{axis}"].to_numpy(dtype=float),
                mode="lines",
                name=f"g{axis}",
                legendgroup=axis,
                showlegend=False,
                line=dict(color=IMU_AXIS_COLORS[axis], width=1),
            ),
            row=2,
            col=1,
        )
    fig.update_yaxes(title_text=accel_units, row=1, col=1)
    fig.update_yaxes(title_text=gyro_units, row=2, col=1)
    fig.update_xaxes(title_text="session_time_s", row=2, col=1)
    fig.update_layout(
        height=480,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        title=f"{site_label} IMU (input to model session)",
    )
    return fig


def load_session_phones(session_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    thigh = pd.read_csv(session_dir / "phone_1.csv")
    shank = pd.read_csv(session_dir / "phone_2.csv")
    accel_u = str(thigh["accel_units"].iloc[0]) if "accel_units" in thigh.columns else "accel"
    gyro_u = str(thigh["gyro_units"].iloc[0]) if "gyro_units" in thigh.columns else "gyro"
    return thigh, shank, accel_u, gyro_u


def combined_imu_table(thigh: pd.DataFrame, shank: pd.DataFrame) -> pd.DataFrame:
    """One table: overlapping thigh+shank accel/gyro on a shared time base."""
    t_start = max(float(thigh["common_timestamp"].iloc[0]), float(shank["common_timestamp"].iloc[0]))
    t_stop = min(float(thigh["common_timestamp"].iloc[-1]), float(shank["common_timestamp"].iloc[-1]))
    if not np.isfinite(t_start) or not np.isfinite(t_stop) or t_stop <= t_start:
        raise ValueError("No overlapping thigh/shank timestamps for IMU table")

    th = thigh[
        (thigh["common_timestamp"] >= t_start) & (thigh["common_timestamp"] <= t_stop)
    ].copy()
    t = th["common_timestamp"].to_numpy(dtype=float)
    out: Dict[str, np.ndarray] = {"session_time_s": t - t_start}
    channels = ("ax", "ay", "az", "gx", "gy", "gz")
    for site, df in (("thigh", thigh), ("shank", shank)):
        src_t = df["common_timestamp"].to_numpy(dtype=float)
        for ch in channels:
            out[f"{site}_{ch}"] = np.interp(t, src_t, df[ch].to_numpy(dtype=float))
    return pd.DataFrame(out)


def is_sensorlog_folder(path: Path) -> bool:
    return path.is_dir() and (path / "Accelerometer.csv").is_file() and (path / "Gyroscope.csv").is_file()


def find_sensorlog_root(extracted: Path) -> Path:
    """Return folder that contains Accelerometer.csv (+ Gyroscope.csv)."""
    if is_sensorlog_folder(extracted):
        return extracted
    matches = [
        p
        for p in extracted.rglob("Accelerometer.csv")
        if (p.parent / "Gyroscope.csv").is_file()
    ]
    if not matches:
        raise ValueError(
            "Sensor Logger folder/zip must include Accelerometer.csv and Gyroscope.csv "
            "(Gravity.csv optional)."
        )
    if len(matches) > 1:
        # Prefer shallowest match
        matches.sort(key=lambda p: len(p.parts))
    return matches[0].parent


def extract_sensorlog_zip(file_bytes: bytes, label: str) -> Path:
    dest = Path(tempfile.mkdtemp(prefix=f"sensorlog_{label}_"))
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        zf.extractall(dest)
    return find_sensorlog_root(dest)


def _mot_header_token(line: str) -> str:
    """First field of an OpenSim/OpenCap header line (handles trailing CSV commas)."""
    return line.strip().split(",")[0].strip().strip('"').lower()


def load_opensim_mot(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse OpenSim/OpenCap .mot / .mot.csv (endheader) or plain CSV with joint columns."""
    text = file_bytes.decode("utf-8-sig", errors="replace")
    lower_name = filename.lower()
    lines = text.splitlines()

    header_end = None
    for i, line in enumerate(lines):
        if _mot_header_token(line) == "endheader":
            header_end = i
            break

    if header_end is not None:
        body = "\n".join(lines[header_end + 1 :])
        # OpenCap writes comma-separated .mot.csv; classic .mot is whitespace.
        if "," in body.split("\n", 1)[0]:
            df = pd.read_csv(io.StringIO(body))
        else:
            df = pd.read_csv(io.StringIO(body), sep=r"\s+", engine="python")
    elif "endheader" in text.lower() or lower_name.endswith((".mot", ".mot.csv")):
        # Header marker present but malformed; still try skipping until a time, column row.
        start = 0
        for i, line in enumerate(lines):
            tok = _mot_header_token(line)
            if tok == "time" or tok.startswith("time"):
                start = i
                break
        body = "\n".join(lines[start:])
        df = pd.read_csv(io.StringIO(body)) if "," in body.split("\n", 1)[0] else pd.read_csv(
            io.StringIO(body), sep=r"\s+", engine="python"
        )
    else:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")

    df.columns = [str(c).strip() for c in df.columns]
    # Drop unnamed padding columns from OpenCap CSV exports
    df = df.loc[:, [c for c in df.columns if not str(c).lower().startswith("unnamed")]]

    time_col = _find_column(list(df.columns), ["time", "Time", "t", "session_time_s"])
    hip_col = _find_column(list(df.columns), ["hip_flexion_r", "hip flexion r", "hip_flexion"])
    knee_col = _find_column(list(df.columns), ["knee_angle_r", "knee angle r", "knee_angle"])
    ankle_col = _find_column(list(df.columns), ["ankle_angle_r", "ankle angle r", "ankle_angle"])
    missing = [
        name
        for name, col in (
            ("time", time_col),
            ("hip_flexion_r", hip_col),
            ("knee_angle_r", knee_col),
            ("ankle_angle_r", ankle_col),
        )
        if col is None
    ]
    if missing:
        raise ValueError(
            f"Ground-truth file missing required columns: {', '.join(missing)}. "
            f"Found: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "time": pd.to_numeric(df[time_col], errors="coerce"),
            "hip_flexion_r": pd.to_numeric(df[hip_col], errors="coerce"),
            "knee_angle_r": pd.to_numeric(df[knee_col], errors="coerce"),
            "ankle_angle_r": pd.to_numeric(df[ankle_col], errors="coerce"),
        }
    ).dropna()
    if len(out) < 2:
        raise ValueError("Ground-truth file has fewer than 2 valid rows")
    out = out.sort_values("time").drop_duplicates("time", keep="first").reset_index(drop=True)
    # Normalize to session-relative time starting at 0
    out["time"] = out["time"].to_numpy(dtype=float) - float(out["time"].iloc[0])
    return out


def _signed_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 20:
        return 0.0
    x = a[mask]
    y = b[mask]
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return float(np.mean(x * y))


def resolve_joint_sign_flips(
    angles: pd.DataFrame,
    gt: pd.DataFrame,
    gt_time_offset_s: float = 0.0,
    force_flip_pred: Optional[Dict[str, bool]] = None,
) -> Dict[str, bool]:
    """Choose whether to negate prediction per joint so it matches GT (not both).

    OpenCap knee often has the opposite sign of BioMAT/Camargo outputs. Flipping
    both leaves them opposed; we only flip the prediction when -pred correlates
    better with GT.
    """
    t_pred = angles["session_time_s"].to_numpy(dtype=float)
    pred_ok = angles["contributing_windows"].to_numpy(dtype=int) > 0
    t_gt = gt["time"].to_numpy(dtype=float) + float(gt_time_offset_s)
    flips: Dict[str, bool] = {}
    for name in TARGETS:
        if force_flip_pred is not None and name in force_flip_pred:
            flips[name] = bool(force_flip_pred[name])
            continue
        pred = angles[name + "_degrees_unvalidated"].to_numpy(dtype=float)
        truth = np.interp(t_pred, t_gt, gt[name].to_numpy(dtype=float), left=np.nan, right=np.nan)
        pred_m = np.where(pred_ok, pred, np.nan)
        c_pos = _signed_corr(pred_m, truth)
        c_neg = _signed_corr(-pred_m, truth)
        flips[name] = c_neg > c_pos + 0.05  # require a clear improvement
    return flips


def align_prediction_to_gt(
    angles: pd.DataFrame,
    gt: pd.DataFrame,
    flip_knee: bool,
    gt_time_offset_s: float = 0.0,
    pred_sign_flips: Optional[Dict[str, bool]] = None,
) -> pd.DataFrame:
    """Interpolate GT onto prediction timeline; keep only samples with predictions.

    ``pred_sign_flips`` negates prediction only (GT stays OpenSim-native).
    Legacy ``flip_knee`` alone is ignored for validation overlays — use sign flips.
    """
    del flip_knee  # display-only checkbox; validation uses pred_sign_flips
    t_pred = angles["session_time_s"].to_numpy(dtype=float)
    pred_ok = angles["contributing_windows"].to_numpy(dtype=int) > 0
    t_gt = gt["time"].to_numpy(dtype=float) + float(gt_time_offset_s)
    if pred_sign_flips is None:
        pred_sign_flips = resolve_joint_sign_flips(angles, gt, gt_time_offset_s)

    rows = {
        "session_time_s": t_pred,
        "contributing_windows": angles["contributing_windows"].to_numpy(dtype=int),
    }
    for name in TARGETS:
        pred = angles[name + "_degrees_unvalidated"].to_numpy(dtype=float)
        truth = np.interp(t_pred, t_gt, gt[name].to_numpy(dtype=float), left=np.nan, right=np.nan)
        if pred_sign_flips.get(name, False):
            pred = -pred
        rows[f"{name}_pred"] = pred
        rows[f"{name}_gt"] = truth
        err = pred - truth
        err[~pred_ok] = np.nan
        rows[f"{name}_error"] = err
    return pd.DataFrame(rows)


def comparison_metrics(compare: pd.DataFrame) -> pd.DataFrame:
    records = []
    for name, label in zip(TARGETS, JOINT_LABELS):
        err = compare[f"{name}_error"].to_numpy(dtype=float)
        valid = np.isfinite(err)
        if not valid.any():
            records.append({"joint": label, "n": 0, "RMSE_deg": np.nan, "MAE_deg": np.nan})
            continue
        e = err[valid]
        records.append(
            {
                "joint": label,
                "n": int(valid.sum()),
                "RMSE_deg": float(np.sqrt(np.mean(e**2))),
                "MAE_deg": float(np.mean(np.abs(e))),
            }
        )
    return pd.DataFrame(records)


def estimate_gt_offset_from_hip(
    angles: pd.DataFrame,
    gt: pd.DataFrame,
    max_lag_s: float = 5.0,
    step_s: float = 0.01,
) -> Tuple[float, dict]:
    """Estimate GT time offset by matching hip flexion peaks + refining with correlation.

    Positive offset adds to GT time (shifts GT later on the prediction clock).
    """
    t_pred = angles["session_time_s"].to_numpy(dtype=float)
    pred_ok = angles["contributing_windows"].to_numpy(dtype=int) > 0
    pred_hip = angles["hip_flexion_r_degrees_unvalidated"].to_numpy(dtype=float)
    t_gt = gt["time"].to_numpy(dtype=float)
    gt_hip = gt["hip_flexion_r"].to_numpy(dtype=float)

    if pred_ok.sum() < 20 or len(t_gt) < 20:
        return 0.0, {"method": "none", "reason": "too few samples", "corr": np.nan}

    # Coarse peak-based seed: align first strong hip peaks in the accepted pred region.
    pred_t = t_pred[pred_ok]
    pred_h = pred_hip[pred_ok]
    prom_pred = max(5.0, 0.25 * float(np.nanstd(pred_h)))
    prom_gt = max(5.0, 0.25 * float(np.nanstd(gt_hip)))
    pred_peaks, _ = find_peaks(pred_h, prominence=prom_pred, distance=max(3, int(0.3 / max(np.median(np.diff(pred_t)), 1e-3))))
    gt_peaks, _ = find_peaks(gt_hip, prominence=prom_gt, distance=max(3, int(0.3 / max(np.median(np.diff(t_gt)), 1e-3))))

    peak_offset = 0.0
    peak_note = "no peaks"
    if len(pred_peaks) and len(gt_peaks):
        # Match each pred peak to nearest GT peak; use median lag
        lags = []
        for pi in pred_peaks:
            tp = float(pred_t[pi])
            j = int(np.argmin(np.abs(t_gt[gt_peaks] - tp)))
            lags.append(tp - float(t_gt[gt_peaks[j]]))
        peak_offset = float(np.median(lags))
        peak_note = f"{len(pred_peaks)} pred peaks, {len(gt_peaks)} GT peaks"

    # Refine with normalized cross-correlation around the peak seed (and also around 0).
    search_centers = sorted({0.0, peak_offset})
    best_offset, best_corr = peak_offset, -np.inf
    for center in search_centers:
        lo = max(-max_lag_s, center - 1.5)
        hi = min(max_lag_s, center + 1.5)
        for offset in np.arange(lo, hi + 1e-12, step_s):
            truth = np.interp(t_pred, t_gt + offset, gt_hip, left=np.nan, right=np.nan)
            mask = pred_ok & np.isfinite(truth) & np.isfinite(pred_hip)
            if int(mask.sum()) < 50:
                continue
            p = pred_hip[mask]
            g = truth[mask]
            p = (p - p.mean()) / (p.std() + 1e-8)
            g = (g - g.mean()) / (g.std() + 1e-8)
            corr = float(np.mean(p * g))
            if corr > best_corr:
                best_corr = corr
                best_offset = float(offset)

    return best_offset, {
        "method": "hip_peaks+xcorr",
        "peak_offset_s": peak_offset,
        "peak_note": peak_note,
        "corr": best_corr,
        "offset_s": best_offset,
    }


JOINT_PLOT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def plot_angles_vs_gt(
    compare: pd.DataFrame,
    gt_time_offset_s: float = 0.0,
    pred_sign_flips: Optional[Dict[str, bool]] = None,
) -> go.Figure:
    t = compare["session_time_s"].to_numpy(dtype=float)
    pred_sign_flips = pred_sign_flips or {}
    titles = [
        f"{label}" + (" (pred sign inverted to match GT)" if pred_sign_flips.get(name, False) else "")
        for name, label in zip(TARGETS, JOINT_LABELS)
    ]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=titles, vertical_spacing=0.06)
    for i, name in enumerate(TARGETS):
        color = JOINT_PLOT_COLORS[i]
        fig.add_trace(
            go.Scatter(
                x=t,
                y=compare[f"{name}_gt"],
                mode="lines",
                name=f"{JOINT_LABELS[i]} GT",
                line=dict(color=color, width=2, dash="dot"),
                legendgroup=f"joint{i}",
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=compare[f"{name}_pred"],
                mode="lines",
                name=f"{JOINT_LABELS[i]} pred",
                line=dict(color=color, width=2, dash="solid"),
                legendgroup=f"joint{i}",
            ),
            row=i + 1,
            col=1,
        )
        fig.update_yaxes(title_text="deg", row=i + 1, col=1)
    fig.update_xaxes(title_text="session_time_s", row=3, col=1)
    fig.update_layout(
        height=780,
        margin=dict(l=40, r=20, t=50, b=40),
        title=f"Validation overlay · GT dotted · pred solid · offset {gt_time_offset_s:+.3f}s",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def render_validation_overlay(
    angles: pd.DataFrame,
    gt: pd.DataFrame,
    flip_knee: bool,
    gt_time_offset_s: float,
) -> None:
    del flip_knee  # validation uses per-joint pred-only sign matching vs GT
    st.subheader("Validation vs ground truth")
    st.caption(
        "Solid = prediction · dotted = OpenSim GT. "
        "Offset updates live. Knee/hip/ankle: if OpenCap sign disagrees with the model, "
        "**only the prediction** is inverted (GT stays native)."
    )

    align_cols = st.columns([1, 2])
    with align_cols[0]:
        if st.button("Auto-align hip peaks", key="btn_hip_align"):
            offset, info = estimate_gt_offset_from_hip(angles, gt)
            st.session_state["_pending_hip_align_offset"] = float(offset)
            st.session_state["validation_align_info"] = info
            st.rerun()
    with align_cols[1]:
        info = st.session_state.get("validation_align_info")
        if isinstance(info, dict) and info.get("offset_s") is not None:
            corr = info.get("corr")
            corr_txt = f"{corr:.3f}" if corr is not None and np.isfinite(corr) else "n/a"
            st.caption(
                f"Last auto-align: offset **{float(info['offset_s']):+.3f}s** · "
                f"hip corr {corr_txt} · {info.get('peak_note', '')}"
            )

    auto_flips = resolve_joint_sign_flips(angles, gt, gt_time_offset_s)
    run_id = None
    cache = st.session_state.get("validation_run_cache")
    if isinstance(cache, dict):
        run_id = cache.get("run_id")
    if st.session_state.get("_knee_sign_run_id") != run_id:
        st.session_state["force_invert_pred_knee"] = bool(auto_flips.get("knee_angle_r", False))
        st.session_state["_knee_sign_run_id"] = run_id

    # Manual override for knee (common OpenCap vs BioMAT sign mismatch)
    force_knee = st.checkbox(
        "Invert prediction knee only (match OpenCap GT)",
        key="force_invert_pred_knee",
        help="OpenCap knee is often opposite BioMAT. This negates prediction only — GT stays as in the .mot file.",
    )
    pred_sign_flips = dict(auto_flips)
    pred_sign_flips["knee_angle_r"] = bool(force_knee)
    # Keep auto decisions for hip/ankle unless clearly wrong
    flipped = [JOINT_LABELS[i] for i, name in enumerate(TARGETS) if pred_sign_flips.get(name)]
    if flipped:
        st.info(f"Prediction sign inverted to match GT for: **{', '.join(flipped)}**")
    else:
        st.caption("No prediction sign inversions applied.")

    compare = align_prediction_to_gt(
        angles,
        gt,
        flip_knee=False,
        gt_time_offset_s=gt_time_offset_s,
        pred_sign_flips=pred_sign_flips,
    )
    metrics = comparison_metrics(compare)

    st.markdown("**RMSE (accepted prediction samples)**")
    mcols = st.columns(4)
    for i, (_, row) in enumerate(metrics.iterrows()):
        with mcols[i]:
            rmse = row["RMSE_deg"]
            mae = row["MAE_deg"]
            st.metric(
                f"{row['joint']} RMSE",
                "—" if not np.isfinite(rmse) else f"{rmse:.2f}°",
                help=(
                    f"n={int(row['n'])} · MAE={mae:.2f}°"
                    if np.isfinite(mae)
                    else f"n={int(row['n'])}"
                ),
            )
    with mcols[3]:
        all_err = np.concatenate(
            [compare[f"{name}_error"].to_numpy(dtype=float) for name in TARGETS]
        )
        all_err = all_err[np.isfinite(all_err)]
        overall = float(np.sqrt(np.mean(all_err**2))) if len(all_err) else np.nan
        st.metric("Overall RMSE", "—" if not np.isfinite(overall) else f"{overall:.2f}°")

    st.dataframe(metrics, use_container_width=True)
    st.plotly_chart(
        plot_angles_vs_gt(
            compare,
            gt_time_offset_s=gt_time_offset_s,
            pred_sign_flips=pred_sign_flips,
        ),
        use_container_width=True,
    )
    st.dataframe(compare, use_container_width=True)
    flip_key = "-".join(f"{k}{int(v)}" for k, v in sorted(pred_sign_flips.items()))
    st.download_button(
        "Download validation_compare.csv",
        compare.to_csv(index=False).encode("utf-8"),
        file_name="validation_compare.csv",
        mime="text/csv",
        key=f"dl_val_compare_{gt_time_offset_s:.4f}_{flip_key}",
    )


def mapping_ui(df: pd.DataFrame, detected: Dict[str, Optional[str]], key_prefix: str) -> Dict[str, Optional[str]]:
    cols = ["—"] + list(df.columns)
    keys = [
        ("time", "Time"),
        ("acc_x", "Accel X"),
        ("acc_y", "Accel Y"),
        ("acc_z", "Accel Z"),
        ("gyro_x", "Gyro X"),
        ("gyro_y", "Gyro Y"),
        ("gyro_z", "Gyro Z"),
    ]
    mapping: Dict[str, Optional[str]] = {}
    grid = st.columns(4)
    for i, (key, label) in enumerate(keys):
        default = detected.get(key)
        index = cols.index(default) if default in cols else 0
        with grid[i % 4]:
            choice = st.selectbox(f"{key_prefix}: {label}", cols, index=index, key=f"{key_prefix}_{key}")
            mapping[key] = None if choice == "—" else choice
    return mapping


def unit_from_df(df: pd.DataFrame, detected: Dict[str, Optional[str]], key: str, fallback: str) -> str:
    col = detected.get(key)
    if col and col in df.columns:
        return str(df[col].iloc[0])
    return fallback


def main():
    st.set_page_config(page_title="BioMAT Shank+Thigh", layout="wide")
    st.title("BioMAT shank + thigh")
    st.caption(
        "Uses the **NEW** standalone 60/100 Hz shank+thigh weights "
        "(not the old `caches/trained_model` checkpoints from `biomat_app.py`). "
        "Pick a Sensor Logger pair from `sensorlogData`, or upload CSVs."
    )
    st.info(
        "Preprocessing matches training: each window goes through `canonicalize()` "
        "(median accel norm + signal-derived rotation), then the **checkpoint** "
        "`mean`/`scale` — never a scaler refit on phone data. "
        "Raw IMU → these models without that step hurts accuracy."
    )

    available = {label: path for label, path in PACKAGE_DIRS.items() if (path / "model.pt").is_file()}
    if not available:
        st.error("No shank/thigh packages found next to the biomat folder.")
        return

    with st.sidebar:
        st.header("Model")
        model_label = st.selectbox("Checkpoint", list(available.keys()))
        package_dir = available[model_label]
        target_hz = 60 if model_label.startswith("60") else 100
        st.caption(f"NEW weights · `{package_dir.name}`")

        st.header("Input mode")
        input_mode = st.radio(
            "Files",
            [
                "Validation Test",
                "sensorlogData library",
                "Two files (thigh + shank)",
                "One combined CSV (device_id)",
                "Bundled 60 Hz example",
            ],
            index=0,
        )
        force_resample = st.checkbox(
            f"Resample to {target_hz} Hz if needed",
            value=True,
            help="Official CLI rejects wrong rates. Keep this on for phone exports that are not exactly 60/100 Hz.",
        )
        # Validation compares to OpenSim .mot — keep native signs (don't invert knee).
        if input_mode == "Validation Test":
            if st.session_state.get("_flip_knee_mode") != "validation":
                st.session_state["flip_knee_checkbox"] = False
                st.session_state["_flip_knee_mode"] = "validation"
        elif st.session_state.get("_flip_knee_mode") != "other":
            st.session_state["flip_knee_checkbox"] = True
            st.session_state["_flip_knee_mode"] = "other"
        flip_knee = st.checkbox(
            "Flip knee sign (display flexion +)",
            value=input_mode != "Validation Test",
            help="Off for Validation Test so pred matches OpenSim GT signs. On for phone-only viewing.",
            key="flip_knee_checkbox",
        )
        gt_time_offset_s = 0.0
        if input_mode == "Validation Test":
            if flip_knee:
                st.caption(
                    "Knee flip is on — both pred and GT are negated for display. "
                    "Turn it off to match OpenSim/OpenCap knee sign."
                )
            if "_pending_hip_align_offset" in st.session_state:
                st.session_state["_gt_offset_slider"] = float(
                    st.session_state.pop("_pending_hip_align_offset")
                )
            gt_time_offset_s = st.slider(
                "GT time offset (s)",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.01,
                help="Shifts ground truth on the time axis. Auto-align uses hip flexion peaks; slider stays live.",
                key="_gt_offset_slider",
            )
            if target_hz != 100:
                st.warning("Validation Test is intended for 100 Hz Sensor Logger + 100 Hz model.")

        st.header("Units (if not in CSV)")
        accel_units = st.selectbox("Acceleration units", ["m/s^2", "g"], index=0)
        gyro_units = st.selectbox("Gyroscope units", ["rad/s", "deg/s"], index=0)

    session_dir: Optional[Path] = None
    prep_notes: List[dict] = []
    ready = False
    selected_pair = None
    validation_gt: Optional[pd.DataFrame] = None

    if input_mode == "Validation Test":
        st.subheader("Validation Test")
        st.caption(
            "Upload **thigh** and **shank** Sensor Logger folders as zips (or local paths), "
            "plus an OpenSim **`.mot` / `.mot.csv`** ground-truth file with "
            "`time`, `hip_flexion_r`, `knee_angle_r`, `ankle_angle_r`."
        )
        c1, c2 = st.columns(2)
        with c1:
            thigh_zip = st.file_uploader(
                "Thigh Sensor Logger zip (phone_1)",
                type=["zip"],
                key="val_thigh_zip",
            )
            thigh_path_str = st.text_input(
                "Or local thigh folder path",
                value="",
                key="val_thigh_path",
                placeholder=r"C:\...\walking_phone_1_trial_1-...",
            )
        with c2:
            shank_zip = st.file_uploader(
                "Shank Sensor Logger zip (phone_2)",
                type=["zip"],
                key="val_shank_zip",
            )
            shank_path_str = st.text_input(
                "Or local shank folder path",
                value="",
                key="val_shank_path",
                placeholder=r"C:\...\walking_phone_2_trial_1-...",
            )
        mot_upload = st.file_uploader(
            "Ground truth OpenSim file (.mot / .mot.csv / .csv)",
            type=["mot", "csv"],
            key="val_mot",
        )
        thigh_ready = thigh_zip is not None or bool(thigh_path_str.strip())
        shank_ready = shank_zip is not None or bool(shank_path_str.strip())
        if mot_upload is not None:
            try:
                validation_gt = load_opensim_mot(mot_upload.getvalue(), mot_upload.name)
                gt_hz = estimate_rate_hz(validation_gt["time"].to_numpy())
                gt_dur = float(validation_gt["time"].iloc[-1])
                st.success(
                    f"Ground truth loaded · **{len(validation_gt)} samples** · "
                    f"~{gt_hz:.1f} Hz · **{gt_dur:.2f} s** full duration "
                    f"(time {float(validation_gt['time'].iloc[0]):.3f} → {gt_dur:.3f})"
                )
                st.caption(
                    "Table below is the full ground-truth series (scroll). "
                    "Earlier builds only previewed 8 rows (~0.1 s) — that was not the full file."
                )
                st.dataframe(validation_gt, use_container_width=True, height=280)
                gt_fig = make_subplots(
                    rows=3,
                    cols=1,
                    shared_xaxes=True,
                    subplot_titles=JOINT_LABELS,
                    vertical_spacing=0.06,
                )
                for i, name in enumerate(TARGETS):
                    gt_fig.add_trace(
                        go.Scatter(
                            x=validation_gt["time"],
                            y=validation_gt[name],
                            mode="lines",
                            name=JOINT_LABELS[i],
                            showlegend=False,
                        ),
                        row=i + 1,
                        col=1,
                    )
                    gt_fig.update_yaxes(title_text="deg", row=i + 1, col=1)
                gt_fig.update_xaxes(title_text="time (s)", row=3, col=1)
                gt_fig.update_layout(
                    height=520,
                    margin=dict(l=40, r=20, t=40, b=40),
                    title="Ground truth (full OpenSim trial)",
                )
                st.plotly_chart(gt_fig, use_container_width=True)
            except Exception as gt_exc:
                st.error(f"Could not parse ground-truth file: {gt_exc}")
                validation_gt = None
        if thigh_ready and shank_ready and validation_gt is not None:
            st.session_state["_build"] = {
                "mode": "validation",
                "thigh_zip": thigh_zip.getvalue() if thigh_zip is not None else None,
                "shank_zip": shank_zip.getvalue() if shank_zip is not None else None,
                "thigh_path": thigh_path_str.strip() or None,
                "shank_path": shank_path_str.strip() or None,
                "gt": validation_gt,
            }
            ready = True
    elif input_mode == "sensorlogData library":
        pairs = discover_pairs(DEFAULT_SENSORLOG_ROOT)
        st.write(f"Library: `{DEFAULT_SENSORLOG_ROOT}`")
        if not pairs:
            st.error("No paired phone_1/phone_2 Sensor Logger folders found in sensorlogData.")
        else:
            labels = [p.key for p in pairs]
            choice = st.selectbox("Paired trial (phone_1=thigh, phone_2=shank)", labels)
            selected_pair = next(p for p in pairs if p.key == choice)
            st.caption(
                f"Thigh: `{selected_pair.thigh.path.name}`  \n"
                f"Shank: `{selected_pair.shank.path.name}`"
            )
            st.session_state["_build"] = {"mode": "sensorlog", "pair_key": selected_pair.key}
            ready = True
    elif input_mode == "Bundled 60 Hz example":
        example = TESTING_MODEL / "biomat_shank_thigh" / "biomat_shank_thigh" / "example_g_rad_s"
        if not example.is_dir():
            st.error(f"Missing example session: {example}")
            return
        if target_hz != 60:
            st.warning("Bundled example is ~60 Hz. Prefer the 60 Hz model, or upload/resample your own files for 100 Hz.")
        session_dir = example
        prep_notes = [{"file": "bundled example_g_rad_s", "native_hz": 60.0, "output_hz": 60.0, "resampled": False, "samples": 0}]
        ready = True
    elif input_mode.startswith("Two"):
        c1, c2 = st.columns(2)
        with c1:
            thigh_upload = st.file_uploader("Thigh CSV (phone_1)", type=["csv", "xlsx", "xls"], key="thigh")
        with c2:
            shank_upload = st.file_uploader("Shank CSV (phone_2)", type=["csv", "xlsx", "xls"], key="shank")
        if thigh_upload and shank_upload:
            thigh_raw = load_table(thigh_upload.getvalue(), thigh_upload.name)
            shank_raw = load_table(shank_upload.getvalue(), shank_upload.name)
            st.subheader("Column mapping")
            thigh_map = mapping_ui(thigh_raw, detect_columns(thigh_raw), "Thigh")
            shank_map = mapping_ui(shank_raw, detect_columns(shank_raw), "Shank")
            thigh_det, shank_det = detect_columns(thigh_raw), detect_columns(shank_raw)
            st.session_state["_build"] = {
                "mode": "two",
                "thigh_raw": thigh_raw,
                "shank_raw": shank_raw,
                "thigh_map": thigh_map,
                "shank_map": shank_map,
                "t_acc": unit_from_df(thigh_raw, thigh_det, "accel_units", accel_units),
                "t_gyro": unit_from_df(thigh_raw, thigh_det, "gyro_units", gyro_units),
                "s_acc": unit_from_df(shank_raw, shank_det, "accel_units", accel_units),
                "s_gyro": unit_from_df(shank_raw, shank_det, "gyro_units", gyro_units),
            }
            ready = True
    else:
        combined_upload = st.file_uploader(
            "Combined CSV with device_id / phone_1+phone_2 rows",
            type=["csv", "xlsx", "xls"],
            key="combined",
        )
        if combined_upload:
            raw = load_table(combined_upload.getvalue(), combined_upload.name)
            det = detect_columns(raw)
            if not det.get("device_id"):
                st.error("Combined mode needs a device_id column (phone_1 / phone_2).")
            else:
                st.subheader("Column mapping")
                mapping = mapping_ui(raw, det, "Combined")
                st.session_state["_build"] = {
                    "mode": "combined",
                    "raw": raw,
                    "det": det,
                    "mapping": mapping,
                    "acc": unit_from_df(raw, det, "accel_units", accel_units),
                    "gyro": unit_from_df(raw, det, "gyro_units", gyro_units),
                }
                ready = True

    run_clicked = st.button("Run prediction", type="primary", disabled=not ready)
    validation_cache = st.session_state.get("validation_run_cache")
    use_validation_cache = (
        input_mode == "Validation Test"
        and (not run_clicked)
        and isinstance(validation_cache, dict)
        and "angles" in validation_cache
        and "gt" in validation_cache
    )

    if not run_clicked and not use_validation_cache:
        st.info(
            "Uses **NEW** 60/100 Hz shank+thigh weights. "
            "Validation Test: Sensor Logger thigh/shank folders + OpenSim `.mot.csv` ground truth. "
            "Or pick a `sensorlogData` pair / upload CSVs."
        )
        return

    if use_validation_cache:
        result = validation_cache["result"]
        session_dir = Path(validation_cache["session_dir"])
        prep_notes = validation_cache.get("prep_notes", [])
        imu_combined = validation_cache.get("imu_combined")
        accel_u = validation_cache.get("accel_u", "m/s^2")
        gyro_u = validation_cache.get("gyro_u", "rad/s")
        thigh_imu = validation_cache.get("thigh_imu")
        shank_imu = validation_cache.get("shank_imu")
        angles = validation_cache["angles"]
        st.info(
            "Showing last Validation Test run. "
            "Move **GT time offset** to realign overlays (no model re-run). "
            "Click **Run prediction** to recompute from new files."
        )
        for note in prep_notes:
            if note.get("samples"):
                st.caption(f"{note['file']}: ~{note.get('native_hz', 0):.1f} Hz, {note['samples']} samples")
        st.success(
            f"Cached run — {result['accepted']} / {result['windows']} windows accepted by motion gate."
        )
        if thigh_imu is not None and shank_imu is not None:
            st.subheader("Input IMU")
            c_thigh, c_shank = st.columns(2)
            with c_thigh:
                st.plotly_chart(
                    plot_imu_site(thigh_imu, "Thigh (phone_1)", accel_u, gyro_u),
                    use_container_width=True,
                )
            with c_shank:
                st.plotly_chart(
                    plot_imu_site(shank_imu, "Shank (phone_2)", accel_u, gyro_u),
                    use_container_width=True,
                )
        st.subheader("Predicted joint angles")
        st.plotly_chart(plot_angles(angles, flip_knee=flip_knee), use_container_width=True)
        try:
            render_validation_overlay(
                angles,
                validation_cache["gt"],
                flip_knee=flip_knee,
                gt_time_offset_s=float(gt_time_offset_s),
            )
        except Exception as cmp_exc:
            st.exception(cmp_exc)
        if result.get("activities") is not None:
            st.subheader("Activity windows (100 Hz classifier)")
            st.dataframe(result["activities"], use_container_width=True)
        if imu_combined is not None:
            st.subheader("Combined IMU table (thigh + shank)")
            st.dataframe(imu_combined, use_container_width=True)
        st.subheader("Angle table")
        st.dataframe(angles, use_container_width=True)
        return

    validation_compare_meta: Optional[dict] = None
    try:
        target = target_hz if force_resample else None
        if input_mode == "Bundled 60 Hz example":
            assert session_dir is not None
            if target_hz == 100 and force_resample:
                thigh_raw = pd.read_csv(session_dir / "phone_1.csv")
                shank_raw = pd.read_csv(session_dir / "phone_2.csv")
                det = detect_columns(thigh_raw)
                mapping = {
                    "time": det["time"],
                    "acc_x": det["acc_x"],
                    "acc_y": det["acc_y"],
                    "acc_z": det["acc_z"],
                    "gyro_x": det["gyro_x"],
                    "gyro_y": det["gyro_y"],
                    "gyro_z": det["gyro_z"],
                }
                t_acc = unit_from_df(thigh_raw, det, "accel_units", "g")
                t_gyro = unit_from_df(thigh_raw, det, "gyro_units", "rad/s")
                thigh_df, t_info = dataframe_to_phone_schema(thigh_raw, mapping, t_acc, t_gyro, target)
                shank_df, s_info = dataframe_to_phone_schema(shank_raw, mapping, t_acc, t_gyro, target)
                prep_notes = [{"file": "thigh", **t_info}, {"file": "shank", **s_info}]
                session_dir = write_session(Path(tempfile.mkdtemp(prefix="biomat_session_")), thigh_df, shank_df)
        else:
            build = st.session_state.get("_build")
            if not build:
                st.error("Select/upload data first.")
                return
            if build["mode"] == "validation":
                if build.get("thigh_zip"):
                    thigh_folder = extract_sensorlog_zip(build["thigh_zip"], "thigh")
                else:
                    thigh_folder = Path(build["thigh_path"]).expanduser().resolve()
                    if not is_sensorlog_folder(thigh_folder):
                        raise ValueError(f"Not a Sensor Logger folder: {thigh_folder}")
                if build.get("shank_zip"):
                    shank_folder = extract_sensorlog_zip(build["shank_zip"], "shank")
                else:
                    shank_folder = Path(build["shank_path"]).expanduser().resolve()
                    if not is_sensorlog_folder(shank_folder):
                        raise ValueError(f"Not a Sensor Logger folder: {shank_folder}")

                thigh_raw, t_meta = load_sensorlog_phone_df(thigh_folder)
                shank_raw, s_meta = load_sensorlog_phone_df(shank_folder)
                mapping = {
                    "time": "common_timestamp",
                    "acc_x": "ax",
                    "acc_y": "ay",
                    "acc_z": "az",
                    "gyro_x": "gx",
                    "gyro_y": "gy",
                    "gyro_z": "gz",
                }
                thigh_df, t_info = dataframe_to_phone_schema(
                    thigh_raw, mapping, "m/s^2", "rad/s", target
                )
                shank_df, s_info = dataframe_to_phone_schema(
                    shank_raw, mapping, "m/s^2", "rad/s", target
                )
                t_info = {**t_meta, **t_info, "file": "thigh(validation)"}
                s_info = {**s_meta, **s_info, "file": "shank(validation)"}
                prep_notes = [t_info, s_info]
                if t_meta.get("gravity_added") or s_meta.get("gravity_added"):
                    st.caption("Added Gravity.csv into Accelerometer (Sensor Logger linear accel → total accel).")
                validation_compare_meta = {"gt": build["gt"]}
                session_dir = write_session(Path(tempfile.mkdtemp(prefix="biomat_session_")), thigh_df, shank_df)
            elif build["mode"] == "sensorlog":
                pairs = discover_pairs(DEFAULT_SENSORLOG_ROOT)
                pair = next(p for p in pairs if p.key == build["pair_key"])
                thigh_raw, t_meta = load_sensorlog_phone_df(pair.thigh.path)
                shank_raw, s_meta = load_sensorlog_phone_df(pair.shank.path)
                mapping = {
                    "time": "common_timestamp",
                    "acc_x": "ax",
                    "acc_y": "ay",
                    "acc_z": "az",
                    "gyro_x": "gx",
                    "gyro_y": "gy",
                    "gyro_z": "gz",
                }
                thigh_df, t_info = dataframe_to_phone_schema(
                    thigh_raw, mapping, "m/s^2", "rad/s", target
                )
                shank_df, s_info = dataframe_to_phone_schema(
                    shank_raw, mapping, "m/s^2", "rad/s", target
                )
                t_info = {**t_meta, **t_info, "file": "thigh(sensorlog)"}
                s_info = {**s_meta, **s_info, "file": "shank(sensorlog)"}
                prep_notes = [t_info, s_info]
                if t_meta.get("gravity_added") or s_meta.get("gravity_added"):
                    st.caption("Added Gravity.csv into Accelerometer (Sensor Logger linear accel → total accel).")
                session_dir = write_session(Path(tempfile.mkdtemp(prefix="biomat_session_")), thigh_df, shank_df)
            elif build["mode"] == "two":
                thigh_df, t_info = dataframe_to_phone_schema(
                    build["thigh_raw"], build["thigh_map"], build["t_acc"], build["t_gyro"], target
                )
                shank_df, s_info = dataframe_to_phone_schema(
                    build["shank_raw"], build["shank_map"], build["s_acc"], build["s_gyro"], target
                )
                prep_notes = [{"file": "thigh", **t_info}, {"file": "shank", **s_info}]
                session_dir = write_session(Path(tempfile.mkdtemp(prefix="biomat_session_")), thigh_df, shank_df)
            else:
                thigh_raw, shank_raw = split_combined(build["raw"], build["det"]["device_id"])
                thigh_df, t_info = dataframe_to_phone_schema(
                    thigh_raw, build["mapping"], build["acc"], build["gyro"], target
                )
                shank_df, s_info = dataframe_to_phone_schema(
                    shank_raw, build["mapping"], build["acc"], build["gyro"], target
                )
                prep_notes = [{"file": "thigh", **t_info}, {"file": "shank", **s_info}]
                session_dir = write_session(Path(tempfile.mkdtemp(prefix="biomat_session_")), thigh_df, shank_df)
    except Exception as exc:
        st.exception(exc)
        return

    for note in prep_notes:
        if note.get("resampled"):
            st.warning(
                f"{note['file']}: native ~{note['native_hz']:.1f} Hz → resampled to ~{note['output_hz']:.1f} Hz "
                f"({note['samples']} samples)."
            )
        elif note.get("samples"):
            st.caption(f"{note['file']}: ~{note['native_hz']:.1f} Hz, {note['samples']} samples")

    try:
        with st.spinner(f"Loading {model_label}…"):
            bundle = load_60hz_bundle(str(package_dir)) if target_hz == 60 else load_100hz_bundle(str(package_dir))
        with st.spinner("Running inference…"):
            result = run_inference(bundle, Path(session_dir))
    except Exception as exc:
        st.exception(exc)
        st.error(
            "If this is a rate error, enable **Resample to target Hz**. "
            "Also check accel/gyro units (g vs m/s^2, deg/s vs rad/s)."
        )
        return

    st.success(f"Done — {result['accepted']} / {result['windows']} windows accepted by motion gate.")

    imu_combined: Optional[pd.DataFrame] = None
    thigh_imu = shank_imu = None
    accel_u, gyro_u = "m/s^2", "rad/s"
    try:
        thigh_imu, shank_imu, accel_u, gyro_u = load_session_phones(Path(session_dir))
        imu_combined = combined_imu_table(thigh_imu, shank_imu)
        st.subheader("Input IMU")
        st.caption(
            "Phone session CSVs after unit mapping / optional resample "
            "(before package filter + canonicalize)."
        )
        c_thigh, c_shank = st.columns(2)
        with c_thigh:
            st.plotly_chart(
                plot_imu_site(thigh_imu, "Thigh (phone_1)", accel_u, gyro_u),
                use_container_width=True,
            )
        with c_shank:
            st.plotly_chart(
                plot_imu_site(shank_imu, "Shank (phone_2)", accel_u, gyro_u),
                use_container_width=True,
            )
    except Exception as imu_exc:
        st.warning(f"Could not plot IMU traces: {imu_exc}")

    angles = result["angles"]
    st.subheader("Predicted joint angles")
    st.plotly_chart(plot_angles(angles, flip_knee=flip_knee), use_container_width=True)

    if validation_compare_meta is not None:
        gt_df = validation_compare_meta["gt"]
        run_id = (
            f"{len(angles)}_{float(angles['session_time_s'].iloc[-1]):.4f}_"
            f"{len(gt_df)}_{float(gt_df['time'].iloc[-1]):.4f}"
        )
        st.session_state["validation_run_cache"] = {
            "result": result,
            "angles": angles,
            "gt": gt_df,
            "session_dir": str(session_dir),
            "prep_notes": prep_notes,
            "imu_combined": imu_combined,
            "thigh_imu": thigh_imu,
            "shank_imu": shank_imu,
            "accel_u": accel_u,
            "gyro_u": gyro_u,
            "run_id": run_id,
        }
        # First time for this run: auto-align hip peaks, then rerun so slider/overlay pick it up.
        if st.session_state.get("_last_auto_align_run_id") != run_id:
            offset, info = estimate_gt_offset_from_hip(angles, gt_df)
            st.session_state["_pending_hip_align_offset"] = float(offset)
            st.session_state["validation_align_info"] = info
            st.session_state["_last_auto_align_run_id"] = run_id
            st.rerun()
        try:
            render_validation_overlay(
                angles,
                gt_df,
                flip_knee=flip_knee,
                gt_time_offset_s=float(gt_time_offset_s),
            )
        except Exception as cmp_exc:
            st.exception(cmp_exc)

    if result["activities"] is not None:
        st.subheader("Activity windows (100 Hz classifier)")
        st.dataframe(result["activities"], use_container_width=True)
        st.download_button(
            "Download activity_windows.csv",
            result["activities"].to_csv(index=False).encode("utf-8"),
            file_name="activity_windows.csv",
            mime="text/csv",
        )

    if imu_combined is not None:
        st.subheader("Combined IMU table (thigh + shank)")
        st.caption(
            f"Overlapping samples on a shared clock · accel `{accel_u}` · gyro `{gyro_u}` · "
            f"{len(imu_combined)} rows"
        )
        st.dataframe(imu_combined, use_container_width=True)
        st.download_button(
            "Download combined_imu.csv",
            imu_combined.to_csv(index=False).encode("utf-8"),
            file_name="combined_imu.csv",
            mime="text/csv",
        )

    st.subheader("Angle table")
    st.dataframe(angles, use_container_width=True)
    st.download_button(
        "Download joint_angles_experimental.csv",
        angles.to_csv(index=False).encode("utf-8"),
        file_name="joint_angles_experimental.csv",
        mime="text/csv",
    )
    blank = int((angles["contributing_windows"] == 0).sum())
    st.caption(
        f"{blank} time samples left blank (gate rejected / no overlapping windows). "
        "Expected at quiet start/end segments."
    )


if __name__ == "__main__":
    main()
