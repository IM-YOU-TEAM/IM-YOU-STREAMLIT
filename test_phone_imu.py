"""
POC Streamlit app: run BioMAT on phyphox phone-IMU shank CSV for level-ground walking.

Expected phyphox columns (names are matched flexibly):
  Time (s), Acceleration x/y/z (m/s^2), Gyroscope x/y/z (rad/s)

Run:
  streamlit run test_phone_imu.py
"""

from __future__ import annotations

import io
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import torch
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader, TensorDataset

from config import get_config_universal
from test import Test
from utils.pytorchtools import load_trained_model


TARGET_FS = 100.0
WINSIZE = 256
OVERLAP = 0.5
JOINT_NAMES = ["Hip flexion", "Knee", "Ankle"]
SHANK_CHANNELS = [
    "shank_Accel_X",
    "shank_Accel_Y",
    "shank_Accel_Z",
    "shank_Gyro_X",
    "shank_Gyro_Y",
    "shank_Gyro_Z",
]
# Camargo / BioMAT sensor packing order for the full 3-IMU model.
FULL_SENSOR_ORDER = ["foot", "shank", "thigh"]

SHANK_MODEL = "shank_transformertsai_g1g2rardsasd_g1g2rardsasd.pt"
FULL_MODEL = "transformertsai_g1g2rardsasd_g1g2rardsasd.pt"
MODEL_DIR = os.path.join(".", "caches", "trained_model")


def _norm(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", col.lower())


def _find_column(columns: List[str], patterns: List[str]) -> Optional[str]:
    normalized = {_norm(c): c for c in columns}
    for pattern in patterns:
        key = _norm(pattern)
        if key in normalized:
            return normalized[key]
        for ncol, original in normalized.items():
            if key in ncol:
                return original
    return None


def detect_phyphox_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    return {
        "time": _find_column(cols, ["Time (s)", "time", "t"]),
        "acc_x": _find_column(cols, ["Acceleration x (m/s^2)", "Acceleration x", "Linear Acceleration x", "ax", "accx"]),
        "acc_y": _find_column(cols, ["Acceleration y (m/s^2)", "Acceleration y", "Linear Acceleration y", "ay", "accy"]),
        "acc_z": _find_column(cols, ["Acceleration z (m/s^2)", "Acceleration z", "Linear Acceleration z", "az", "accz"]),
        "gyro_x": _find_column(cols, ["Gyroscope x (rad/s)", "Gyroscope x (deg/s)", "Gyroscope x", "wx", "gyrox"]),
        "gyro_y": _find_column(cols, ["Gyroscope y (rad/s)", "Gyroscope y (deg/s)", "Gyroscope y", "wy", "gyroy"]),
        "gyro_z": _find_column(cols, ["Gyroscope z (rad/s)", "Gyroscope z (deg/s)", "Gyroscope z", "wz", "gyroz"]),
    }


def load_phyphox_csv(file_bytes: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    # Phyphox sometimes writes a metadata header before the column names.
    try:
        return pd.read_csv(buffer)
    except Exception:
        buffer.seek(0)
        return pd.read_csv(buffer, comment="#")


def extract_imu_arrays(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    gyro_in_deg: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    required = ["time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    missing = [k for k in required if not mapping.get(k)]
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
    if gyro_in_deg:
        imu[:, 3:] = np.deg2rad(imu[:, 3:])

    # Drop non-finite / non-monotonic time samples.
    valid = np.isfinite(time) & np.all(np.isfinite(imu), axis=1)
    time, imu = time[valid], imu[valid]
    order = np.argsort(time)
    time, imu = time[order], imu[order]
    keep = np.concatenate([[True], np.diff(time) > 0])
    return time[keep], imu[keep]


def resample_imu(time: np.ndarray, imu: np.ndarray, target_fs: float = TARGET_FS) -> Tuple[np.ndarray, np.ndarray]:
    if len(time) < 2:
        raise ValueError("Need at least 2 samples to resample.")
    duration = time[-1] - time[0]
    if duration <= 0:
        raise ValueError("Time vector must be increasing.")
    n_out = max(int(np.floor(duration * target_fs)) + 1, WINSIZE)
    t_new = time[0] + np.arange(n_out) / target_fs
    imu_new = np.column_stack([np.interp(t_new, time, imu[:, i]) for i in range(imu.shape[1])])
    return t_new, imu_new


def segment_windows(signal: np.ndarray, winsize: int = WINSIZE, overlap: float = OVERLAP) -> np.ndarray:
    """Match FixWindowSegmentation: step = round(overlap * winsize)."""
    step = max(int(round(overlap * winsize)), 1)
    if len(signal) < winsize:
        pad = np.zeros((winsize - len(signal), signal.shape[1]), dtype=float)
        return np.expand_dims(np.vstack([signal, pad]), axis=0)

    windows = []
    for start in range(0, len(signal) - winsize + 1, step):
        windows.append(signal[start : start + winsize])
    return np.asarray(windows, dtype=float)


def expand_shank_to_full(shank_windows: np.ndarray) -> np.ndarray:
    """Pack shank (6) into foot/shank/thigh (18) with zeros for missing IMUs."""
    n, t, _ = shank_windows.shape
    full = np.zeros((n, t, 18), dtype=float)
    # foot=0:6, shank=6:12, thigh=12:18
    full[:, :, 6:12] = shank_windows
    return full


def resolve_model_path() -> Tuple[str, str, bool]:
    """
    Returns (path, label, use_full_padding).
    Prefer a true shank-only checkpoint; fall back to the available 3-IMU model.
    """
    shank_path = os.path.join(MODEL_DIR, SHANK_MODEL)
    full_path = os.path.join(MODEL_DIR, FULL_MODEL)
    if os.path.isfile(shank_path):
        return shank_path, "shank-only BioMAT", False
    if os.path.isfile(full_path):
        return full_path, "full BioMAT (foot/thigh zero-filled)", True
    raise FileNotFoundError(
        f"No BioMAT checkpoint found. Expected `{SHANK_MODEL}` or `{FULL_MODEL}` in {MODEL_DIR}."
    )


@st.cache_resource
def load_model(model_path: str):
    return load_trained_model(model_path)


def run_inference(config: dict, model, x_windows: np.ndarray) -> np.ndarray:
    y_dummy = np.zeros((x_windows.shape[0], x_windows.shape[1], len(config["selected_opensim_labels"])), dtype=float)
    dataset = TensorDataset(
        torch.from_numpy(x_windows).double(),
        torch.from_numpy(y_dummy).double(),
    )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    y_pred, _, _ = Test().run_testing(config, model, test_dataloader=dataloader)
    return y_pred.detach().cpu().numpy()


def stitch_windows(windows: np.ndarray, overlap: float = OVERLAP) -> np.ndarray:
    """Average overlapping predictions into a continuous sequence."""
    n, winsize, n_out = windows.shape
    step = max(int(round(overlap * winsize)), 1)
    total_len = winsize + (n - 1) * step
    out = np.zeros((total_len, n_out), dtype=float)
    weight = np.zeros((total_len, 1), dtype=float)
    for i in range(n):
        start = i * step
        out[start : start + winsize] += windows[i]
        weight[start : start + winsize] += 1.0
    weight = np.maximum(weight, 1.0)
    return out / weight


def plot_predictions(y_pred: np.ndarray, window_index: int, flip_sign: bool) -> go.Figure:
    pred = y_pred[window_index]
    if flip_sign:
        pred = -pred
    t = np.arange(pred.shape[0]) / TARGET_FS
    fig = make_subplots(rows=1, cols=3, subplot_titles=JOINT_NAMES)
    for j, name in enumerate(JOINT_NAMES):
        fig.add_trace(
            go.Scatter(x=t, y=pred[:, j], mode="lines", name=name, showlegend=False),
            row=1,
            col=j + 1,
        )
        fig.update_yaxes(title_text="deg", row=1, col=j + 1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=j + 1)
    fig.update_layout(
        title=f"Level-ground walking kinematics — window {window_index}",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def plot_stitched(y_pred: np.ndarray, flip_sign: bool) -> go.Figure:
    stitched = stitch_windows(y_pred)
    if flip_sign:
        stitched = -stitched
    t = np.arange(stitched.shape[0]) / TARGET_FS
    fig = make_subplots(rows=1, cols=3, subplot_titles=JOINT_NAMES)
    for j, name in enumerate(JOINT_NAMES):
        fig.add_trace(
            go.Scatter(x=t, y=stitched[:, j], mode="lines", name=name, showlegend=False),
            row=1,
            col=j + 1,
        )
        fig.update_yaxes(title_text="deg", row=1, col=j + 1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=j + 1)
    fig.update_layout(
        title="Stitched level-ground walking kinematics",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def plot_imu_preview(time: np.ndarray, imu: np.ndarray) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Acceleration (m/s²)", "Gyro (rad/s)"])
    labels_acc = ["Ax", "Ay", "Az"]
    labels_gyro = ["Gx", "Gy", "Gz"]
    for i in range(3):
        fig.add_trace(go.Scatter(x=time, y=imu[:, i], mode="lines", name=labels_acc[i]), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=imu[:, i + 3], mode="lines", name=labels_gyro[i]), row=2, col=1)
    fig.update_layout(height=420, title="Resampled shank IMU @ 100 Hz")
    return fig


def main():
    st.set_page_config(page_title="BioMAT Phone IMU POC", layout="wide")
    st.title("BioMAT phone IMU POC")
    st.caption(
        "Upload a phyphox shank IMU CSV, treat the trial as level-ground walking, "
        "and predict hip / knee / ankle kinematics with BioMAT."
    )

    config = get_config_universal("camargo")
    config["model_name"] = "transformertsai"
    config["selected_opensim_labels"] = ["hip_flexion_r", "knee_angle_r", "ankle_angle_r"]
    config["batch_size"] = min(int(config.get("batch_size", 50)), 32)

    with st.sidebar:
        st.header("Input")
        uploaded = st.file_uploader("Phyphox CSV (shank)", type=["csv"])
        gyro_in_deg = st.checkbox("Gyro columns are in deg/s", value=False)
        flip_sign = st.checkbox("Flip prediction sign (Camargo convention)", value=True)
        st.markdown("**Activity:** Level ground walking")
        st.markdown(
            "Phone axes rarely match Camargo IMU axes. This POC assumes your CSV "
            "order is Accel XYZ then Gyro XYZ in a shank-mounted frame."
        )

    if uploaded is None:
        st.info("Upload a phyphox CSV with time + accel + gyro columns to begin.")
        return

    raw_df = load_phyphox_csv(uploaded.getvalue())
    detected = detect_phyphox_columns(raw_df)

    st.subheader("Column mapping")
    cols = ["—"] + list(raw_df.columns)
    mapping = {}
    map_cols = st.columns(4)
    keys = ["time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    labels = ["Time", "Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z"]
    for i, (key, label) in enumerate(zip(keys, labels)):
        default = detected.get(key)
        index = cols.index(default) if default in cols else 0
        with map_cols[i % 4]:
            choice = st.selectbox(label, cols, index=index, key=f"map_{key}")
            mapping[key] = None if choice == "—" else choice

    with st.expander("CSV preview", expanded=False):
        st.dataframe(raw_df.head(20), use_container_width=True)

    if not st.button("Run BioMAT", type="primary"):
        return

    try:
        time_raw, imu_raw = extract_imu_arrays(raw_df, mapping, gyro_in_deg=gyro_in_deg)
        time_rs, imu_rs = resample_imu(time_raw, imu_raw, TARGET_FS)
        shank_windows = segment_windows(imu_rs, WINSIZE, OVERLAP)

        model_path, model_label, use_full_padding = resolve_model_path()
        if use_full_padding:
            config["selected_sensors"] = FULL_SENSOR_ORDER
            x_windows = expand_shank_to_full(shank_windows)
            st.warning(
                f"Using `{os.path.basename(model_path)}` ({model_label}). "
                "Only shank channels are filled; foot/thigh are zeros."
            )
        else:
            config["selected_sensors"] = ["shank"]
            x_windows = shank_windows
            st.success(f"Using `{os.path.basename(model_path)}` ({model_label}).")

        with st.spinner("Loading model…"):
            model = load_model(model_path)
            model.eval()

        with st.spinner("Running inference…"):
            y_pred = run_inference(config, model, x_windows)

    except Exception as exc:
        st.error(f"Failed to run POC: {exc}")
        return

    duration_s = time_rs[-1] - time_rs[0]
    st.success(
        f"Processed {len(time_raw)} raw samples → {len(time_rs)} @ {TARGET_FS:.0f} Hz "
        f"({duration_s:.1f}s) → {len(x_windows)} windows of {WINSIZE}."
    )

    st.plotly_chart(plot_imu_preview(time_rs, imu_rs), use_container_width=True)

    st.subheader("Predicted joint angles")
    view = st.radio("View", ["Stitched trial", "Single window"], horizontal=True)
    if view == "Stitched trial":
        st.plotly_chart(plot_stitched(y_pred, flip_sign=flip_sign), use_container_width=True)
    else:
        idx = st.slider("Window index", 0, max(len(y_pred) - 1, 0), 0)
        st.plotly_chart(plot_predictions(y_pred, idx, flip_sign=flip_sign), use_container_width=True)

    out_df = pd.DataFrame(
        stitch_windows(y_pred) * (-1.0 if flip_sign else 1.0),
        columns=["hip_flexion_r_deg", "knee_angle_r_deg", "ankle_angle_r_deg"],
    )
    out_df.insert(0, "time_s", np.arange(len(out_df)) / TARGET_FS)
    st.download_button(
        "Download stitched predictions (CSV)",
        out_df.to_csv(index=False).encode("utf-8"),
        file_name="biomat_phone_imu_predictions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
