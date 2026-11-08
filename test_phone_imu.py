"""
POC Streamlit app: run BioMAT on phone IMU recordings.

  - Acceleration x/y/z (g) 
  - Gyroscope x/y/z (rad/s)

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
JOINT_NAMES = ["Hip flexion", "Knee"]
FEATURES_PER_SENSOR = 6
MODEL_DIR = os.path.join(".", "caches", "trained_model")
MODEL_STEM = "transformertsai_g1g2rardsasd_g1g2rardsasd.pt"

# Same options as biomat_app.py
SENSOR_OPTIONS = {
    "Thigh & Shank": ["thigh", "shank"],
    "Thigh": ["thigh"],
    "Shank": ["shank"],
}
SENSOR_FILE_KEYS = {
    "Thigh & Shank": ["thigh", "shank"],
    "Thigh": ["thigh"],
    "Shank": ["shank"],
}

ACC_KEYS = ["acc_x", "acc_y", "acc_z"]
GYRO_KEYS = ["gyro_x", "gyro_y", "gyro_z"]


def _norm(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", col.lower())


def _find_column(columns: List[str], patterns: List[str]) -> Optional[str]:
    normalized = {_norm(c): c for c in columns}
    for pattern in patterns:
        key = _norm(pattern)
        if key in normalized:
            return normalized[key]
        for ncol, original in normalized.items():
            # Prefer exact-ish matches; skip "absoluteacceleration" for axis patterns.
            if key in ncol and "absolute" not in ncol:
                return original
    return None


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    return {
        "time": _find_column(cols, ["Time (s)", "time", "t"]),
        "acc_x": _find_column(
            cols,
            ["Acceleration x (m/s^2)", "Acceleration x (m/s²)", "Acceleration x", "Linear Acceleration x", "ax", "accx"],
        ),
        "acc_y": _find_column(
            cols,
            ["Acceleration y (m/s^2)", "Acceleration y (m/s²)", "Acceleration y", "Linear Acceleration y", "ay", "accy"],
        ),
        "acc_z": _find_column(
            cols,
            ["Acceleration z (m/s^2)", "Acceleration z (m/s²)", "Acceleration z", "Linear Acceleration z", "az", "accz"],
        ),
        "gyro_x": _find_column(cols, ["Gyroscope x (rad/s)", "Gyroscope x (deg/s)", "Gyroscope x", "wx", "gx"]),
        "gyro_y": _find_column(cols, ["Gyroscope y (rad/s)", "Gyroscope y (deg/s)", "Gyroscope y", "wy", "gy"]),
        "gyro_z": _find_column(cols, ["Gyroscope z (rad/s)", "Gyroscope z (deg/s)", "Gyroscope z", "wz", "gz"]),
    }


def load_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load CSV or Excel export."""
    name = filename.lower()
    buffer = io.BytesIO(file_bytes)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    try:
        return pd.read_csv(buffer)
    except Exception:
        buffer.seek(0)
        return pd.read_csv(buffer, comment="#")


def _has_group(mapping: Dict[str, Optional[str]], keys: List[str]) -> bool:
    return all(mapping.get(k) for k in keys)


def modality_status(mapping: Dict[str, Optional[str]]) -> Tuple[bool, bool]:
    return _has_group(mapping, ACC_KEYS), _has_group(mapping, GYRO_KEYS)


def extract_imu_arrays(
    df: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
    """
    Build (time, imu[N,6]) = Accel XYZ + Gyro XYZ.
    """
    if not mapping.get("time"):
        raise ValueError("Missing required Time (s) column.")

    has_acc, has_gyro = modality_status(mapping)
    if not has_acc and not has_gyro:
        raise ValueError(
            "Need acceleration x/y/z and/or Gyroscope x/y/z. "
        )

    time = df[mapping["time"]].to_numpy(dtype=float)
    n = len(time)
    imu = np.zeros((n, 6), dtype=float)

    if has_acc:
        imu[:, 0] = df[mapping["acc_x"]].to_numpy(dtype=float)
        imu[:, 1] = df[mapping["acc_y"]].to_numpy(dtype=float)
        imu[:, 2] = df[mapping["acc_z"]].to_numpy(dtype=float)

    if has_gyro:
        imu[:, 3] = df[mapping["gyro_x"]].to_numpy(dtype=float)
        imu[:, 4] = df[mapping["gyro_y"]].to_numpy(dtype=float)
        imu[:, 5] = df[mapping["gyro_z"]].to_numpy(dtype=float)
        

    valid = np.isfinite(time) & np.all(np.isfinite(imu), axis=1)
    time, imu = time[valid], imu[valid]
    order = np.argsort(time)
    time, imu = time[order], imu[order]
    keep = np.concatenate([[True], np.diff(time) > 0])
    return time[keep], imu[keep], has_acc, has_gyro



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


def pack_phone_into_sensors(
    phone_windows: np.ndarray,
    selected_sensors: List[str],
    phone_placement: str,
) -> np.ndarray:
    """
    Pack one phone IMU (N, T, 6) into the selected sensor channel layout.
    Channel order matches LoadPickleDataSet.combine_sensors_features for selected_sensors.
    """
    if phone_placement not in selected_sensors:
        raise ValueError(
            f"Phone placement '{phone_placement}' is not in selected sensors {selected_sensors}."
        )
    n, t, c = phone_windows.shape
    if c != FEATURES_PER_SENSOR:
        raise ValueError(f"Expected {FEATURES_PER_SENSOR} phone channels, got {c}.")

    n_sensors = len(selected_sensors)
    packed = np.zeros((n, t, n_sensors * FEATURES_PER_SENSOR), dtype=float)
    slot = selected_sensors.index(phone_placement)
    start = slot * FEATURES_PER_SENSOR
    packed[:, :, start : start + FEATURES_PER_SENSOR] = phone_windows
    return packed


def resolve_model_path(sensor_label: str) -> Tuple[str, str]:
    """Return (path, display_label) for the BioMAT checkpoint matching biomat_app naming."""
    path = os.path.join(MODEL_DIR, MODEL_STEM)    
    return path, f"BioMAT ({sensor_label})"


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


def plot_imu_preview(time: np.ndarray, imu: np.ndarray, has_acc: bool, has_gyro: bool) -> go.Figure:
    titles = [
        "Acceleration (m/s²)" + ("" if has_acc else " — zero-filled"),
        "Gyro (rad/s)" + ("" if has_gyro else " — zero-filled"),
    ]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=titles)
    labels_acc = ["Ax", "Ay", "Az"]
    labels_gyro = ["Gx", "Gy", "Gz"]
    for i in range(3):
        fig.add_trace(go.Scatter(x=time, y=imu[:, i], mode="lines", name=labels_acc[i]), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=imu[:, i + 3], mode="lines", name=labels_gyro[i]), row=2, col=1)
    fig.update_layout(height=420, title="Resampled shank IMU @ 100 Hz")
    return fig


def _mapping_ui(raw_df: pd.DataFrame, detected: Dict[str, Optional[str]], key_prefix: str) -> Dict[str, Optional[str]]:
    st.subheader("Column mapping")
    cols = ["—"] + list(raw_df.columns)
    mapping: Dict[str, Optional[str]] = {}
    map_cols = st.columns(4)
    keys = ["time", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    labels = ["Time", "Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z"]
    for i, (key, label) in enumerate(zip(keys, labels)):
        default = detected.get(key)
        index = cols.index(default) if default in cols else 0
        with map_cols[i % 4]:
            choice = st.selectbox(label, cols, index=index, key=f"{key_prefix}_map_{key}")
            mapping[key] = None if choice == "—" else choice
    return mapping


def main():
    st.set_page_config(page_title="BioMAT Phone IMU POC", layout="wide")
    st.title("BioMAT phone IMU POC")
    st.caption(
        "choose a sensor configuration like BioMAT, place the phone on one segment, "
        "and predict hip / knee / ankle kinematics. Missing modality / sensor channels are zero-filled."
    )

    config = get_config_universal("camargo")
    config["model_name"] = "transformertsai"
    config["selected_opensim_labels"] = ["hip_flexion_r", "knee_angle_r", "ankle_angle_r"]
    config["batch_size"] = min(int(config.get("batch_size", 50)), 32)

    with st.sidebar:
        st.header("Sensor Configuration")
        selected_sensor_label = st.selectbox(
            "Pick one",
            list(SENSOR_OPTIONS.keys()),
            index=list(SENSOR_OPTIONS.keys()).index("Shank"),
        )
        selected_sensors = SENSOR_OPTIONS[selected_sensor_label]
        config["selected_sensors"] = selected_sensors

        default_placement = "shank" if "shank" in selected_sensors else selected_sensors[0]
        phone_placement = st.selectbox(
            "Phone placement (filled IMU)",
            selected_sensors,
            index=selected_sensors.index(default_placement),
            help="Only one phone file is used; other sensors in the configuration are zero-filled.",
        )
        try:
            model_path_preview, _ = resolve_model_path(selected_sensor_label)
            st.caption(f"Checkpoint: `{os.path.basename(model_path_preview)}`")
        except FileNotFoundError as exc:
            st.error(str(exc))

        st.header("Input")
        uploaded = st.file_uploader(
            "Primary  file (accel or gyro)",
            type=["csv", "xlsx", "xls"],
            key="primary_upload",
        )
        
        flip_sign = st.checkbox("Flip prediction sign (Camargo convention)", value=True)
        st.markdown("**Activity:** Level ground walking")
        st.markdown(
            "Example accel export columns: `Time (s)`, `Acceleration x/y/z (m/s^2)`. "
            "`Absolute acceleration` is ignored. Axes rarely match Camargo — treat this as a POC."
        )

    if uploaded is None:
        st.info(
            "Upload a  CSV/Excel with **Time + Acceleration x/y/z** "
            "(or Time + Gyroscope x/y/z) to begin."
        )
        return

    raw_df = load_table(uploaded.getvalue(), uploaded.name)
    detected = detect_columns(raw_df)
    mapping = _mapping_ui(raw_df, detected, key_prefix="primary")
    has_acc, has_gyro = modality_status(mapping)

    if has_acc and not has_gyro:
        st.warning("Acceleration-only file detected — gyro channels will be zero-filled")
    elif has_gyro and not has_acc:
        st.warning("Gyroscope-only file detected — accel channels will be zero-filled")
    elif has_acc and has_gyro:
        st.success("Both acceleration and gyroscope columns mapped.")
    else:
        st.error("Map Time plus either Acceleration x/y/z or Gyroscope x/y/z.")

    with st.expander("Primary file preview", expanded=False):
        st.dataframe(raw_df.head(20), use_container_width=True)

    
    if not st.button("Run BioMAT", type="primary"):
        return

    try:
        time_raw, imu_raw, has_acc, has_gyro = extract_imu_arrays(
            raw_df, mapping
        )
        

        time_rs, imu_rs = resample_imu(time_raw, imu_raw, TARGET_FS)
        phone_windows = segment_windows(imu_rs, WINSIZE, OVERLAP)

        model_path, model_label = resolve_model_path(selected_sensor_label)
        x_windows = pack_phone_into_sensors(phone_windows, selected_sensors, phone_placement)
        config["selected_sensors"] = selected_sensors

        zeroed = [s for s in selected_sensors if s != phone_placement]
        if zeroed:
            st.warning(
                f"Using `{os.path.basename(model_path)}` ({model_label}). "
                f"Phone fills **{phone_placement}**; zero-filled: {', '.join(zeroed)}."
            )
        else:
            st.success(f"Using `{os.path.basename(model_path)}` ({model_label}) on **{phone_placement}**.")

        modality_note = []
        modality_note.append("accel ✓" if has_acc else "accel zero-filled")
        modality_note.append("gyro ✓" if has_gyro else "gyro zero-filled")
        st.info(f"{phone_placement.capitalize()} modalities: " + ", ".join(modality_note))

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

    st.plotly_chart(
        plot_imu_preview(time_rs, imu_rs, has_acc=has_acc, has_gyro=has_gyro),
        use_container_width=True,
    )

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
