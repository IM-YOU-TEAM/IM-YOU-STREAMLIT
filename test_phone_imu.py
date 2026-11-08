"""
POC Streamlit app: run the thigh/shank BioMAT model on phone IMU recordings.

Requires four files:
  - RIGHT SHANK accelerometer_uncalibrated
  - RIGHT SHANK gyroscope
  - RIGHT THIGH accelerometer_uncalibrated
  - RIGHT THIGH gyroscope

Accel and gyro for each site are merged on `seconds_elapsed`, then shank/thigh
are aligned to a shared 100 Hz timeline. Uses the same canonical100_v1 contract
as biomat_app.py when checkpoint metadata says so.

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
from preprocessing.canonicalize import (
    LENGTH as CANONICAL_LENGTH,
    SITES,
    as_numpy,
    is_canonical100_meta,
    prepare_canonical100_inputs,
)
from test import Test
from utils.pytorchtools import load_trained_model, peek_checkpoint_meta


TARGET_FS = 100.0
OVERLAP = 0.5
JOINT_NAMES = ["Hip flexion", "Knee", "Ankle"]
FEATURES_PER_SENSOR = 6
MODEL_PATH = os.path.join(
    ".",
    "caches",
    "trained_model",
    "thighshank_transformertsai_g1g2rardsasd_g1g2rardsasd.pt",
)

UploadKey = str  # e.g. "shank_accel"


def _norm(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", col.lower())


def _find_column(columns: List[str], patterns: List[str]) -> Optional[str]:
    normalized = {_norm(c): c for c in columns}
    for pattern in patterns:
        key = _norm(pattern)
        if key in normalized:
            return normalized[key]
        for ncol, original in normalized.items():
            if key in ncol and "absolute" not in ncol and "bias" not in ncol:
                return original
    return None


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """Prefer seconds_elapsed over raw nanosecond `time` stamps."""
    return _find_column(
        list(df.columns),
        [
            "seconds_elapsed",
            "seconds_e",
            "Time (s)",
            "common_timestamp",
            "time_s",
            "t",
            "time",
        ],
    )


def detect_xyz_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Detect axis columns; prefer exact x/y/z so 'seconds_elapsed' is not matched."""
    cols = list(df.columns)
    normalized = {_norm(c): c for c in cols}

    def pick(*patterns: str) -> Optional[str]:
        for pattern in patterns:
            key = _norm(pattern)
            if key in normalized:
                return normalized[key]
        return None

    return {
        "x": pick("Acceleration x", "Accel_X", "ax", "accx", "gx", "x"),
        "y": pick("Acceleration y", "Accel_Y", "ay", "accy", "gy", "y"),
        "z": pick("Acceleration z", "Accel_Z", "az", "accz", "gz", "z"),
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


def extract_triad(
    df: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (seconds_elapsed, values[N,3]) for one accel or gyro file."""
    required = ["time", "x", "y", "z"]
    missing = [k for k in required if not mapping.get(k)]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    time = df[mapping["time"]].to_numpy(dtype=float)
    values = np.column_stack(
        [
            df[mapping["x"]].to_numpy(dtype=float),
            df[mapping["y"]].to_numpy(dtype=float),
            df[mapping["z"]].to_numpy(dtype=float),
        ]
    )
    valid = np.isfinite(time) & np.all(np.isfinite(values), axis=1)
    time, values = time[valid], values[valid]
    order = np.argsort(time)
    time, values = time[order], values[order]
    keep = np.concatenate([[True], np.diff(time) > 0])
    return time[keep], values[keep]


def merge_accel_gyro_on_seconds(
    acc_time: np.ndarray,
    acc: np.ndarray,
    gyro_time: np.ndarray,
    gyro: np.ndarray,
    target_fs: float = TARGET_FS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match one site's accelerometer_uncalibrated + gyroscope on seconds_elapsed.
    Builds a shared 100 Hz timeline over their overlap and interpolates both.
    Returns (timeline, imu[N,6]) with accel XYZ then gyro XYZ.
    """
    start = max(float(acc_time[0]), float(gyro_time[0]))
    stop = min(float(acc_time[-1]), float(gyro_time[-1]))
    if stop <= start:
        raise ValueError("Accel and gyro have no overlapping seconds_elapsed range.")
    timeline = start + np.arange(int(np.floor((stop - start) * target_fs)) + 1) / target_fs
    if len(timeline) < 2:
        raise ValueError("Insufficient accel/gyro overlap after matching on seconds_elapsed.")
    acc_i = np.column_stack([np.interp(timeline, acc_time, acc[:, j]) for j in range(3)])
    gyro_i = np.column_stack([np.interp(timeline, gyro_time, gyro[:, j]) for j in range(3)])
    return timeline, np.concatenate([acc_i, gyro_i], axis=1)


def align_dual_phone_streams(
    streams: Dict[str, Tuple[np.ndarray, np.ndarray]],
    target_fs: float = TARGET_FS,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Align already-merged shank/thigh IMUs onto a shared 100 Hz overlap timeline."""
    start = max(t[0] for t, _ in streams.values())
    stop = min(t[-1] for t, _ in streams.values())
    if stop <= start:
        raise ValueError("Shank and thigh recordings have no overlapping time range.")
    timeline = start + np.arange(int(np.floor((stop - start) * target_fs)) + 1) / target_fs
    if len(timeline) < 2:
        raise ValueError("Insufficient synchronized overlap between shank and thigh.")
    aligned = {
        site: np.column_stack([np.interp(timeline, time, imu[:, j]) for j in range(imu.shape[1])])
        for site, (time, imu) in streams.items()
    }
    return timeline, aligned


def segment_windows(signal: np.ndarray, winsize: int, overlap: float = OVERLAP) -> np.ndarray:
    step = max(int(round(overlap * winsize)), 1)
    if len(signal) < winsize:
        raise ValueError(f"Need at least {winsize} samples; got {len(signal)}.")
    windows = [signal[start : start + winsize] for start in range(0, len(signal) - winsize + 1, step)]
    return np.asarray(windows, dtype=float)


def pack_shank_thigh_windows(
    shank_windows: np.ndarray,
    thigh_windows: np.ndarray,
    sites: List[str] = None,
) -> np.ndarray:
    sites = list(sites or SITES)
    if sites != ["shank", "thigh"]:
        raise ValueError(f"Expected sites ['shank', 'thigh'], got {sites}")
    if shank_windows.shape != thigh_windows.shape:
        raise ValueError(f"Window shape mismatch: shank {shank_windows.shape} vs thigh {thigh_windows.shape}")
    if shank_windows.ndim != 3 or shank_windows.shape[-1] != FEATURES_PER_SENSOR:
        raise ValueError(f"Expected [N, T, 6] site windows, got {shank_windows.shape}")
    by_site = {"shank": shank_windows, "thigh": thigh_windows}
    return np.concatenate([by_site[site] for site in sites], axis=-1)


def merge_checkpoint_meta(model, model_path: str) -> dict:
    meta = dict(peek_checkpoint_meta(model_path) or {})
    model_meta = getattr(model, "checkpoint_meta", None) or {}
    for key, value in model_meta.items():
        meta.setdefault(key, value)
    return meta


def apply_model_contract(config: dict, model, checkpoint_meta: Optional[dict] = None):
    seq_len = getattr(model, "seq_len", None)
    if seq_len is None and hasattr(model, "W_pos"):
        seq_len = int(model.W_pos.shape[0])
    if seq_len is not None:
        config["target_padding_length"] = int(seq_len)

    meta = checkpoint_meta if checkpoint_meta is not None else (getattr(model, "checkpoint_meta", None) or {})
    if meta.get("window_length"):
        config["target_padding_length"] = int(meta["window_length"])
    if meta.get("sites"):
        config["selected_sensors"] = list(meta["sites"])
    if meta.get("targets"):
        config["selected_opensim_labels"] = list(meta["targets"])
    return config, meta


@st.cache_resource
def load_model(model_path: str):
    return load_trained_model(model_path)


def run_inference(config: dict, model, x_windows: np.ndarray) -> np.ndarray:
    y_dummy = np.zeros(
        (x_windows.shape[0], x_windows.shape[1], len(config["selected_opensim_labels"])),
        dtype=float,
    )
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(x_windows, dtype=np.float64)),
        torch.from_numpy(y_dummy).double(),
    )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    y_pred, _, _ = Test().run_testing(config, model, test_dataloader=dataloader)
    return y_pred.detach().cpu().numpy()


def stitch_windows(windows: np.ndarray, overlap: float = OVERLAP) -> np.ndarray:
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
        fig.update_yaxes(title_text="Angle (deg)", row=1, col=j + 1)
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
        fig.update_yaxes(title_text="Angle (deg)", row=1, col=j + 1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=j + 1)
    fig.update_layout(
        title="Stitched level-ground walking kinematics",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def plot_imu_preview(time: np.ndarray, imu: np.ndarray, title: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Acceleration (m/s²)", "Gyroscope (rad/s)"],
    )
    for i, label in enumerate(["Ax", "Ay", "Az"]):
        fig.add_trace(go.Scatter(x=time, y=imu[:, i], mode="lines", name=label), row=1, col=1)
    for i, label in enumerate(["Gx", "Gy", "Gz"]):
        fig.add_trace(go.Scatter(x=time, y=imu[:, i + 3], mode="lines", name=label), row=2, col=1)
    fig.update_yaxes(title_text="m/s²", row=1, col=1)
    fig.update_yaxes(title_text="rad/s", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(height=420, title=title)
    return fig


def _triad_mapping_ui(
    raw_df: pd.DataFrame,
    key_prefix: str,
    modality: str,
) -> Dict[str, Optional[str]]:
    st.markdown(f"**{key_prefix}** (`{modality}`)")
    cols = ["—"] + list(raw_df.columns)
    detected_time = detect_time_column(raw_df)
    detected_xyz = detect_xyz_columns(raw_df)
    mapping: Dict[str, Optional[str]] = {}
    map_cols = st.columns(4)
    fields = [
        ("time", "seconds_elapsed", detected_time),
        ("x", "X", detected_xyz.get("x")),
        ("y", "Y", detected_xyz.get("y")),
        ("z", "Z", detected_xyz.get("z")),
    ]
    for i, (key, label, default) in enumerate(fields):
        index = cols.index(default) if default in cols else 0
        with map_cols[i]:
            choice = st.selectbox(label, cols, index=index, key=f"{key_prefix}_map_{key}")
            mapping[key] = None if choice == "—" else choice
    return mapping


def main():
    st.set_page_config(page_title="BioMAT Thigh/Shank Phone IMU", layout="wide")
    st.title("BioMAT thigh/shank phone IMU")
    st.caption(
        "Loads only `thighshank_transformertsai_g1g2rardsasd_g1g2rardsasd.pt`. "
        "Upload four files: accelerometer_uncalibrated + gyroscope for RIGHT SHANK and RIGHT THIGH. "
        "Streams are matched on `seconds_elapsed`."
    )

    config = get_config_universal("camargo")
    config["model_name"] = "transformertsai"
    config["selected_sensors"] = list(SITES)
    config["selected_opensim_labels"] = ["hip_flexion_r", "knee_angle_r", "ankle_angle_r"]
    config["batch_size"] = min(int(config.get("batch_size", 50)), 32)
    config["target_padding_length"] = CANONICAL_LENGTH

    with st.sidebar:
        st.header("Model")
        st.caption(f"Checkpoint: `{os.path.basename(MODEL_PATH)}`")
        if not os.path.isfile(MODEL_PATH):
            st.error(f"Model file not found: {MODEL_PATH}")

        st.header("Input (4 files)")
        uploaded_shank_accel = st.file_uploader(
            "RIGHT SHANK accelerometer_uncalibrated",
            type=["csv", "xlsx", "xls"],
            key="shank_accel_upload",
        )
        uploaded_shank_gyro = st.file_uploader(
            "RIGHT SHANK gyroscope",
            type=["csv", "xlsx", "xls"],
            key="shank_gyro_upload",
        )
        uploaded_thigh_accel = st.file_uploader(
            "RIGHT THIGH accelerometer_uncalibrated",
            type=["csv", "xlsx", "xls"],
            key="thigh_accel_upload",
        )
        uploaded_thigh_gyro = st.file_uploader(
            "RIGHT THIGH gyroscope",
            type=["csv", "xlsx", "xls"],
            key="thigh_gyro_upload",
        )
        flip_sign = st.checkbox("Flip prediction sign (Camargo convention)", value=True)
        st.markdown("**Activity:** Level ground walking")
        st.caption("Expected units: acceleration m/s², gyroscope rad/s (no conversion applied).")

    uploads = {
        "shank_accel": uploaded_shank_accel,
        "shank_gyro": uploaded_shank_gyro,
        "thigh_accel": uploaded_thigh_accel,
        "thigh_gyro": uploaded_thigh_gyro,
    }
    if any(v is None for v in uploads.values()):
        st.info(
            "Upload all four files to begin: shank accel, shank gyro, thigh accel, thigh gyro."
        )
        return

    tables: Dict[str, Tuple[pd.DataFrame, Dict[str, Optional[str]]]] = {}
    labels = {
        "shank_accel": ("SHANK accelerometer_uncalibrated", "accelerometer"),
        "shank_gyro": ("SHANK gyroscope", "gyroscope"),
        "thigh_accel": ("THIGH accelerometer_uncalibrated", "accelerometer"),
        "thigh_gyro": ("THIGH gyroscope", "gyroscope"),
    }
    for key, uploaded in uploads.items():
        raw_df = load_table(uploaded.getvalue(), uploaded.name)
        title, modality = labels[key]
        with st.expander(f"{title} — `{uploaded.name}`", expanded=True):
            mapping = _triad_mapping_ui(raw_df, key_prefix=key, modality=modality)
            st.dataframe(raw_df.head(12), use_container_width=True)
        tables[key] = (raw_df, mapping)

    if not st.button("Run BioMAT", type="primary"):
        return

    try:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        with st.spinner("Loading model…"):
            model = load_model(MODEL_PATH)
            model.eval()

        checkpoint_meta = merge_checkpoint_meta(model, MODEL_PATH)
        config, checkpoint_meta = apply_model_contract(config, model, checkpoint_meta=checkpoint_meta)
        use_canonical100 = is_canonical100_meta(checkpoint_meta)
        selected_sensors = list(config["selected_sensors"])
        winsize = int(config.get("target_padding_length", CANONICAL_LENGTH))

        if selected_sensors != ["shank", "thigh"]:
            raise ValueError(f"Thigh/shank model expects sites ['shank', 'thigh']; got {selected_sensors}")
        if use_canonical100 and winsize != CANONICAL_LENGTH:
            winsize = CANONICAL_LENGTH

        st.info(
            f"Using model contract: sensors={selected_sensors}, window={winsize}, "
            f"preprocess={'canonical100_v1' if use_canonical100 else 'default'}"
        )

        site_streams: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for site in ("shank", "thigh"):
            acc_df, acc_map = tables[f"{site}_accel"]
            gyro_df, gyro_map = tables[f"{site}_gyro"]
            acc_t, acc = extract_triad(acc_df, acc_map)
            gyro_t, gyro = extract_triad(gyro_df, gyro_map)
            site_t, site_imu = merge_accel_gyro_on_seconds(acc_t, acc, gyro_t, gyro, TARGET_FS)
            site_streams[site] = (site_t, site_imu)

        timeline, aligned = align_dual_phone_streams(site_streams, TARGET_FS)
        shank_windows = segment_windows(aligned["shank"], winsize, OVERLAP)
        thigh_windows = segment_windows(aligned["thigh"], winsize, OVERLAP)
        n_windows = min(len(shank_windows), len(thigh_windows))
        x_windows = pack_shank_thigh_windows(
            shank_windows[:n_windows],
            thigh_windows[:n_windows],
            sites=selected_sensors,
        )

        if use_canonical100:
            mean = checkpoint_meta.get("mean")
            scale = checkpoint_meta.get("scale")
            if mean is None or scale is None:
                raise ValueError("canonical100 checkpoint is missing mean/scale.")
            x_windows = prepare_canonical100_inputs(
                x_windows,
                as_numpy(mean),
                as_numpy(scale),
                apply_lowpass=True,
            )

        st.success(
            f"Using `{os.path.basename(MODEL_PATH)}`; matched accel+gyro on seconds_elapsed; "
            f"preprocess={'canonical100_v1' if use_canonical100 else 'default'}."
        )

        with st.spinner("Running inference…"):
            y_pred = run_inference(config, model, x_windows)

    except Exception as exc:
        st.error(f"Failed to run POC: {exc}")
        return

    duration_s = float(timeline[-1] - timeline[0])
    n_raw = sum(len(t) for t, _ in site_streams.values())
    st.success(
        f"Merged site streams → {len(timeline)} overlapped samples @ {TARGET_FS:.0f} Hz "
        f"({duration_s:.1f}s; {n_raw} site-timeline points) → {len(x_windows)} windows of {winsize}."
    )
    st.caption(
        "IMU plots show accel (m/s²) + gyro (rad/s) after seconds_elapsed matching and 100 Hz alignment "
        "(before package filter + canonicalize)."
    )

    for site in selected_sensors:
        st.plotly_chart(
            plot_imu_preview(timeline, aligned[site], title=f"{site} IMU @ {TARGET_FS:.0f} Hz"),
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
