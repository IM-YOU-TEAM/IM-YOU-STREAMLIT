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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

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
        flip_knee = st.checkbox("Flip knee sign (display flexion +)", value=True)

        st.header("Units (if not in CSV)")
        accel_units = st.selectbox("Acceleration units", ["m/s^2", "g"], index=0)
        gyro_units = st.selectbox("Gyroscope units", ["rad/s", "deg/s"], index=0)

    session_dir: Optional[Path] = None
    prep_notes: List[dict] = []
    ready = False
    selected_pair = None

    if input_mode == "sensorlogData library":
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
    if not run_clicked:
        st.info(
            "Uses **NEW** 60/100 Hz shank+thigh weights. "
            "Choose a paired trial from `sensorlogData`, or upload thigh+shank / combined CSVs."
        )
        return

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
            if build["mode"] == "sensorlog":
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
            elif build["mode"] == "two":
                thigh_df, t_info = dataframe_to_phone_schema(
                    build["thigh_raw"], build["thigh_map"], build["t_acc"], build["t_gyro"], target
                )
                shank_df, s_info = dataframe_to_phone_schema(
                    build["shank_raw"], build["shank_map"], build["s_acc"], build["s_gyro"], target
                )
                prep_notes = [{"file": "thigh", **t_info}, {"file": "shank", **s_info}]
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
    angles = result["angles"]
    st.plotly_chart(plot_angles(angles, flip_knee=flip_knee), use_container_width=True)

    if result["activities"] is not None:
        st.subheader("Activity windows (100 Hz classifier)")
        st.dataframe(result["activities"], use_container_width=True)
        st.download_button(
            "Download activity_windows.csv",
            result["activities"].to_csv(index=False).encode("utf-8"),
            file_name="activity_windows.csv",
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
