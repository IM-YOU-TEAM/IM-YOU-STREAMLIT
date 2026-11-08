import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from utils.utils import get_subject_index_test, get_activity_index_test


def stitch_windows(windows: np.ndarray, overlap: float = 0.5) -> np.ndarray:
    """Average overlapping windows into a continuous sequence (same as phone IMU POC)."""
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


ACTIVITY_NAME_MAP = {
    "LevelGround Walking": "levelground1",
    "Ramp Ascent": "rampascent",
    "Ramp Descent": "rampdescent",
    "Stair Ascent": "stairascent",
    "Stair Descent": "stairdescent",
}


def selected_window_indices(label, selected_subject, activity_display_name):
    activity_to_evaluate = ACTIVITY_NAME_MAP[activity_display_name]
    subject_index = get_subject_index_test(label, selected_subject)
    activity_index = get_activity_index_test(label, activity_to_evaluate)
    return sorted(set(subject_index) & set(activity_index))


def plot_imu_site(time_s, imu_6, title, sample_rate=100.0):
    """Plot one site's accel XYZ + gyro XYZ (same layout as test_phone_imu)."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Acceleration", "Gyroscope"],
    )
    for i, name in enumerate(["Ax", "Ay", "Az"]):
        fig.add_trace(go.Scatter(x=time_s, y=imu_6[:, i], mode="lines", name=name), row=1, col=1)
    for i, name in enumerate(["Gx", "Gy", "Gz"]):
        fig.add_trace(go.Scatter(x=time_s, y=imu_6[:, i + 3], mode="lines", name=name), row=2, col=1)
    fig.update_yaxes(title_text="Accel", row=1, col=1)
    fig.update_yaxes(title_text="Gyro", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(height=420, title=title, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_raw_imu_for_selection(
    x_windows,
    label,
    selected_subject,
    selected_activities,
    selected_sensors,
    features_per_sensor=6,
    overlap=0.5,
    sample_rate=100.0,
):
    """
    Stitch raw (pre-canonicalize) IMU windows for subject+activity and
    return one accel/gyro figure per site per activity.
    """
    figures = []
    if x_windows is None or len(selected_activities) == 0:
        return figures
    x_windows = np.asarray(x_windows)
    expected_c = len(selected_sensors) * features_per_sensor
    if x_windows.ndim != 3 or x_windows.shape[-1] != expected_c:
        return figures

    for activity in selected_activities:
        idxs = selected_window_indices(label, selected_subject, activity)
        if not idxs:
            continue
        stitched = stitch_windows(x_windows[idxs], overlap=overlap)
        t_s = np.arange(stitched.shape[0]) / sample_rate
        for s, site in enumerate(selected_sensors):
            start = s * features_per_sensor
            imu_6 = stitched[:, start : start + features_per_sensor]
            figures.append(
                plot_imu_site(
                    t_s,
                    imu_6,
                    title=f"{selected_subject} / {activity} / {site} IMU "
                    f"(raw windows, before canonicalize)",
                    sample_rate=sample_rate,
                )
            )
    return figures


def plot_kinematic_predictions(
    y_true,
    y_pred,
    label,
    selected_subject,
    selected_activities,
    selected_index_to_plot=0,
    n_windows=None,
    overlap=0.5,
    sample_rate=100.0,
):
    """Stitch windows into continuous Hip/Knee/Ankle (deg) plots per activity.

    n_windows=None stitches the entire subject+activity selection.
    """
    colors = px.colors.sequential.Turbo
    joint_names = ["Hip", "Knee", "Ankle"]
    n_activities = len(selected_activities)
    if n_activities == 0:
        fig = go.Figure()
        fig.update_layout(title="Select at least one activity")
        return fig

    fig = make_subplots(
        rows=n_activities,
        cols=3,
        subplot_titles=[
            title
            for activity in selected_activities
            for title in (f"{activity} — Hip", "Knee", "Ankle")
        ],
        vertical_spacing=min(0.12, 0.35 / max(n_activities, 1)),
        horizontal_spacing=0.06,
    )

    show_legend = True
    used_windows = 0
    for a, activity in enumerate(selected_activities):
        selected_index = selected_window_indices(label, selected_subject, activity)

        if n_windows is None:
            start, end = 0, len(selected_index)
        else:
            start = selected_index_to_plot
            end = min(start + n_windows, len(selected_index))

        if start >= len(selected_index) or end <= start:
            for j, joint_name in enumerate(joint_names):
                fig.update_yaxes(title_text=f"{joint_name} (deg)", row=a + 1, col=j + 1)
                fig.update_xaxes(title_text="Time (s)", row=a + 1, col=j + 1)
            continue

        true_windows = y_true[selected_index][start:end]
        pred_windows = y_pred[selected_index][start:end]
        used_windows = max(used_windows, end - start)
        true_stitched = stitch_windows(true_windows, overlap=overlap)
        pred_stitched = stitch_windows(pred_windows, overlap=overlap)
        t_s = np.arange(true_stitched.shape[0]) / sample_rate

        for j, joint_name in enumerate(joint_names):
            fig.append_trace(
                go.Scatter(
                    x=t_s,
                    y=-true_stitched[:, j],
                    mode="lines",
                    name="True",
                    legendgroup="True",
                    showlegend=show_legend,
                    line=dict(color=colors[5], dash="solid"),
                ),
                row=a + 1,
                col=j + 1,
            )
            fig.append_trace(
                go.Scatter(
                    x=t_s,
                    y=-pred_stitched[:, j],
                    mode="lines",
                    name="Prediction",
                    legendgroup="Prediction",
                    showlegend=show_legend,
                    line=dict(color=colors[8], dash="dot"),
                ),
                row=a + 1,
                col=j + 1,
            )
            show_legend = False
            fig.update_yaxes(title_text=f"{joint_name} (deg)", row=a + 1, col=j + 1)
            fig.update_xaxes(title_text="Time (s)", row=a + 1, col=j + 1)

    span = "all windows" if n_windows is None else f"{used_windows} windows"
    fig.update_layout(
        height=max(420, 360 * n_activities),
        width=1000,
        title=f"Stitched kinematics ({span}, {int(overlap * 100)}% overlap)",
    )
    return fig
