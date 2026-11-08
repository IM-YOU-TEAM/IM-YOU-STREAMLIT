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


def plot_kinematic_predictions(
    y_true,
    y_pred,
    label,
    selected_subject,
    selected_activities,
    selected_index_to_plot,
    n_windows=3,
    overlap=0.5,
    sample_rate=100.0,
):
    """Stitch n_windows into one continuous Hip/Knee/Ankle (deg) plot per activity."""
    exact_activity_names = {
        "LevelGround Walking": "levelground1",
        "Ramp Ascent": "rampascent",
        "Ramp Descent": "rampdescent",
        "Stair Ascent": "stairascent",
        "Stair Descent": "stairdescent",
    }
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
    for a, activity in enumerate(selected_activities):
        activity_to_evaluate = exact_activity_names[activity]
        subject_index = get_subject_index_test(label, selected_subject)
        activity_index = get_activity_index_test(label, activity_to_evaluate)
        selected_index = sorted(set(subject_index) & set(activity_index))

        start = selected_index_to_plot
        end = min(start + n_windows, len(selected_index))
        if start >= len(selected_index) or end <= start:
            for j, joint_name in enumerate(joint_names):
                fig.update_yaxes(title_text=f"{joint_name} (deg)", row=a + 1, col=j + 1)
                fig.update_xaxes(title_text="Time (s)", row=a + 1, col=j + 1)
            continue

        true_windows = y_true[selected_index][start:end]
        pred_windows = y_pred[selected_index][start:end]
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

    fig.update_layout(
        height=max(420, 360 * n_activities),
        width=1000,
        title=f"Stitched kinematics ({n_windows} windows, {int(overlap * 100)}% overlap)",
    )
    return fig
