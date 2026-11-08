import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from utils.utils import get_subject_index_test, get_activity_index_test


def plot_kinematic_predictions(
    y_true,
    y_pred,
    label,
    selected_subject,
    selected_activities,
    selected_index_to_plot,
    n_windows=3,
):
    """Plot n_windows consecutive samples for Hip/Knee/Ankle (deg).

    Layout: one block of rows per activity; within each activity, one row per window
    and three columns (Hip, Knee, Ankle).
    """
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

    n_rows = n_activities * n_windows
    subplot_titles = []
    for activity in selected_activities:
        for w in range(n_windows):
            win_idx = selected_index_to_plot + w
            for j in range(3):
                if j == 1:
                    subplot_titles.append(f"{activity} — window {win_idx}")
                else:
                    subplot_titles.append(" ")

    fig = make_subplots(
        rows=n_rows,
        cols=3,
        subplot_titles=subplot_titles,
        vertical_spacing=min(0.08, 0.4 / max(n_rows, 1)),
        horizontal_spacing=0.06,
    )

    show_legend = True
    for a, activity in enumerate(selected_activities):
        activity_to_evaluate = exact_activity_names[activity]
        subject_index = get_subject_index_test(label, selected_subject)
        activity_index = get_activity_index_test(label, activity_to_evaluate)
        selected_index = sorted(set(subject_index) & set(activity_index))

        for w in range(n_windows):
            row = a * n_windows + w + 1
            win_idx = selected_index_to_plot + w
            if win_idx >= len(selected_index):
                for j, joint_name in enumerate(joint_names):
                    fig.update_yaxes(title_text=f"{joint_name} (deg)", row=row, col=j + 1)
                    fig.update_xaxes(title_text="Time (s)", row=row, col=j + 1)
                continue

            y_pred_to_plot = y_pred[selected_index][win_idx]
            y_true_to_plot = y_true[selected_index][win_idx]
            t_s = np.arange(y_true_to_plot.shape[0]) / 100.0

            for j, joint_name in enumerate(joint_names):
                fig.append_trace(
                    go.Scatter(
                        x=t_s,
                        y=-y_true_to_plot[:, j],
                        mode="lines",
                        name="True",
                        legendgroup="True",
                        showlegend=show_legend,
                        line=dict(color=colors[5], dash="solid"),
                    ),
                    row=row,
                    col=j + 1,
                )
                fig.append_trace(
                    go.Scatter(
                        x=t_s,
                        y=-y_pred_to_plot[:, j],
                        mode="lines",
                        name="Prediction",
                        legendgroup="Prediction",
                        showlegend=show_legend,
                        line=dict(color=colors[8], dash="dot"),
                    ),
                    row=row,
                    col=j + 1,
                )
                show_legend = False
                fig.update_yaxes(title_text=f"{joint_name} (deg)", row=row, col=j + 1)
                fig.update_xaxes(title_text="Time (s)", row=row, col=j + 1)

    fig.update_layout(height=max(420, 280 * n_rows), width=1000)
    return fig
