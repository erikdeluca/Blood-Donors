import plotly.graph_objects as go
from app import config


def plot_donor_stepped_line(df_long, pred_data, T_dict):
    fig = go.Figure()

    # background grid line for the trend
    fig.add_trace(
        go.Scatter(
            x=df_long["year"],
            y=df_long["donations"],
            mode="lines",
            line_shape="hvh",
            line=dict(color="lightgrey", width=2),
            name=T_dict["legend_trend"],
            hoverinfo="skip",
        )
    )

    # points for the states
    for state in sorted(df_long["state_cat"].unique()):
        subset = df_long[df_long["state_cat"] == state]
        state_id = int(state)
        label_name = T_dict["state_names"].get(state_id, f"State {state_id}")
        point_color = config.COLOR_MAP.get(state, "black")
        fig.add_trace(
            go.Scatter(
                x=subset["year"],
                y=subset["donations"],
                mode="markers",
                marker=dict(
                    size=12, color=point_color, line=dict(width=2, color="white")
                ),
                name=label_name,
                hovertemplate=f"<b>{T_dict['tooltip_year']}:</b> %{{x}}<br>"
                f"<b>{T_dict['tooltip_don']}:</b> %{{y}}<br>"
                f"<b>Status:</b> {label_name}",
            )
        )

    # prediction
    last_year = df_long["year"].max()
    last_val = df_long.loc[df_long["year"] == last_year, "donations"].values[0]
    next_year = last_year + 1
    predicted_val = pred_data["expected_next"]

    fig.add_trace(
        go.Scatter(
            x=[last_year, next_year],
            y=[last_val, predicted_val],
            mode="lines",
            line=dict(color="gray", width=2, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[next_year],
            y=[predicted_val],
            mode="markers",
            marker=dict(
                size=14,
                symbol="square",
                color="#e377c2",
                line=dict(width=2, color="white"),
            ),
            name=T_dict["legend_pred"],
            hovertemplate=f"<b>{T_dict['tooltip_year']}:</b> {next_year}<br><b>{T_dict['tooltip_exp']}:</b> %{{y:.3f}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=T_dict["chart_title"],
        xaxis_title=T_dict["axis_x"],
        yaxis_title=T_dict["axis_y"],
        dragmode=False,
        xaxis=dict(tickmode="linear", showgrid=False, fixedrange=True),
        yaxis=dict(
            dtick=1,
            range=[-0.5, max(4.5, predicted_val + 0.5)],
            showgrid=True,
            gridcolor="#eee",
            fixedrange=True,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig
