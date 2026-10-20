"""Local-time distribution plots."""

import numpy as np
import plotly.graph_objects as go

def plot_lt(data):
    """Return a one-hour-bin histogram of detection local times."""
    #Plot local time histogram
    x = data['local_time']
    n = len(x)

    #Local time bins
    lt_edges_deg = np.arange(0, 25, 1)
    #Actual counts
    lt_counts, _ = np.histogram(data['local_time'], bins=lt_edges_deg)
    N = lt_counts.sum()
    widths = np.diff(lt_edges_deg)
    bin_centers = 0.5 * (lt_edges_deg[:-1] + lt_edges_deg[1:])

    fig1 = go.Figure()
    fig1.add_trace(
        go.Bar(
            x=bin_centers,
            y=lt_counts,
            width=widths,
            opacity=0.6,
            marker=dict(
                line=dict(
                    color="black",
                    width=1
                )
            ),
            error_y=dict(
                type="data",
                array=np.sqrt(lt_counts),
                visible=True
            ),
            hovertemplate=(
                "Local Time: %{x:.0f} hours<br>"
                "Counts: %{y}"
                "<extra></extra>"
            )
        )
    )

    fig1.update_layout(
        title=f"Local Time Distribution",
        xaxis_title="Local Time (hours)",
        yaxis_title="Counts",
        paper_bgcolor="#080c16",
        plot_bgcolor="#101827",
        font=dict(color="#dbe5f0",),
        title_font=dict(color="#ffffff",),
        bargap=.05,
        autosize=True,
    )

    fig1.update_yaxes(
        gridcolor="#263348",
        linecolor="#34445c",
    )

    fig1.update_xaxes(
        range=[0, 24],
        gridcolor="#263348",
        linecolor="#34445c",
    )

    return fig1
