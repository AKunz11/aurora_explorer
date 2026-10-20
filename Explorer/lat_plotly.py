"""Latitude distribution plots for auroral detections."""

import numpy as np
import plotly.graph_objects as go


def latitude_hist(data, key):
    """Compare observed latitude counts with a uniform spherical distribution."""

    #Latitude bins
    lat_edges_deg = np.arange(-90, 91, 20)

    #Expected distribution from spherical geometry
    le_rad = np.deg2rad(lat_edges_deg)

    geom_lat_w = (np.sin(le_rad[1:]) - np.sin(le_rad[:-1]))
    geom_lat_w /= geom_lat_w.sum()

    #Actual counts
    lat_counts, _ = np.histogram(data["latitude"], bins=lat_edges_deg)

    N = lat_counts.sum()

    expected_lat = geom_lat_w * N

    widths = np.diff(lat_edges_deg)

    bin_centers = 0.5 * (lat_edges_deg[:-1] + lat_edges_deg[1:])

    fig = go.Figure()

    #Actual detections
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=lat_counts,
            width=widths,
            name="Detections",
            opacity=0.6,
            marker=dict(
                line=dict(
                    color="black",
                    width=1
                )
            ),
            error_y=dict(
                type="data",
                array=np.sqrt(lat_counts),
                visible=True
            ),
            hovertemplate=(
                "Latitude: %{x:.0f}°<br>"
                "Detections: %{y}"
                "<extra></extra>"
            )
        )
    )

    #Expected uniform distribution
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=expected_lat,
            mode="lines",
            name="Uniform (geometry)",
            line=dict(
                color="white",
                width=2,
                dash="dash"
            ),
            hovertemplate=(
                "Latitude: %{x:.0f}°<br>"
                "Expected: %{y:.1f}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=f"Latitude Distribution",
        xaxis_title="Latitude (°)",
        yaxis_title="Counts",
        bargap=0,
        legend=dict(
            x=0.02,
            y=0.98
        ),
        paper_bgcolor="#080c16",
        plot_bgcolor="#101827",
        font=dict(color="#dbe5f0",),
        title_font=dict(color="#ffffff",),
        autosize=True,
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=np.arange(-90, 91, 20),
        range=[-90, 90],
        gridcolor="#263348",
        linecolor="#34445c",
    )

    fig.update_yaxes(
        gridcolor="#263348",
        linecolor="#34445c",
    )

    return fig
