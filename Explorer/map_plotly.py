"""Two-dimensional Mars magnetic-topology and detection map."""

import numpy as np
import plotly.graph_objects as go


def plot_map(data, map_img, orb_start, orb_end, key):
    """Overlay filtered detections on the magnetic-topology raster."""
    #Load magnetic topology map
    map_img = np.asarray(map_img, dtype=float).copy()

    data=data[
        (data["orbit_number"] >= orb_start) &
        (data["orbit_number"] < orb_end)]

    orbit_vals = np.asarray(data["orbit_number"])

    orbit_min = np.min(orbit_vals)
    orbit_max = np.max(orbit_vals)

    #Avoid log10 problems
    map_img_plot = np.where(map_img > 0, map_img, np.nan)

    #Convert to log10 because Plotly doesnt have logNorm

    log_map = np.log10(map_img_plot)
    nan_mask = np.where(np.isnan(map_img_plot), 1, np.nan)

    fig = go.Figure()

    #White background for NaN / zero topology values (the "set_bad" from reg function)
    fig.add_trace(
        go.Heatmap(
            z=nan_mask,
            x=np.linspace(0, 360, map_img.shape[1]),
            y=np.linspace(-90, 90, map_img.shape[0]),
            colorscale=[
                [0, "white"],
                [1, "white"]
            ],
            showscale=False,
            hoverinfo="skip"
        )
    )

    #Magnetic topology background
    fig.add_trace(
        go.Heatmap(
            z=log_map,
            x=np.linspace(0, 360, map_img.shape[1]),
            y=np.linspace(-90, 90, map_img.shape[0]),
            colorscale="Greys",
            zmin=np.log10(0.001),
            zmax=np.log10(1),
            colorbar=dict(
                title="Magnetic Topology",
                tickvals=[
                    np.log10(0.001),
                    np.log10(0.01),
                    np.log10(0.1),
                    np.log10(1)
                ],
                ticktext=[
                    "0.001",
                    "0.01",
                    "0.1",
                    "1"
                ],
                x=1.02
            ),
            hovertemplate=(
                "Longitude: %{x:.1f}°<br>"
                "Latitude: %{y:.1f}°<br>"
                "Topology: %{customdata:.4f}"
                "<extra></extra>"
            ),
            customdata=map_img_plot
        )
    )

    # Detection locations
    fig.add_trace(
        go.Scatter(
            x=data["longitude"],
            y=data["latitude"],
            mode="markers",
            marker=dict(
                size=5,
                color=orbit_vals,
                colorscale="Rainbow",
                cmin=orbit_min,
                cmax=orbit_max,
                colorbar=dict(
                    title="Orbit number",
                    orientation="h",
                    x=0.5,
                    xanchor="center",
                    y=-0.25,
                    len=0.7
                ),
                line=dict(
                    width=0
                )
            ),
            customdata=orbit_vals,
            name="Detections",
            hovertemplate=(
                "Longitude: %{x:.2f}°<br>"
                "Latitude: %{y:.2f}°<br>"
                "Orbit: %{customdata}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=f"Location of Detections (n={len(data)})",
        xaxis_title="Longitude (°)",
        yaxis_title="Latitude (°)",
        autosize=True,
        xaxis=dict(
            range=[0, 360],
            dtick=60
        ),
        yaxis=dict(
            range=[-90, 90],
            dtick=30
        ),
        margin=dict(
            l=70,
            r=120,
            t=70,
            b=160
        ),
        paper_bgcolor="#080c16",
        plot_bgcolor="#101827",
        font=dict(color="#dbe5f0",),
        title_font=dict(color="#ffffff",),
    )

    fig.update_xaxes(
        gridcolor="#263348",
        linecolor="#34445c",
        range=[0, 360],
    )

    fig.update_yaxes(
        gridcolor="#263348",
        linecolor="#34445c",
        range=[-90, 90],
    )

    return fig
