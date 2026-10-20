"""Solar-zenith-angle distribution plots."""

import numpy as np
import plotly.graph_objects as go


def sza_hist(data, key):
    """Compare observed SZA counts with the spherical-geometry expectation."""
    SZA_min = 105
    plot_edges_deg = np.arange(100, 181, 5)

    # Observed data should match allowed range
    sza = data["sza"]
    sza = sza[sza >= SZA_min]


    counts, _ = np.histogram(sza, bins=plot_edges_deg)
    N = counts.sum()

    # Geometry uses true lower limit in first bin
    geom_edges_deg = plot_edges_deg.astype(float).copy()
    geom_edges_deg[0] = SZA_min

    edges_rad = np.deg2rad(geom_edges_deg)

    geom_w = (np.cos(edges_rad[:-1]) - np.cos(edges_rad[1:]))
    geom_w /= geom_w.sum()

    expected = geom_w * N

    bin_centers = 0.5 * (plot_edges_deg[:-1] + plot_edges_deg[1:])

    bin_widths = np.diff(plot_edges_deg)

    fig = go.Figure()

    # Actual detections
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=counts,
            width=bin_widths,
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
                array=np.sqrt(counts),
                visible=True
            )
        )
    )

    # Expected uniform geometry distribution
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=expected,
            mode="lines",
            name="Uniform (geometry)",
            line=dict(
                color="white",
                width=2,
                dash="dash"
            )
        )
    )

    fig.update_layout(
        title=f"SZA Distribution",
        xaxis_title="SZA (°)",
        yaxis_title="Counts",
        autosize=True,
        bargap=0,
        legend=dict(
            x=0.02,
            y=0.98
        ),
        paper_bgcolor="#080c16",
        plot_bgcolor="#101827",
        font=dict(color="#dbe5f0",),
        title_font=dict(color="#ffffff",),
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=np.arange(100, 181, 10),
        range=[100, 180],
        gridcolor="#263348",
        linecolor="#34445c",
    )

    fig.update_yaxes(
        gridcolor="#263348",
        linecolor="#34445c",
    )

    return fig
