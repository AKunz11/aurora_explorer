"""Interactive three-dimensional Mars detection map."""

import numpy as np
import plotly.graph_objects as go

def plot_sphere(data, map_img):
    """Plot detections just above a topology-textured unit sphere."""
    map_img = np.asarray(map_img, dtype=float).copy()

    # Rotate map 180 degrees in longitude
    map_img = np.roll(map_img, map_img.shape[1] // 2, axis=1)

    # Downsample for interactive 3D rendering
    step = 5
    map_img = map_img[::step, ::step]

    # Keep original values for hover
    map_img_plot = np.asarray(map_img, dtype=float)

    # For plotting, make zero / invalid values equal the bottom of the colorscale
    # instead of NaN. This keeps the actual sphere intact.
    map_img_safe = np.where(np.isfinite(map_img_plot) & (map_img_plot > 0), map_img_plot, 0.001)

    #Prevent anything below plotting minimum
    map_img_safe = np.clip(map_img_safe, 0.001, 1.0)
    log_map = np.log10(map_img_safe)

    orbit_vals = np.asarray(data["orbit_number"])
    orbit_min = np.min(orbit_vals)
    orbit_max = np.max(orbit_vals)

    altitudes = np.asarray(data["altitude"])

    n_lat, n_lon = map_img.shape

    #Center on lon=0
    lon_sphere = np.linspace(-180, 180, n_lon)
    lat = np.linspace(-90, 90, n_lat)

    lon_grid, lat_grid = np.meshgrid(lon_sphere, lat)

    lon_display_grid = lon_grid % 360
    surface_customdata = np.dstack((
        lon_display_grid,
        lat_grid,
        map_img_plot
    ))

    #Build unit sphere
    lon_rad = np.radians(lon_grid)
    lat_rad = np.radians(lat_grid)

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    fig = go.Figure()

    #Sphere
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=log_map,
            colorscale="Greys",
            cmin=np.log10(0.001),
            cmax=np.log10(1),
            colorbar=dict(
                title="Magnetic Topology",
                tickvals=[
                    np.log10(0.001),
                    np.log10(0.01),
                    np.log10(0.1),
                    np.log10(1)
                ],
                x=1.02
            ),
            hoverinfo="skip",
            lighting=dict(
                ambient=0.9,
                diffuse=0.5,
                roughness=1
            ),
            showscale=True
        )
    )

    #Latitude / Longitude grid
    grid_r = 1.003  #Slightly above surface
    grid_color = "rgba(100, 160, 220, 0.45)"

    #Latitude lines every 30 degrees
    for grid_lat in np.arange(-60, 90, 30):

        grid_lon = np.linspace(-180, 180, 181)

        lat_r = np.radians(grid_lat)
        lon_r = np.radians(grid_lon)

        grid_x = grid_r * np.cos(lat_r) * np.cos(lon_r)
        grid_y = grid_r * np.cos(lat_r) * np.sin(lon_r)
        grid_z = grid_r * np.sin(lat_r) * np.ones_like(lon_r)

        fig.add_trace(
            go.Scatter3d(
                x=grid_x,
                y=grid_y,
                z=grid_z,
                mode="lines",
                line=dict(
                    color=grid_color,
                    width=2
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )


    #Longitude lines every 30 degrees
    for grid_lon in np.arange(-180, 180, 30):

        grid_lat = np.linspace(-90, 90, 91)

        lon_r = np.radians(grid_lon)
        lat_r = np.radians(grid_lat)

        grid_x = grid_r * np.cos(lat_r) * np.cos(lon_r)
        grid_y = grid_r * np.cos(lat_r) * np.sin(lon_r)
        grid_z = grid_r * np.sin(lat_r)

        fig.add_trace(
            go.Scatter3d(
                x=grid_x,
                y=grid_y,
                z=grid_z,
                mode="lines",
                line=dict(
                    color=grid_color,
                    width=2
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

    #Detections
    det_lon_display = np.asarray(data["longitude"], dtype=float)
    det_lat = np.asarray(data["latitude"], dtype=float)

    #Convert 0–360 longitude into -180–180 if necessary
    det_lon_sphere = ((det_lon_display + 180) % 360) - 180

    det_lon_rad = np.radians(det_lon_sphere)
    det_lat_rad = np.radians(det_lat)

    #Put marks lightly above sphere so don't disappear
    r = 1.015

    #Put on sphere coords
    det_x = r * np.cos(det_lat_rad) * np.cos(det_lon_rad)
    det_y = r * np.cos(det_lat_rad) * np.sin(det_lon_rad)
    det_z = r * np.sin(det_lat_rad)

    fig.add_trace(
        go.Scatter3d(
            x=det_x,
            y=det_y,
            z=det_z,
            mode="markers",
            marker=dict(
                size=4,
                color=orbit_vals,
                colorscale="Rainbow",
                cmin=orbit_min,
                cmax=orbit_max,
                colorbar=dict(
                    title="Orbit number",
                    x=-0.1,
                    len=0.6
                )
            ),
            customdata=np.column_stack((det_lon_display, det_lat, orbit_vals)),
            hovertemplate=(
                "Longitude: %{customdata[0]:.2f}°<br>"
                "Latitude: %{customdata[1]:.2f}°<br>"
                "Orbit: %{customdata[2]:.0f}"
                "<extra></extra>"
            ),
            name="Detections"
        )   
    )

    #Update layout
    fig.update_layout(
        title=f"Location of Detections (n={len(data)})",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(
                eye=dict(
                    x=1.8,
                    y=0,
                    z=0.3
                )
            ),
            bgcolor="#101827"
        ),

        margin=dict(
            l=20,
            r=100,
            t=70,
            b=20
        ),
        paper_bgcolor="#080c16",
        font=dict(
            color="#dbe5f0"
        ),

        title_font=dict(
            color="#ffffff"
        )
    )

    return fig
