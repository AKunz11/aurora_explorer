"""Nightside projections in geographic and Mars Solar Orbital coordinates."""

import numpy as np
import plotly.graph_objects as go

def plot_nightside(data):
    """Return MSO polar, geographic Cartesian, and MSO Cartesian figures.

    The polar projection places the antisolar point at the origin using
    ``x=-cos(latitude)*sin(longitude)`` and ``y=sin(latitude)``.
    """
    lat = np.asarray(data['latitude'], dtype=float)
    ls  = np.asarray(data['solar_longitude'], dtype=float)
    lt = np.asarray(data['local_time'], dtype=float)  # must exist
    mso_lon= np.asarray(data['mso_longitude'], dtype=float)
    mso_lat= np.asarray(data['mso_latitude'], dtype=float)

    if 'hours_from_midnight' in data.columns:
        hfm = np.asarray(data['hours_from_midnight'], dtype=float)
    else:
        lt = np.asarray(data['local_time'], dtype=float)  # must exist
        hfm = np.where(lt <= 12, lt, lt - 24)

    ok = np.isfinite(lat) & np.isfinite(hfm)
    orbit_vals = np.asarray(data["orbit_number"])[ok]
    lat = lat[ok]
    hfm = hfm[ok]
    ls = ls[ok]
    lt = lt[ok]
    mso_lon = mso_lon[ok]
    mso_lat = mso_lat[ok]

    #Polar plot (circle one)
    #Antisolar point in center, looking down at Mars nightside
    mso_lon_rad = np.deg2rad(mso_lon)
    mso_lat_rad = np.deg2rad(mso_lat)
    x = -np.cos(mso_lat_rad) * np.sin(mso_lon_rad)
    y = np.sin(mso_lat_rad)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=5,
                color=orbit_vals,
                colorscale="Viridis",
                colorbar=dict(
                    title='Orbit')
                    ),
            showlegend=False,
            customdata=orbit_vals,
            hovertemplate=(
                "Orbit: %{customdata}"
                "<extra></extra>"
            )
        )
    )

    #SZA rings
    szas = [160, 140, 120, 110, 100, 90]
    for sza in szas:
        sza_rad = np.deg2rad(sza)
        #Generate rings in the polar space
        x_ring = np.cos(np.linspace(0, 2 * np.pi, 100)) * np.sin(sza_rad)
        y_ring = -np.sin(np.linspace(0, 2 * np.pi, 100)) * np.sin(sza_rad)
        fig.add_trace(
            go.Scatter(
                x=x_ring,
                y=y_ring,
                mode='lines',
                line=dict(
                    color='white',
                    dash='dot'),
                name=f'SZA {sza}°',
                opacity=0.25,
                showlegend=False,
                hovertemplate=(
                    f"SZA: {sza:.0f}°<br>"
                    "<extra></extra>"
                )
            )
        )

    #MSO contours
    mso_lats = np.arange(-80, 81, 20)

    lon_contour = np.linspace(90, 270, 200) #lon is needed to bound to the circle interior
    lon_rad = np.deg2rad(lon_contour)

    for mso_lat_contour in mso_lats:
        lat_rad = np.deg2rad(mso_lat_contour)

        x_contour = -np.cos(lat_rad) * np.sin(lon_rad)
        y_contour = np.full_like(x_contour, np.sin(lat_rad))

        fig.add_trace(
            go.Scatter(
                x=x_contour,
                y=y_contour,
                mode="lines",
                line=dict(
                    color="white",
                    width=1,
                    dash="dash"
                ),
                opacity=0.25,
                showlegend=False,
                hovertemplate=(
                    f"MSO Latitude: {mso_lat_contour}°"
                    "<extra></extra>"
                )
            )
        )
        


    #Add antisolar point (0,0) in polar space
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode='markers',
            marker=dict(
                size=5,
                color='white'
                ),
            name='Antisolar Point',
            opacity=0.5,
            showlegend=False
        )
    )

    fig.update_layout(
        title='Nightside Observations (MSO Coords)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="#080c16",
        plot_bgcolor="#101827",
        font=dict(color="#dbe5f0",),
        title_font=dict(color="#ffffff",),
        autosize=True,
    )

    fig.update_xaxes(
        gridcolor="#263348",
        linecolor="#34445c",
        range=[-1.05, 1.05],
        visible=False,
    )

    fig.update_yaxes(
        gridcolor="#263348",
        linecolor="#34445c",
        range=[-1.05, 1.05],
        visible=False,
        scaleanchor="x",
        scaleratio=1,
    )


    #Cartesian version
    fig_cartesian = go.Figure()
    fig_cartesian.add_trace(
        go.Scatter(
            x=hfm,
            y=lat,
            mode='markers',
            marker=dict(
                size=5,
                color=orbit_vals,
                colorscale="Viridis",
                colorbar=dict(
                    title='Orbit'
                )
            ),
            showlegend=False,
            customdata=orbit_vals,
            hovertemplate=(
                "Orbit: %{customdata}"
                "<extra></extra>"
            )
        )
    )

    fig_cartesian.update_layout(
        title='Nightside Observations (Cartesian)',
        xaxis_title='Hours from Midnight',
        yaxis_title='Latitude',
    )

    fig_cartesian.update_layout(
        paper_bgcolor="#080c16",
        plot_bgcolor="#101827",
        font=dict(color="#dbe5f0",),
        title_font=dict(color="#ffffff",),
        autosize=True,
    )

    fig_cartesian.update_xaxes(
        gridcolor="#263348",
        linecolor="#34445c",
    )

    fig_cartesian.update_yaxes(
        gridcolor="#263348",
        linecolor="#34445c",
    )

    fig_cartesian_mso = go.Figure()
    fig_cartesian_mso.add_trace(
        go.Scatter(
            x=hfm,
            y=mso_lat,
            mode='markers',
            marker=dict(
                size=5,
                color=orbit_vals,
                colorscale="Viridis",
                colorbar=dict(
                    title='Orbit'
                )
            ),
            showlegend=False,
            customdata=orbit_vals,
            hovertemplate=(
                "Orbit: %{customdata}"
                "<extra></extra>"
            )
        )
    )

    fig_cartesian_mso.update_layout(
        title='Nightside Observations (Cartesian, MSO)',
        xaxis_title='Hours from Midnight',
        yaxis_title='MSO Latitude',
    )

    fig_cartesian_mso.update_layout(
        paper_bgcolor="#080c16",
        plot_bgcolor="#101827",
        font=dict(color="#dbe5f0",),
        title_font=dict(color="#ffffff",),
        autosize=True,
    )

    fig_cartesian_mso.update_xaxes(
        gridcolor="#263348",
        linecolor="#34445c",
    )

    fig_cartesian_mso.update_yaxes(
        gridcolor="#263348",
        linecolor="#34445c",
    )



    return fig, fig_cartesian, fig_cartesian_mso
