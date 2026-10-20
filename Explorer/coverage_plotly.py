"""Temporal-coverage and seasonal-ratio plots."""

import numpy as np
import plotly.graph_objects as go
import datetime as dt

def plot_coverage(data):
    """Return date-versus-season and CO/CO2+ ratio-versus-season figures."""
    #Plot year vs solar longitude
    x = [dt.datetime.strptime(date, '%Y-%b-%d') for date in data['date']]
    y = data['solar_longitude']
    orbit_vals = np.asarray(data["orbit_number"])

    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='markers',
            customdata=orbit_vals,
            hovertemplate=(
                "Orbit: %{customdata}"
                "<extra></extra>"
            )
        )
    )

    fig1.update_layout(
        title=f"Coverage Plot",
        xaxis_title="Date",
        yaxis_title="Solar Longitude",
        paper_bgcolor="#080c16",
        plot_bgcolor="#101827",
        font=dict(color="#dbe5f0",),
        title_font=dict(color="#ffffff",),
        autosize=True,
    )

    fig1.update_yaxes(
        range=[0, 360],
        gridcolor="#263348",
        linecolor="#34445c",
    )

    fig1.update_xaxes(
        gridcolor="#263348",
        linecolor="#34445c",
    )

    #Plot solar longitude vs ratio
    x = data['solar_longitude']
    y = data['CO/CO2p']

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=y, mode='markers'))
    fig2.update_layout(
        title=f"CO/CO2p Ratio vs Solar Longitude",
        xaxis_title="Solar Longitude",
        yaxis_title="CO/CO2p Ratio",
        paper_bgcolor="#080c16",
        plot_bgcolor="#101827",
        font=dict(color="#dbe5f0",),
        title_font=dict(color="#ffffff",),
        autosize=True,
    )

    fig2.update_yaxes(
        range=[-600, 1000],
        gridcolor="#263348",
        linecolor="#34445c",
    )

    fig2.update_xaxes(
        gridcolor="#263348",
        linecolor="#34445c",
    )

    return fig1, fig2
