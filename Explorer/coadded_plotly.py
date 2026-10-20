"""Build the coadded background-spectrum figure for filtered detections."""

import numpy as np
import plotly.graph_objects as go


def coadd(data, key, wl_dim=512, spectra_root="E:/nightglow/spectra_out"):
    """Coadd valid per-scan spectra and return ``(figure, wavelength, sum)``.

    ``data`` may be a pandas DataFrame or an iterable of mappings. Missing
    per-orbit NPZ files are skipped, and a labeled empty figure is returned
    when no matching spectra are available.
    """
    peaks = [193, 199.5, 206, 216, 228, 240, 289]

    if hasattr(data, "drop_duplicates"):
            data = data.drop_duplicates(subset=["orbit_number", "scan"])

    bg = np.zeros(wl_dim, dtype=float)
    spectra_counts = 0
    wl = None

    #Works for pandas DataFrame or list of dicts
    if hasattr(data, "iterrows"):
        row_iter = (row for _, row in data.iterrows())
    else:
        row_iter = iter(data)

    for row in row_iter:

        o = int(row["orbit_number"])
        s = int(row["scan"])
        o_block = (o // 100) * 100

        spectra_path = f"{spectra_root}/{o_block:05d}/orbit{o:05d}_spectra.npz"

        try:
            spectra_data = np.load(
                spectra_path,
                allow_pickle=True
            )
        except FileNotFoundError:
            print(f"Spectra file not found: {spectra_path}")
            continue

        items = spectra_data["items"]

        # Select appropriate spectra
        if key == "Anomalous":
            good = [
                it for it in items
                if (
                    it["background_kR"].size == wl_dim
                    and it["flag"] == 1
                    and it["scan"] == s
                    and (it["co_average_kR"] / it["co2p_average_kR"]) >= 12.7
                )
            ]

        elif key == "Normal":
            good = [
                it for it in items
                if (
                    it["background_kR"].size == wl_dim
                    and it["flag"] == 1
                    and it["scan"] == s
                    and (it["co_average_kR"] / it["co2p_average_kR"]) < 12.7
                )
            ]

        else:  # All
            good = [
                it for it in items
                if (
                    it["background_kR"].size == wl_dim
                    and it["flag"] == 1
                    and it["scan"] == s
                )
            ]

        if not good:
            continue

        #Set wavelength grid from first valid spectrum
        if wl is None:
            wl = good[0]["wavelength_nm"]

        #Coadd spectra
        stack = np.stack([it["background_kR"] for it in good], axis=0)

        bg += np.nansum(stack, axis=0)

        spectra_counts += len(good)

    #If there happen to be no files, return blank fig
    if wl is None:
        fig = go.Figure()

        fig.add_annotation(
            text=(
                f"No spectra with {wl_dim} wavelength bins found "
                f"for {key} detections in this orbit range."),
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=18)
        )

        fig.update_layout(
            title=f"Coadded Background Spectra",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            paper_bgcolor="#080c16",
            plot_bgcolor="#101827",
            font=dict(color="#dbe5f0",),
            title_font=dict(color="#ffffff",),
            autosize=True,
        )

        return fig, None, None
    
    #Plotly figure if everything works
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=wl,
            y=bg,
            mode="lines",
            name="Coadded Background",
            hovertemplate=(
                "Wavelength: %{x:.2f} nm<br>"
                "Intensity: %{y:.3g}"
                "<extra></extra>"
            )
        )
    )

    #Add vertical lines for peaks
    for peak in peaks:
        fig.add_vline(x=peak, line=dict(color="white", width=2, dash="dot"), opacity=0.5)

    fig.update_layout(
        title=f"Coadded Background Spectra (n={spectra_counts})",
        xaxis_title="Wavelength (nm)",
        yaxis=dict(
            visible=False
        ),
        showlegend=False,
        paper_bgcolor="#080c16",
        plot_bgcolor="#101827",
        font=dict(color="#dbe5f0",),
        title_font=dict(color="#ffffff",),
        autosize=True,
    )

    fig.update_xaxes(
        gridcolor="#263348",
        linecolor="#34445c",
    )

    fig.update_yaxes(
        gridcolor="#263348",
        linecolor="#34445c",
    )

    return fig, wl, bg
