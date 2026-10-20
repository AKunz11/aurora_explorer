"""Create a quick-look grid of L1B detector images for an entire orbit."""

import glob
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
from astropy.io import fits


BASE_L1B = Path("E:/nightglow/b")


def get_l1b_files(orbit, channel="muv"):
    """Return sorted periapse L1B paths for an orbit and detector channel."""
    orbit_str = f"{orbit:05d}"
    folder = BASE_L1B / f"{(orbit // 100) * 100:05d}"

    files = sorted(folder.glob(f"*periapse*orbit{orbit_str}-{channel}*.gz"))

    if len(files) == 0:
        raise FileNotFoundError(f"No L1B {channel.upper()} files found for orbit {orbit} in {folder}")

    return files


def place_detector_images(axes, plot_wavelength, plot_spectra, n_spa):
    """
    Render integrations with the quick-look convention: an inferno colormap,
    black invalid values, and logarithmic normalization from 50 to 5000 DN.
    """

    spec_cmap = plt.get_cmap("inferno").copy()
    spec_cmap.set_bad((0, 0, 0))

    for integration in range(plot_spectra.shape[0]):
        x, y = np.meshgrid(
            np.linspace(
                plot_wavelength[4, 0],
                plot_wavelength[4, -1],
                plot_spectra.shape[2] + 1
            ),
            np.linspace(
                0,
                n_spa,
                plot_spectra.shape[1] + 1
            )
        )

        axes[integration].pcolormesh(
            x,
            y,
            plot_spectra[integration, :, :],
            cmap=spec_cmap,
            norm=colors.LogNorm(vmin=50, vmax=5000),
            rasterized=True
        )

        axes[integration].set_frame_on(False)


def plot_detector(
    orbit, channel="muv", cr_df=None, cr_column=None, plot_title=None, overlay_alpha=0.30, save=False, save_name=None, show_plots=True):
    """
    Plot detector images like the quick-look detector-image panel.

    One L1B file represents one scan and each integration gets a thin axis.
    If ``cr_df`` and ``cr_column`` are supplied, rows whose flag equals one are
    highlighted. The DataFrame must also contain orbit, scan, and integration.
    """

    l1b_files = get_l1b_files(orbit, channel=channel)

    n_scans = len(l1b_files)

    # Build a set of all (scan, integration) pairs marked as CR
    cr_integrations = set()

    if cr_df is not None and cr_column is not None:
        required_columns = {
            "orbit",
            "scan",
            "integration",
            cr_column
        }

        missing_columns = required_columns - set(cr_df.columns)

        if missing_columns:
            raise KeyError(
                f"Missing columns in CR dataframe: {missing_columns}"
            )

        orbit_cr_df = cr_df.loc[
            (cr_df["orbit"] == orbit) &
            (cr_df[cr_column] == 1)
        ]

        cr_integrations = set(
            zip(
                orbit_cr_df["scan"].astype(int),
                orbit_cr_df["integration"].astype(int)
            )
        )

    # Open first scan to get dimensions
    with fits.open(l1b_files[0], ignore_missing_simple=True, ignore_missing_end=True) as hdul:
        test_spectra = hdul["detector_dark_subtracted"].data
        n_int, n_spa, n_spec = test_spectra.shape

    det_rows = int(np.ceil(n_scans / 6))

    fig_width = 20.25
    det_offset = (0.125 * n_int + 0.5) * det_rows + 0.25
    fig_height = det_offset + 0.75

    plt.style.use("dark_background")

    fig = plt.figure(
        figsize=(fig_width, fig_height),
        facecolor="black"    # optional but makes it explicit
    )

    peaks = [193, 199.5, 206, 216, 228, 240, 289]

    detector_image_axes = []

    for row in range(det_rows):
        for col in range(6):
            for inte in range(n_int):
                x0 = 0.5 + ((fig_width - 0.5) / 6) * col
                y0 = 0.5 * (row + 1) + 0.125 * n_int * row + 0.125 * inte
                xw = 19.25 / 6 - 0.5
                yw = 0.1

                ax = plt.axes([
                    x0 / fig_width,
                    y0 / fig_height,
                    xw / fig_width,
                    yw / fig_height
                ])

                ax.tick_params(
                    which="minor",
                    axis="x",
                    bottom=True,
                    top=True,
                    labeltop=False,
                    labelbottom=False
                )

                if inte == 0:
                    ax.set_xticks(peaks)
                    ax.tick_params(
                        which="major",
                        axis="x",
                        bottom=True,
                        top=False,
                        labeltop=False,
                        labelbottom=False
                    )
                elif inte == n_int - 1:
                    ax.set_xticks(peaks)
                    ax.tick_params(
                        which="major",
                        axis="x",
                        bottom=False,
                        top=True,
                        labeltop=False,
                        labelbottom=False
                    )
                else:
                    ax.set_xticks([])
                    ax.set_xticks(peaks, minor=True)

                ax.set_yticks([])
                detector_image_axes.append(ax)

    detector_image_axes = np.reshape(
        detector_image_axes,
        (int(np.ceil(n_scans / 6) * 6), n_int)
    )

    for scan, l1b_file in enumerate(l1b_files):
        with fits.open(l1b_file, ignore_missing_simple=True, ignore_missing_end=True) as hdul:
            wavelength = np.squeeze(hdul["observation"].data["wavelength"])
            spectra = hdul["detector_dark_subtracted"].data

        place_detector_images(
            detector_image_axes[scan],
            wavelength,
            spectra,
            n_spa
        )

        # Label each integration on the right side
        for integration in range(n_int):
            detector_image_axes[scan, integration].text(
                1.02,
                0.5,
                f"{integration + 1}",
                ha="left",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
                transform=detector_image_axes[scan, integration].transAxes,
                clip_on=False
            )

            # Add a transparent blue overlay when this integration
            # is identified as containing a cosmic ray
            if (scan, integration) in cr_integrations:
                overlay = Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=detector_image_axes[
                        scan, integration
                    ].transAxes,
                    facecolor="dodgerblue",
                    edgecolor="deepskyblue",
                    linewidth=1.5,
                    alpha=overlay_alpha,
                    zorder=10,
                    clip_on=False
                )

                detector_image_axes[
                    scan, integration
                ].add_patch(overlay)


        detector_image_axes[scan, n_int - 1].text(
            0.5,
            1.6,
            f"Scan {scan + 1}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="white",
            transform=detector_image_axes[scan, n_int - 1].transAxes,
            clip_on=False
        )

    for ax in detector_image_axes[n_scans:].ravel():
        ax.axis("off")

    if plot_title is None:
        plot_title = (
            f"Orbit {orbit:05d} L1B "
            f"{channel.upper()} detector images"
        )

    fig.suptitle(plot_title, fontsize=16)

    if save==True:
        plt.savefig(f'{save_name}.pdf')

    if show_plots:
        plt.show()


def main():
    """Run the editable example used when this file is executed as a script."""
    plot_detector(orbit=8693,
                  cr_df=None,
                  cr_column=None,
                  plot_title=None,
                  overlay_alpha=0.30,
                  save=False,
                  save_name=None,
                  show_plots=True)

if __name__ == "__main__":
    main()
