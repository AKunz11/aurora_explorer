"""Prepare a single L1B detector integration for the Dash heatmap."""

from astropy.io import fits
from .misc_helpers import *

def build_detector_image(orbit_num, scan, integ):
    """Load one detector image and return wavelength, DN, and log10(DN).

    ``scan`` and ``integ`` are one-based values matching the user interface.
    DN values are clipped to 50 before taking the logarithm, matching the
    detector plot's lower color limit.
    """
    l1b_paths = open_l1b(orbit_num)
    with fits.open(l1b_paths[scan - 1]) as hdull1b:
        wavelength = np.squeeze(hdull1b["observation"].data["wavelength"])
        spectra = hdull1b["detector_dark_subtracted"].data[integ - 1, :, :]
        log_spectra = np.log10(np.clip(spectra, 50, None))
        #detector_image = np.nanmean(spectra, axis=0)
    return wavelength, spectra, log_spectra
