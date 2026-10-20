"""Shared discovery, FITS-dimension, and fit-quality helpers."""

import glob
import re
import numpy as np

from collections import defaultdict
from scipy.stats import pearsonr
import sklearn.metrics as metrics

def open_l1c(orbit_num):
    """Return the first L1C periapse product path for ``orbit_num``.

    Products are searched below the hard-coded local L1C root in a directory
    named for the orbit rounded down to the nearest hundred.
    """
    orbit_num0=orbit_num
    orbit_num=(orbit_num0//100)*100 #rounds down to nearest 100

    path=f"E:/nightglow/c/{orbit_num:05d}"
    file_list=glob.glob(path+'/mvn_iuv_l1c_periapse-orbit**.fits.gz')

    orbit_groups=defaultdict(list)

    for file in file_list:
        match = re.search(r'orbit(\d+)', file)
        if match:
            orbit=int(match.group(1))
            orbit_groups[orbit].append(file)

    return orbit_groups[orbit_num0][0]

def open_l1b(orbit_num):
    """Return all L1B periapse product paths associated with an orbit."""
    orbit_num0=orbit_num
    orbit_num=(orbit_num0//100)*100 #rounds down to nearest 100

    path=f"E:/nightglow/b/{orbit_num:05d}"
    file_list=glob.glob(path+'/mvn_iuv_l1b_periapse-orbit**.fits.gz')

    orbit_groups=defaultdict(list)

    for file in file_list:
        match = re.search(r'orbit(\d+)', file)
        if match:
            orbit=int(match.group(1))
            orbit_groups[orbit].append(file)

    return orbit_groups[orbit_num0]

def get_dims(l1c_hdul):
    """Return scan, integration, spatial, and spectral sizes from an L1C HDU."""
    dimensions = l1c_hdul['spectrum_noaltbin'].data['muv'].shape
    scans_count = dimensions[0]
    integration_count = dimensions[1]
    spatial_bin_count = dimensions[2]
    spectral_bin_count = dimensions[3]
    return scans_count, integration_count, spatial_bin_count, spectral_bin_count

def calculate_correlation(plot_wavelength, background_spectrum, aurora_spectrum):
    """Measure agreement in the 190--246 nm Cameron-band region.

    Returns ``(r_squared, pearson_r)``. Both values are NaN when the input is
    too short or otherwise unsuitable for the underlying metrics.
    """

    # Calculate fit quality using only the bright Cameron-band features.
    minwave_ind = np.argmin(np.abs(plot_wavelength - 190))
    maxwave_ind = np.argmin(np.abs(plot_wavelength - 246))
    y_pred = aurora_spectrum[minwave_ind:maxwave_ind]
    y_true = background_spectrum[minwave_ind:maxwave_ind]
    try:
        r2 = metrics.r2_score(y_true, y_pred)
        r = pearsonr(y_true, y_pred)[0]
    except ValueError:
        r2 = np.nan
        r = np.nan

    # return the goodness-of-fit metrics
    return r2, r
