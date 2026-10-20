"""Reconstruct MLR spectral components from MAVEN/IUVS L1B and L1C data."""

import numpy as np
from astropy.io import fits
from pathlib import Path
from .misc_helpers import *
from .PyUVS.data import calculate_calibration_curve

def get_l1c_baseline(l1c_file, scan, int):
    """Return one baseline and the orbit-wide mean-plus-sigma threshold."""
    hdul=l1c_file
    orbit_const = hdul['radiance_noaltbin'].data['baseline_muv']
    const = hdul['radiance_noaltbin'].data['baseline_muv'][scan-1][int-1]

    threshold_const = np.nanmean(orbit_const) + np.nanstd(orbit_const)

    return const, threshold_const

def get_coeff(MLR, template_DN, dlambda, cal):
    """Convert an integrated MLR radiance to a detector-space coefficient."""
    return MLR / (np.sum(template_DN / cal) * dlambda)

def get_feature_indices(l1c_hdul):
    """Locate the required emission-component indices in an L1C product."""
    features = l1c_hdul['emission_features'].data['emission']
    solar_index = np.where(features == 'Solar cont. muv')[0][0]
    co_index = np.where(features == 'CO Cameron')[0][0]
    co2puvd_index = np.where(features == 'CO2pUVD')[0][0]
    o2972_index = np.where(features == 'O2972')[0][0]
    no_nightglow_index = np.where(features == 'NO band')[0][0]
    o1304_index = np.where(features == 'OI 130.4')[0][0]
    o1356_index = np.where(features == 'OI 135.6')[0][0]

    return solar_index, co_index, co2puvd_index, o2972_index, no_nightglow_index, o1304_index, o1356_index

def get_radiances(hdull1c, scan, solar_ind, co_ind, co2puvd_ind, o2972_ind, o1304_ind, o1356_ind, no_nightglow_ind):
    """Extract component radiance cubes for a zero-based scan index."""
    # get the radiances
    solar_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, solar_ind]
    co_cameron_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, co_ind]
    co2p_uvd_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, co2puvd_ind]
    o2972_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, o2972_ind]
    o1304_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, o1304_ind]
    o1356_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, o1356_ind]
    no_nightglow_radiance = hdull1c['radiance_noaltbin'].data['profile'][scan, :, :, no_nightglow_ind]

    return solar_radiance, co_cameron_radiance, co2p_uvd_radiance, o2972_radiance, o1304_radiance, o1356_radiance, no_nightglow_radiance

def get_all_coeffs(solar_radiance, co_cameron_radiance, co2p_uvd_radiance, no_nightglow_radiance,
                   template_solar_continuum, template_co_cameron, template_co2p_uvd, template_no_nightglow,
                   dwavelength, calibration_curve, good_int, good_spa):
    """Calculate detector-space coefficients for every fitted component."""
    
    # get DN coefficients
    solar_coeff = get_coeff(solar_radiance[good_int, good_spa], template_solar_continuum, dwavelength,
                            calibration_curve[3, :])
    co_coeff = get_coeff(co_cameron_radiance[good_int, good_spa], template_co_cameron, dwavelength,
                         calibration_curve[3, :])
    co2p_coeff = get_coeff(co2p_uvd_radiance[good_int, good_spa], template_co2p_uvd, dwavelength,
                           calibration_curve[3, :])
    no_coeff = get_coeff(no_nightglow_radiance[good_int, good_spa], template_no_nightglow, dwavelength,
                         calibration_curve[3, :])

    return solar_coeff, co_coeff, co2p_coeff, no_coeff

def get_templates(n_spec):
    """Load spectral templates and adapt them to an observed bin count.

    The 219-bin mode is a slice of the 256-bin template. The returned tuple
    contains wavelength, spacing, four component templates, and the empirical
    CO Cameron scale factor.
    """
    if n_spec == 219:
        load_spec = 256
    else:
        load_spec = n_spec

    template_path = (
        Path(__file__).resolve().parent
        / "PyUVS"
        / "ancillary"
        / f"templates_{load_spec}.npy"
    )

    templates = np.load(template_path, allow_pickle=True)
    template_wavelength = templates.item().get('wavelength')
    dwavelength = np.diff(template_wavelength)[0]
    template_solar_continuum = templates.item().get('solar_continuum')
    template_co_cameron = templates.item().get('co_cameron')
    template_co2p_uvd = templates.item().get('co2p_uvd')
    template_no_nightglow = templates.item().get('no_nightglow')

    # account for 219 spectral bins
    if n_spec == 219:
        start = 6
        template_wavelength = template_wavelength[start:start + n_spec]
        template_solar_continuum = template_solar_continuum[start:start + n_spec]
        template_co_cameron = template_co_cameron[start:start + n_spec]
        template_co2p_uvd = template_co2p_uvd[start:start + n_spec]
        template_no_nightglow = template_no_nightglow[start:start + n_spec]

    # scaling factors
    co_cameron_scale = 1.0
    if n_spec == 200:
        co_cameron_scale = 1.2010
    if n_spec == 219:
        co_cameron_scale = 1.0029

    return template_wavelength, dwavelength, template_solar_continuum, template_co_cameron, template_co2p_uvd, template_no_nightglow, co_cameron_scale


def build_spectrum(orbit_num, scan, integ):
    """Build measured and fitted spectra for one UI-selected integration.

    Parameters are an orbit number plus one-based scan and integration values.
    The result is a dictionary of equally sized arrays ready for plotting,
    along with scalar baseline diagnostics.
    """
    # Treat scan and integration as human-readable, starting at 1
    scan_index = scan - 1
    integration_index = integ - 1

    l1c_path = open_l1c(orbit_num)
    l1b_paths = open_l1b(orbit_num)

    with fits.open(l1c_path) as hdull1c, fits.open(l1b_paths[scan_index]) as hdull1b:
        n_scans, n_int, n_spa, n_spec = get_dims(hdull1c)

        if not 1 <= scan <= n_scans:
            raise ValueError(
                f"Scan must be between 1 and {n_scans}; received {scan}."
            )

        if not 1 <= integ <= n_int:
            raise ValueError(
                f"Integration must be between 1 and {n_int}; received {integ}."
            )

        # Select the requested integration for every spatial bin
        good_int = np.full(n_spa, integration_index, dtype=int)
        good_spa = np.arange(n_spa)

        # Calculate these for the selected L1B file instead of relying on globals
        calibration_curve = calculate_calibration_curve(hdull1b)
        wavelength = np.squeeze(
            hdull1b["observation"].data["wavelength"]
        )
        spectra = hdull1b["detector_dark_subtracted"].data

        const, threshold_const = get_l1c_baseline(
            hdull1c,
            scan,
            integ
        )

        (
            solar_ind,
            co_ind,
            co2puvd_ind,
            o2972_ind,
            no_nightglow_ind,
            o1304_ind,
            o1356_ind,
        ) = get_feature_indices(hdull1c)

        (
            template_wavelength,
            dwavelength,
            template_solar_continuum,
            template_co_cameron,
            template_co2p_uvd,
            template_no_nightglow,
            co_cameron_scale,
        ) = get_templates(n_spec)

        (
            solar_radiance,
            co_cameron_radiance,
            co2p_uvd_radiance,
            o2972_radiance,
            o1304_radiance,
            o1356_radiance,
            no_nightglow_radiance,
        ) = get_radiances(
            hdull1c,
            scan_index,
            solar_ind,
            co_ind,
            co2puvd_ind,
            o2972_ind,
            o1304_ind,
            o1356_ind,
            no_nightglow_ind,
        )

        (
            solar_coeff,
            co_coeff,
            co2p_coeff,
            no_coeff,
        ) = get_all_coeffs(
            solar_radiance,
            co_cameron_radiance,
            co2p_uvd_radiance,
            no_nightglow_radiance,
            template_solar_continuum,
            template_co_cameron,
            template_co2p_uvd,
            template_no_nightglow,
            dwavelength,
            calibration_curve,
            good_int,
            good_spa,
        )

        # Average the seven spatial bins for the measured spectrum
        measured_spectrum = np.nanmean(
            spectra[good_int, good_spa],
            axis=0
        )

        # Fitted components
        solar_component = (
            template_solar_continuum * np.nanmean(solar_coeff)
        )

        co_component = (
            template_co_cameron
            * np.nanmean(co_coeff)
            * co_cameron_scale
        )

        co2p_component = (
            template_co2p_uvd * np.nanmean(co2p_coeff)
        )

        no_component = (
            template_no_nightglow * np.nanmean(no_coeff)
        )

        constant_value = np.nanmean(const)

        # Convert scalar constant into an array for plotting
        constant_component = np.full(
            n_spec,
            constant_value,
            dtype=float
        )

        aurora_spectrum = co_component + co2p_component

        total_fit = (
            solar_component
            + co_component
            + co2p_component
            + no_component
            + constant_component
        )

        # Background after removing only the background-model terms
        background_spectrum = (
            measured_spectrum
            - solar_component
            - no_component
            - constant_component
        )

        plot_wavelength = np.nanmean(
            wavelength[good_spa],
            axis=0
        )

    return {
        "wavelength": plot_wavelength,
        "measured": measured_spectrum,
        "aurora": aurora_spectrum,
        "solar": solar_component,
        "co": co_component,
        "co2p": co2p_component,
        "no": no_component,
        "constant": constant_component,
        "total_fit": total_fit,
        "background": background_spectrum,
        "threshold_constant": threshold_const,
        "constant_value": constant_value,
    }
