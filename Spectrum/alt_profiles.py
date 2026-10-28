from astropy.io import fits

from Spectrum.misc_helpers import open_l1c


def get_alt_profiles(orbit, scan):
    """Return Cameron-band and UVD altitude profiles for a one-based scan."""
    l1c_path = open_l1c(orbit)
    scan_index = scan - 1

    with fits.open(l1c_path, memmap=False) as l1c:
        geometry = l1c["geometry_altbin"].data
        radiance = l1c["radiance_altbin"].data

        n_scans = len(geometry["TANGENT_ALT"])
        if not 0 <= scan_index < n_scans:
            raise ValueError(
                f"Scan must be between 1 and {n_scans}; received {scan}."
            )

        # Copy the selected rows so they remain valid after closing the FITS file.
        alt = geometry["TANGENT_ALT"][scan_index].copy()
        brightness = radiance["PROFILE"][scan_index].copy()
        random_unc = radiance["RANDOM_UNC"][scan_index].copy()

    # Feature indices verified in this file's emission_features table
    cameron = brightness[:, 18]   # kR
    uvd = brightness[:, 21]       # kR

    return cameron, uvd, alt, random_unc
