import os

import autofit as af
import autolens as al

test_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))


def pixel_scale_from_data_resolution(data_resolution):
    """Determine the pixel scale from a data_type resolution type based on real observations.

    These options are representative of LSST, Euclid, HST, over-sampled HST and Adaptive Optics image.

    Parameters
    ----------
    data_resolution : str
        A string giving the resolution of the desired data_type type (LSST | Euclid | HST | HST_Up | AO).
    """
    if data_resolution == "sma":
        return (0.05, 0.05)
    else:
        raise ValueError(
            "An invalid data_type resolution was entered - ", data_resolution
        )


def load_test_interferometer(data_type, data_resolution):

    dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=test_path,
        folder_names=["dataset", "interferometer", data_type, data_resolution],
    )

    return al.Interferometer.from_fits(
        visibilities_path=dataset_path + "/visibilities.fits",
        noise_map_path=dataset_path + "/noise_map.fits",
        uv_wavelengths_path=dataset_path + "/uv_wavelengths.fits",
    )
