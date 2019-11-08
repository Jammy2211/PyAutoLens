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
    if data_resolution == "lsst":
        return (0.2, 0.2)
    elif data_resolution == "euclid":
        return (0.1, 0.1)
    elif data_resolution == "hst":
        return (0.05, 0.05)
    elif data_resolution == "hst_up":
        return (0.03, 0.03)
    elif data_resolution == "ao":
        return (0.01, 0.01)
    else:
        raise ValueError(
            "An invalid data_type resolution was entered - ", data_resolution
        )


def load_test_imaging(
    data_type, data_resolution, psf_shape_2d=(11, 11), lens_name=None
):

    pixel_scales = pixel_scale_from_data_resolution(data_resolution=data_resolution)

    dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=test_path, folder_names=["dataset", "imaging", data_type, data_resolution]
    )

    return al.imaging.from_fits(
        image_path=dataset_path + "/image.fits",
        psf_path=dataset_path + "/psf.fits",
        noise_map_path=dataset_path + "/noise_map.fits",
        pixel_scales=pixel_scales,
        resized_psf_shape=psf_shape_2d,
        lens_name=lens_name,
    )
