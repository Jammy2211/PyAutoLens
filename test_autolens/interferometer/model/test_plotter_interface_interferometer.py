from os import path

import pytest

import autolens as al

from autolens.interferometer.model.plotter_interface import (
    PlotterInterfaceInterferometer,
)

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plotter_interface_plotter_setup():
    return path.join("{}".format(directory), "files")


def test__fit_interferometer(
    fit_interferometer_x2_plane_7x7,
    include_2d_all,
    plot_path,
    plot_patch,
):
    plotter_interface = PlotterInterfaceInterferometer(image_path=plot_path)

    plotter_interface.fit_interferometer(
        fit=fit_interferometer_x2_plane_7x7,
    )

    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_fit_real_space.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_fit_dirty_images.png") in plot_patch.paths

    # visibilities = ag.util.array_2d.numpy_array_2d_via_fits_from(
    #     file_path=path.join(plot_path, "fit.fits"), hdu=0
    # )
    #
    # assert visibilities.shape == (5, 5)

    image = al.util.array_2d.numpy_array_2d_via_fits_from(
        file_path=path.join(plot_path, "model_galaxy_images.fits"), hdu=0
    )

    assert image.shape == (7, 7)

    image = al.util.array_2d.numpy_array_2d_via_fits_from(
        file_path=path.join(plot_path, "dirty_images.fits"), hdu=0
    )

    assert image.shape == (7, 7)