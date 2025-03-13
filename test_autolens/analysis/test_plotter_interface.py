import os
import shutil
from os import path

import pytest
import autolens as al
from autolens.analysis import plotter_interface as vis

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plotter_interface_plotter_setup():
    return path.join("{}".format(directory), "files")


def test__tracer(
    masked_imaging_7x7, tracer_x2_plane_7x7, include_2d_all, plot_path, plot_patch
):
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)

    plotter_interface = vis.PlotterInterface(image_path=plot_path)

    plotter_interface.tracer(
        tracer=tracer_x2_plane_7x7,
        grid=masked_imaging_7x7.grids.lp,
    )

    assert path.join(plot_path, "subplot_galaxies_images.png") in plot_patch.paths

    image = al.util.array_2d.numpy_array_2d_via_fits_from(
        file_path=path.join(plot_path, "tracer.fits"), hdu=0
    )

    assert image.shape == (5, 5)


def test__image_with_positions(
    image_7x7, positions_x2, include_2d_all, plot_path, plot_patch
):
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)

    plotter_interface = vis.PlotterInterface(image_path=plot_path)

    plotter_interface.image_with_positions(image=image_7x7, positions=positions_x2)

    assert path.join(plot_path, "image_with_positions.png") in plot_patch.paths
