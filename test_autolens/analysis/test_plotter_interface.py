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

    plotter_interface = vis.PlotterInterface(output_path=plot_path)

    plotter_interface.tracer(
        tracer=tracer_x2_plane_7x7, grid=masked_imaging_7x7.grid, during_analysis=False
    )

    plot_path = path.join(plot_path, "tracer")

    assert path.join(plot_path, "subplot_galaxies_images.png") in plot_patch.paths
    assert path.join(plot_path, "image_2d.png") in plot_patch.paths
    assert path.join(plot_path, "plane_image_of_plane_1.png") in plot_patch.paths
    assert path.join(plot_path, "convergence_2d.png") in plot_patch.paths
    assert path.join(plot_path, "potential_2d.png") not in plot_patch.paths
    assert path.join(plot_path, "deflections_y_2d.png") not in plot_patch.paths
    assert path.join(plot_path, "deflections_x_2d.png") not in plot_patch.paths
    assert path.join(plot_path, "magnification_2d.png") in plot_patch.paths

    convergence = al.util.array_2d.numpy_array_2d_via_fits_from(
        file_path=path.join(plot_path, "fits", "convergence_2d.fits"), hdu=0
    )

    assert convergence.shape == (7, 7)


def test__image_with_positions(
    image_7x7, positions_x2, include_2d_all, plot_path, plot_patch
):
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)

    plotter_interface = vis.PlotterInterface(output_path=plot_path)

    plotter_interface.image_with_positions(image=image_7x7, positions=positions_x2)

    plot_path = path.join(plot_path, "positions")

    assert path.join(plot_path, "image_with_positions.png") in plot_patch.paths
