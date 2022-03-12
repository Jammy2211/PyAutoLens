import os
import shutil
from os import path

import pytest
from autoconf import conf
import autolens as al
from autolens.lens.model import visualizer as vis

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_visualizer_plotter_setup():
    return path.join("{}".format(directory), "files")


@pytest.fixture(autouse=True)
def push_config(plot_path):
    conf.instance.push(path.join(directory, "config"), output_path=plot_path)


def test__visualizes_ray_tracing__uses_configs(
    masked_imaging_7x7, tracer_x2_plane_7x7, include_2d_all, plot_path, plot_patch
):

    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)

    visualizer = vis.Visualizer(visualize_path=plot_path)

    visualizer.visualize_tracer(
        tracer=tracer_x2_plane_7x7, grid=masked_imaging_7x7.grid, during_analysis=False
    )

    plot_path = path.join(plot_path, "ray_tracing")

    assert path.join(plot_path, "subplot_tracer.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_plane_images.png") in plot_patch.paths
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

    assert convergence.shape == (5, 5)


def test__visualize_stochastic_histogram(masked_imaging_7x7, plot_path, plot_patch):

    visualizer = vis.Visualizer(visualize_path=plot_path)

    visualizer.visualize_stochastic_histogram(
        stochastic_log_likelihoods=[1.0, 2.0, 1.0, 2.0, 3.0, 2.5], max_log_evidence=3.0
    )
    assert path.join(plot_path, "other", "stochastic_histogram.png") in plot_patch.paths
