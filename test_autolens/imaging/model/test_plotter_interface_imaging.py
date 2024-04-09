import os
import shutil
from os import path

import pytest
from autoconf import conf
import autolens as al
from autolens.imaging.model.plotter_interface import PlotterInterfaceImaging

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plotter_interface_plotter_setup():
    return path.join("{}".format(directory), "files")


def test__visualizes_fit_imaging__uses_configs(
    fit_imaging_x2_plane_inversion_7x7, include_2d_all, plot_path, plot_patch
):
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)

    plotter_interface = PlotterInterfaceImaging(output_path=plot_path)

    plotter_interface.visualize_fit_imaging(
        fit=fit_imaging_x2_plane_inversion_7x7, during_analysis=False
    )

    assert path.join(plot_path, "subplot_tracer.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_fit_log10.png") in plot_patch.paths

    plot_path = path.join(plot_path, "fit_dataset")

    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths

    assert path.join(plot_path, "lens_subtracted_image.png") in plot_patch.paths
    assert path.join(plot_path, "source_model_image.png") not in plot_patch.paths

    assert path.join(plot_path, "reconstruction.png") in plot_patch.paths

    image = al.util.array_2d.numpy_array_2d_via_fits_from(
        file_path=path.join(plot_path, "fits", "data.fits"), hdu=0
    )

    assert image.shape == (7, 7)
