import os
import shutil
from os import path

import pytest
from autoconf import conf
import autolens as al
from autolens.imaging.model.visualizer import VisualizerImaging

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_visualizer_plotter_setup():
    return path.join("{}".format(directory), "files")


@pytest.fixture(autouse=True)
def push_config(plot_path):
    conf.instance.push(path.join(directory, "config"), output_path=plot_path)


class TestVisualizerImaging:
    def test__visualizes_fit_imaging__uses_configs(
        self, fit_imaging_x2_plane_inversion_7x7, include_2d_all, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = VisualizerImaging(visualize_path=plot_path)

        visualizer.visualize_fit_imaging(
            fit=fit_imaging_x2_plane_inversion_7x7, during_analysis=False
        )

        plot_path = path.join(plot_path, "fit_imaging")

        assert path.join(plot_path, "subplot_fit_imaging.png") in plot_patch.paths
        assert path.join(plot_path, "image_2d.png") in plot_patch.paths
        assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "model_image.png") in plot_patch.paths
        assert path.join(plot_path, "residual_map.png") not in plot_patch.paths
        assert path.join(plot_path, "normalized_residual_map.png") in plot_patch.paths
        assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths
        assert (
            path.join(plot_path, "subtracted_image_of_plane_0.png") in plot_patch.paths
        )
        assert (
            path.join(plot_path, "subtracted_image_of_plane_1.png") in plot_patch.paths
        )
        assert (
            path.join(plot_path, "model_image_of_plane_0.png") not in plot_patch.paths
        )
        assert (
            path.join(plot_path, "model_image_of_plane_1.png") not in plot_patch.paths
        )
        assert path.join(plot_path, "plane_image_of_plane_0.png") in plot_patch.paths
        assert path.join(plot_path, "reconstruction.png") in plot_patch.paths

        image = al.util.array_2d.numpy_array_2d_via_fits_from(
            file_path=path.join(plot_path, "fits", "image_2d.fits"), hdu=0
        )

        assert image.shape == (5, 5)
