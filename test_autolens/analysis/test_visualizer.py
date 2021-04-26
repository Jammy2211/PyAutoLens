import os
import shutil
from os import path

import pytest
from autoconf import conf
import autolens as al
from autolens.analysis import visualizer as vis

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_visualizer_plotter_setup():
    return path.join("{}".format(directory), "files")


@pytest.fixture(autouse=True)
def push_config(plot_path):
    conf.instance.push(path.join(directory, "config"), output_path=plot_path)


class TestVisualizer:
    def test__visualizes_ray_tracing__uses_configs(
        self,
        masked_imaging_7x7,
        tracer_x2_plane_7x7,
        include_2d_all,
        plot_path,
        plot_patch,
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.Visualizer(visualize_path=plot_path)

        visualizer.visualize_tracer(
            tracer=tracer_x2_plane_7x7,
            grid=masked_imaging_7x7.grid,
            during_analysis=False,
        )

        plot_path = path.join(plot_path, "ray_tracing")

        assert path.join(plot_path, "subplot_tracer.png") in plot_patch.paths
        assert path.join(plot_path, "image_2d.png") in plot_patch.paths
        assert path.join(plot_path, "plane_image_of_plane_1.png") in plot_patch.paths
        assert path.join(plot_path, "convergence_2d.png") in plot_patch.paths
        assert path.join(plot_path, "potential_2d.png") not in plot_patch.paths
        assert path.join(plot_path, "deflections_y_2d.png") not in plot_patch.paths
        assert path.join(plot_path, "deflections_x_2d.png") not in plot_patch.paths
        assert path.join(plot_path, "magnification_2d.png") in plot_patch.paths

        convergence = al.util.array_2d.numpy_array_2d_from_fits(
            file_path=path.join(plot_path, "fits", "convergence_2d.fits"), hdu=0
        )

        assert convergence.shape == (5, 5)

    def test__visualize_stochastic_histogram(
        self, masked_imaging_7x7, plot_path, plot_patch
    ):

        visualizer = vis.Visualizer(visualize_path=plot_path)

        visualizer.visualize_stochastic_histogram(
            log_evidences=[1.0, 2.0, 1.0, 2.0, 3.0, 2.5], max_log_evidence=3.0
        )
        assert (
            path.join(plot_path, "other", "stochastic_histogram.png")
            in plot_patch.paths
        )

    def test__visualizes_fit_imaging__uses_configs(
        self, fit_imaging_x2_plane_inversion_7x7, include_2d_all, plot_path, plot_patch
    ):

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        visualizer = vis.Visualizer(visualize_path=plot_path)

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

        image = al.util.array_2d.numpy_array_2d_from_fits(
            file_path=path.join(plot_path, "fits", "image_2d.fits"), hdu=0
        )

        assert image.shape == (5, 5)

    def test__visualize_fit_interferometer__uses_configs(
        self,
        fit_interferometer_x2_plane_inversion_7x7,
        include_2d_all,
        plot_path,
        plot_patch,
    ):
        visualizer = vis.Visualizer(visualize_path=plot_path)

        visualizer.visualize_fit_interferometer(
            fit=fit_interferometer_x2_plane_inversion_7x7, during_analysis=True
        )

        plot_path = path.join(plot_path, "fit_interferometer")

        assert (
            path.join(plot_path, "subplot_fit_interferometer.png") in plot_patch.paths
        )
        assert path.join(plot_path, "subplot_fit_real_space.png") in plot_patch.paths
        assert path.join(plot_path, "subplot_fit_dirty_images.png") in plot_patch.paths

        assert path.join(plot_path, "visibilities.png") in plot_patch.paths
        assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
        assert path.join(plot_path, "model_visibilities.png") in plot_patch.paths
        assert (
            path.join(plot_path, "real_residual_map_vs_uv_distances.png")
            not in plot_patch.paths
        )
        assert (
            path.join(plot_path, "real_normalized_residual_map_vs_uv_distances.png")
            in plot_patch.paths
        )
        assert (
            path.join(plot_path, "real_chi_squared_map_vs_uv_distances.png")
            in plot_patch.paths
        )

        print(plot_patch.paths)

        assert path.join(plot_path, "dirty_image_2d.png") in plot_patch.paths
        assert path.join(plot_path, "dirty_noise_map_2d.png") not in plot_patch.paths
        assert (
            path.join(plot_path, "dirty_signal_to_noise_map_2d.png")
            not in plot_patch.paths
        )
        assert path.join(plot_path, "dirty_image_2d.png") in plot_patch.paths
        assert (
            path.join(plot_path, "dirty_real_residual_map_2d.png")
            not in plot_patch.paths
        )
        assert (
            path.join(plot_path, "dirty_normalized_residual_map_2d.png")
            in plot_patch.paths
        )
        assert path.join(plot_path, "dirty_chi_squared_map_2d.png") in plot_patch.paths
