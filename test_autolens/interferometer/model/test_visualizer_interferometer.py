from os import path

import pytest
from autoconf import conf
from autolens.interferometer.model.visualizer import VisualizerInterferometer

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_visualizer_plotter_setup():
    return path.join("{}".format(directory), "files")


@pytest.fixture(autouse=True)
def push_config(plot_path):
    conf.instance.push(path.join(directory, "config"), output_path=plot_path)


class TestVisualizer:
    def test__visualize_fit_interferometer__uses_configs(
        self,
        fit_interferometer_x2_plane_inversion_7x7,
        include_2d_all,
        plot_path,
        plot_patch,
    ):
        visualizer = VisualizerInterferometer(visualize_path=plot_path)

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
