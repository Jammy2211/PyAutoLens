from os import path
import pytest

from autolens.interferometer.plot.fit_interferometer_plots import (
    subplot_fit,
    subplot_fit_real_space,
)


@pytest.fixture(name="plot_path")
def make_fit_interferometer_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "fit"
    )


def test__subplot_fit(
    fit_interferometer_x2_plane_7x7, plot_path, plot_patch
):
    subplot_fit(
        fit=fit_interferometer_x2_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths


def test__subplot_fit_real_space(
    fit_interferometer_x2_plane_7x7,
    fit_interferometer_x2_plane_inversion_7x7,
    plot_path,
    plot_patch,
):
    subplot_fit_real_space(
        fit=fit_interferometer_x2_plane_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert path.join(plot_path, "subplot_fit_real_space.png") in plot_patch.paths
