from pathlib import Path

import pytest

import autolens as al

from autolens.interferometer.model.plotter import (
    PlotterInterferometer,
)

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_plotter_plotter_setup():
    return directory / "files"


def test__fit_interferometer__quick_update__writes_normal_fit_subplot_only(
    fit_interferometer_x2_plane_7x7,
    plot_path,
    plot_patch,
):
    plotter = PlotterInterferometer(image_path=plot_path)

    plotter.fit_interferometer(
        fit=fit_interferometer_x2_plane_7x7, quick_update=True
    )

    assert str(plot_path / "fit.png") in plot_patch.paths
    assert str(plot_path / "fit_quick.png") not in plot_patch.paths
    assert str(plot_path / "fit_dirty_images.png") not in plot_patch.paths


def test__fit_interferometer(
    fit_interferometer_x2_plane_7x7,
    plot_path,
    plot_patch,
):
    plotter = PlotterInterferometer(image_path=plot_path)

    plotter.fit_interferometer(
        fit=fit_interferometer_x2_plane_7x7,
    )

    assert str(plot_path / "fit.png") in plot_patch.paths
    assert str(plot_path / "fit_real_space.png") in plot_patch.paths
    assert str(plot_path / "fit_dirty_images.png") in plot_patch.paths

    image = al.ndarray_via_fits_from(
        file_path=plot_path / "galaxy_images.fits", hdu=0
    )

    assert image.shape == (5, 5)

    image = al.ndarray_via_fits_from(
        file_path=plot_path / "dirty_images.fits", hdu=0
    )

    assert image.shape == (5, 5)
