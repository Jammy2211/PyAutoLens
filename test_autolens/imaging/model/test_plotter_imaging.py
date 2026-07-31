import shutil
from pathlib import Path

import pytest
import autolens as al
from autolens.imaging.model.plotter import PlotterImaging

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_plotter_plotter_setup():
    return directory / "files"


def test__fit_imaging(
    fit_imaging_x2_plane_inversion_7x7, plot_path, plot_patch
):
    if plot_path.exists():
        shutil.rmtree(plot_path)

    plotter = PlotterImaging(image_path=plot_path)

    plotter.fit_imaging(
        fit=fit_imaging_x2_plane_inversion_7x7,
    )

    assert str(plot_path / "tracer.png") in plot_patch.paths
    assert str(plot_path / "fit.png") in plot_patch.paths
    assert str(plot_path / "fit_log10.png") in plot_patch.paths

    image = al.ndarray_via_fits_from(
        file_path=plot_path / "fit.fits", hdu=0
    )

    assert image.shape == (5, 5)

    image = al.ndarray_via_fits_from(
        file_path=plot_path / "model_galaxy_images.fits", hdu=0
    )

    assert image.shape == (5, 5)

def test__fit_imaging__quick_update__writes_normal_fit_subplot_only(
    fit_imaging_x2_plane_inversion_7x7, plot_path, plot_patch
):
    if plot_path.exists():
        shutil.rmtree(plot_path)

    plotter = PlotterImaging(image_path=plot_path)

    plotter.fit_imaging(fit=fit_imaging_x2_plane_inversion_7x7, quick_update=True)

    assert str(plot_path / "fit.png") in plot_patch.paths
    assert str(plot_path / "fit_quick.png") not in plot_patch.paths
    assert str(plot_path / "tracer.png") not in plot_patch.paths
    assert str(plot_path / "fit_log10.png") not in plot_patch.paths


def test__fit_imaging_combined(
    fit_imaging_x2_plane_inversion_7x7, plot_path, plot_patch
):
    if plot_path.exists():
        shutil.rmtree(plot_path)

    visualizer = PlotterImaging(image_path=plot_path)

    visualizer.fit_imaging_combined(fit_list=2 * [fit_imaging_x2_plane_inversion_7x7])

    assert str(plot_path / "fit_combined.png") in plot_patch.paths
