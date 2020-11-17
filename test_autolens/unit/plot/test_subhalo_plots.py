from os import path

import pytest

import autolens as al
import autolens.plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_fit_imaging_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "fit"
    )


def test__subhalo_detection_sub_plot(
    masked_imaging_fit_x2_plane_7x7,
    masked_imaging_fit_x2_plane_inversion_7x7,
    include_all,
    plot_path,
    plot_patch,
):
    arr = al.Array.manual_2d(array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)

    aplt.Subhalo.subplot_detection_imaging(
        fit_imaging_detect=masked_imaging_fit_x2_plane_7x7,
        detection_array=arr,
        mass_array=arr,
        include=include_all,
        sub_plotter=aplt.SubPlotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "subplot_detection_imaging.png") in plot_patch.paths

    aplt.Subhalo.subplot_detection_imaging(
        fit_imaging_detect=masked_imaging_fit_x2_plane_inversion_7x7,
        detection_array=arr,
        mass_array=arr,
        include=include_all,
        sub_plotter=aplt.SubPlotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "subplot_detection_imaging.png") in plot_patch.paths


def test__subhalo_detection_fits(
    masked_imaging_fit_x2_plane_7x7,
    masked_imaging_fit_x2_plane_inversion_7x7,
    include_all,
    plot_path,
    plot_patch,
):
    aplt.Subhalo.subplot_detection_fits(
        fit_imaging_before=masked_imaging_fit_x2_plane_7x7,
        fit_imaging_detect=masked_imaging_fit_x2_plane_7x7,
        include=include_all,
        sub_plotter=aplt.SubPlotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "subplot_detection_fits.png") in plot_patch.paths

    aplt.Subhalo.subplot_detection_fits(
        fit_imaging_before=masked_imaging_fit_x2_plane_inversion_7x7,
        fit_imaging_detect=masked_imaging_fit_x2_plane_inversion_7x7,
        include=include_all,
        sub_plotter=aplt.SubPlotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "subplot_detection_fits.png") in plot_patch.paths
