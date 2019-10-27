import autolens as al
import pytest
import os


@pytest.fixture(name="lens_plotter_util_path")
def make_lens_plotter_util_path_setup():
    return "{}/../../test_files/plotting/lens_plotter_util/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


def test__subtracted_image_of_plane_is_output(
    lens_imaging_fit_x1_plane_7x7,
    lens_imaging_fit_x2_plane_7x7,
    lens_plotter_util_path,
    plot_patch,
):

    al.lens_plotter_util.plot_subtracted_image_of_plane(
        fit=lens_imaging_fit_x1_plane_7x7,
        plane_index=0,
        mask=lens_imaging_fit_x2_plane_7x7.mask,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=lens_plotter_util_path,
        output_format="png",
    )

    assert (
        lens_plotter_util_path + "fit_subtracted_image_of_plane_0.png"
        in plot_patch.paths
    )

    al.lens_plotter_util.plot_subtracted_image_of_plane(
        fit=lens_imaging_fit_x2_plane_7x7,
        plane_index=0,
        mask=lens_imaging_fit_x2_plane_7x7.mask,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=lens_plotter_util_path,
        output_format="png",
    )

    assert (
        lens_plotter_util_path + "fit_subtracted_image_of_plane_0.png"
        in plot_patch.paths
    )

    al.lens_plotter_util.plot_subtracted_image_of_plane(
        fit=lens_imaging_fit_x2_plane_7x7,
        plane_index=1,
        mask=lens_imaging_fit_x2_plane_7x7.mask,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=lens_plotter_util_path,
        output_format="png",
    )

    assert (
        lens_plotter_util_path + "fit_subtracted_image_of_plane_1.png"
        in plot_patch.paths
    )


def test__model_image_of_plane_is_output(
    lens_imaging_fit_x1_plane_7x7,
    lens_imaging_fit_x2_plane_7x7,
    lens_plotter_util_path,
    plot_patch,
):

    al.lens_plotter_util.plot_model_image_of_plane(
        fit=lens_imaging_fit_x1_plane_7x7,
        plane_index=0,
        mask=lens_imaging_fit_x2_plane_7x7.mask,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=lens_plotter_util_path,
        output_format="png",
    )

    assert lens_plotter_util_path + "fit_model_image_of_plane_0.png" in plot_patch.paths

    al.lens_plotter_util.plot_model_image_of_plane(
        fit=lens_imaging_fit_x2_plane_7x7,
        plane_index=0,
        mask=lens_imaging_fit_x2_plane_7x7.mask,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=lens_plotter_util_path,
        output_format="png",
    )

    assert lens_plotter_util_path + "fit_model_image_of_plane_0.png" in plot_patch.paths

    al.lens_plotter_util.plot_model_image_of_plane(
        fit=lens_imaging_fit_x2_plane_7x7,
        plane_index=1,
        mask=lens_imaging_fit_x2_plane_7x7.mask,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=lens_plotter_util_path,
        output_format="png",
    )

    assert lens_plotter_util_path + "fit_model_image_of_plane_1.png" in plot_patch.paths
