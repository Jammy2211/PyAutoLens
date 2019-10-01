import autolens as al
import pytest
import os


@pytest.fixture(name="lens_plotter_util_path")
def make_lens_plotter_util_path_setup():
    return "{}/../../test_files/plotting/lens_plotter_util/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


def test__fit_quantities_are_output(
    lens_imaging_fit_x2_plane_7x7, lens_plotter_util_path, plot_patch
):

    al.lens_plotter_util.plot_image(
        fit=lens_imaging_fit_x2_plane_7x7,
        mask=lens_imaging_fit_x2_plane_7x7.mask,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=lens_plotter_util_path,
        output_format="png",
    )

    assert lens_plotter_util_path + "fit_image.png" in plot_patch.paths

    al.lens_plotter_util.plot_noise_map(
        fit=lens_imaging_fit_x2_plane_7x7,
        mask=lens_imaging_fit_x2_plane_7x7.mask,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=lens_plotter_util_path,
        output_format="png",
    )

    assert lens_plotter_util_path + "fit_noise_map.png" in plot_patch.paths

    al.lens_plotter_util.plot_signal_to_noise_map(
        fit=lens_imaging_fit_x2_plane_7x7,
        mask=lens_imaging_fit_x2_plane_7x7.mask,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=lens_plotter_util_path,
        output_format="png",
    )

    assert lens_plotter_util_path + "fit_signal_to_noise_map.png" in plot_patch.paths

    al.lens_plotter_util.plot_model_data(
        fit=lens_imaging_fit_x2_plane_7x7,
        mask=lens_imaging_fit_x2_plane_7x7.mask,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=lens_plotter_util_path,
        output_format="png",
    )

    assert lens_plotter_util_path + "fit_model_image.png" in plot_patch.paths

    al.lens_plotter_util.plot_residual_map(
        fit=lens_imaging_fit_x2_plane_7x7,
        mask=lens_imaging_fit_x2_plane_7x7.mask,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=lens_plotter_util_path,
        output_format="png",
    )

    assert lens_plotter_util_path + "fit_residual_map.png" in plot_patch.paths

    al.lens_plotter_util.plot_normalized_residual_map(
        fit=lens_imaging_fit_x2_plane_7x7,
        mask=lens_imaging_fit_x2_plane_7x7.mask,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=lens_plotter_util_path,
        output_format="png",
    )

    assert (
        lens_plotter_util_path + "fit_normalized_residual_map.png" in plot_patch.paths
    )

    al.lens_plotter_util.plot_chi_squared_map(
        fit=lens_imaging_fit_x2_plane_7x7,
        mask=lens_imaging_fit_x2_plane_7x7.mask,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=lens_plotter_util_path,
        output_format="png",
    )

    assert lens_plotter_util_path + "fit_chi_squared_map.png" in plot_patch.paths


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
        extract_array_from_mask=True,
        zoom_around_mask=True,
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
        extract_array_from_mask=True,
        zoom_around_mask=True,
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
        extract_array_from_mask=True,
        zoom_around_mask=True,
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
        extract_array_from_mask=True,
        zoom_around_mask=True,
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
        extract_array_from_mask=True,
        zoom_around_mask=True,
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
        extract_array_from_mask=True,
        zoom_around_mask=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=lens_plotter_util_path,
        output_format="png",
    )

    assert lens_plotter_util_path + "fit_model_image_of_plane_1.png" in plot_patch.paths
