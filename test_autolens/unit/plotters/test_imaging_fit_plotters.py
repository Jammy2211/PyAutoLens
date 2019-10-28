import pytest
import os

import autolens as al


@pytest.fixture(name="imaging_fit_plotter_path")
def make_imaging_fit_plotter_setup():
    return "{}/../../test_files/plotting/fit/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


def test__subtracted_image_of_plane_is_output(
    masked_imaging_fit_x1_plane_7x7,
    masked_imaging_fit_x2_plane_7x7,
    imaging_fit_plotter_path,
    plot_patch,
):

    al.plot.imaging_fit.subtracted_image_of_plane(
        fit=masked_imaging_fit_x1_plane_7x7,
        plane_index=0,
        mask_overlay=masked_imaging_fit_x2_plane_7x7.mask,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=imaging_fit_plotter_path,
        output_format="png",
    )

    assert (
        imaging_fit_plotter_path + "fit_subtracted_image_of_plane_0.png"
        in plot_patch.paths
    )

    al.plot.imaging_fit.subtracted_image_of_plane(
        fit=masked_imaging_fit_x2_plane_7x7,
        plane_index=0,
        mask_overlay=masked_imaging_fit_x2_plane_7x7.mask,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=imaging_fit_plotter_path,
        output_format="png",
    )

    assert (
        imaging_fit_plotter_path + "fit_subtracted_image_of_plane_0.png"
        in plot_patch.paths
    )

    al.plot.imaging_fit.subtracted_image_of_plane(
        fit=masked_imaging_fit_x2_plane_7x7,
        plane_index=1,
        mask_overlay=masked_imaging_fit_x2_plane_7x7.mask,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=imaging_fit_plotter_path,
        output_format="png",
    )

    assert (
        imaging_fit_plotter_path + "fit_subtracted_image_of_plane_1.png"
        in plot_patch.paths
    )


def test__model_image_of_plane_is_output(
    masked_imaging_fit_x1_plane_7x7,
    masked_imaging_fit_x2_plane_7x7,
    imaging_fit_plotter_path,
    plot_patch,
):

    al.plot.imaging_fit.model_image_of_plane(
        fit=masked_imaging_fit_x1_plane_7x7,
        plane_index=0,
        mask_overlay=masked_imaging_fit_x2_plane_7x7.mask,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=imaging_fit_plotter_path,
        output_format="png",
    )

    assert (
        imaging_fit_plotter_path + "fit_model_image_of_plane_0.png" in plot_patch.paths
    )

    al.plot.imaging_fit.model_image_of_plane(
        fit=masked_imaging_fit_x2_plane_7x7,
        plane_index=0,
        mask_overlay=masked_imaging_fit_x2_plane_7x7.mask,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=imaging_fit_plotter_path,
        output_format="png",
    )

    assert (
        imaging_fit_plotter_path + "fit_model_image_of_plane_0.png" in plot_patch.paths
    )

    al.plot.imaging_fit.model_image_of_plane(
        fit=masked_imaging_fit_x2_plane_7x7,
        plane_index=1,
        mask_overlay=masked_imaging_fit_x2_plane_7x7.mask,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=imaging_fit_plotter_path,
        output_format="png",
    )

    assert (
        imaging_fit_plotter_path + "fit_model_image_of_plane_1.png" in plot_patch.paths
    )


def test__fit_sub_plot(
    masked_imaging_fit_x2_plane_7x7, imaging_fit_plotter_path, plot_patch
):

    al.plot.imaging_fit.subplot(
        fit=masked_imaging_fit_x2_plane_7x7,
        should_plot_mask_overlay=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=imaging_fit_plotter_path,
        output_format="png",
    )

    assert imaging_fit_plotter_path + "fit.png" in plot_patch.paths


def test__fit_for_plane_subplot(
    masked_imaging_fit_x1_plane_7x7,
    masked_imaging_fit_x2_plane_7x7,
    imaging_fit_plotter_path,
    plot_patch,
):

    al.plot.imaging_fit.subplot_for_plane(
        fit=masked_imaging_fit_x1_plane_7x7,
        plane_index=0,
        should_plot_mask_overlay=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=imaging_fit_plotter_path,
        output_format="png",
    )

    assert imaging_fit_plotter_path + "lens_fit_plane_0.png" in plot_patch.paths

    al.plot.imaging_fit.subplot_for_plane(
        fit=masked_imaging_fit_x2_plane_7x7,
        plane_index=0,
        should_plot_mask_overlay=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=imaging_fit_plotter_path,
        output_format="png",
    )

    assert imaging_fit_plotter_path + "lens_fit_plane_0.png" in plot_patch.paths

    al.plot.imaging_fit.subplot_for_plane(
        fit=masked_imaging_fit_x2_plane_7x7,
        plane_index=1,
        should_plot_mask_overlay=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=imaging_fit_plotter_path,
        output_format="png",
    )

    assert imaging_fit_plotter_path + "lens_fit_plane_1.png" in plot_patch.paths


def test__fit_for_planes_subplot(
    masked_imaging_fit_x1_plane_7x7,
    masked_imaging_fit_x2_plane_7x7,
    imaging_fit_plotter_path,
    plot_patch,
):

    al.plot.imaging_fit.subplot_of_planes(
        fit=masked_imaging_fit_x1_plane_7x7,
        should_plot_mask_overlay=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=imaging_fit_plotter_path,
        output_format="png",
    )

    assert imaging_fit_plotter_path + "lens_fit_plane_0.png" in plot_patch.paths

    al.plot.imaging_fit.subplot_of_planes(
        fit=masked_imaging_fit_x2_plane_7x7,
        should_plot_mask_overlay=True,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=imaging_fit_plotter_path,
        output_format="png",
    )

    assert imaging_fit_plotter_path + "lens_fit_plane_0.png" in plot_patch.paths
    assert imaging_fit_plotter_path + "lens_fit_plane_1.png" in plot_patch.paths


def test__fit_individuals__source_and_lens__depedent_on_input(
    masked_imaging_fit_x1_plane_7x7,
    masked_imaging_fit_x2_plane_7x7,
    imaging_fit_plotter_path,
    plot_patch,
):

    al.plot.imaging_fit.individuals(
        fit=masked_imaging_fit_x1_plane_7x7,
        should_plot_image=True,
        should_plot_noise_map=False,
        should_plot_signal_to_noise_map=False,
        should_plot_model_image=True,
        should_plot_chi_squared_map=True,
        should_plot_subtracted_images_of_planes=True,
        should_plot_model_images_of_planes=True,
        should_plot_plane_images_of_planes=True,
        output_path=imaging_fit_plotter_path,
        output_format="png",
    )

    assert imaging_fit_plotter_path + "fit_image.png" in plot_patch.paths

    assert imaging_fit_plotter_path + "fit_noise_map.png" not in plot_patch.paths

    assert (
        imaging_fit_plotter_path + "fit_signal_to_noise_map.png" not in plot_patch.paths
    )

    assert imaging_fit_plotter_path + "fit_model_image.png" in plot_patch.paths

    assert imaging_fit_plotter_path + "fit_residual_map.png" not in plot_patch.paths

    assert (
        imaging_fit_plotter_path + "fit_normalized_residual_map.png"
        not in plot_patch.paths
    )

    assert imaging_fit_plotter_path + "fit_chi_squared_map.png" in plot_patch.paths

    assert (
        imaging_fit_plotter_path + "fit_subtracted_image_of_plane_0.png"
        in plot_patch.paths
    )

    assert (
        imaging_fit_plotter_path + "fit_model_image_of_plane_0.png" in plot_patch.paths
    )

    assert (
        imaging_fit_plotter_path + "fit_plane_image_of_plane_0.png" in plot_patch.paths
    )

    al.plot.imaging_fit.individuals(
        fit=masked_imaging_fit_x2_plane_7x7,
        should_plot_image=True,
        should_plot_noise_map=False,
        should_plot_signal_to_noise_map=False,
        should_plot_model_image=True,
        should_plot_chi_squared_map=True,
        should_plot_subtracted_images_of_planes=True,
        should_plot_model_images_of_planes=True,
        should_plot_plane_images_of_planes=True,
        output_path=imaging_fit_plotter_path,
        output_format="png",
    )

    assert imaging_fit_plotter_path + "fit_image.png" in plot_patch.paths

    assert imaging_fit_plotter_path + "fit_noise_map.png" not in plot_patch.paths

    assert (
        imaging_fit_plotter_path + "fit_signal_to_noise_map.png" not in plot_patch.paths
    )

    assert imaging_fit_plotter_path + "fit_model_image.png" in plot_patch.paths

    assert imaging_fit_plotter_path + "fit_residual_map.png" not in plot_patch.paths

    assert (
        imaging_fit_plotter_path + "fit_normalized_residual_map.png"
        not in plot_patch.paths
    )

    assert imaging_fit_plotter_path + "fit_chi_squared_map.png" in plot_patch.paths

    assert (
        imaging_fit_plotter_path + "fit_subtracted_image_of_plane_0.png"
        in plot_patch.paths
    )
    assert (
        imaging_fit_plotter_path + "fit_subtracted_image_of_plane_1.png"
        in plot_patch.paths
    )

    assert (
        imaging_fit_plotter_path + "fit_model_image_of_plane_0.png" in plot_patch.paths
    )
    assert (
        imaging_fit_plotter_path + "fit_model_image_of_plane_1.png" in plot_patch.paths
    )

    assert (
        imaging_fit_plotter_path + "fit_plane_image_of_plane_0.png" in plot_patch.paths
    )
    assert (
        imaging_fit_plotter_path + "fit_plane_image_of_plane_1.png" in plot_patch.paths
    )
