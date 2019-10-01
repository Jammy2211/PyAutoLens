import autolens as al
import os

import pytest


@pytest.fixture(name="phase_plotter_path")
def make_phase_plotter_setup():
    return "{}/../../test_files/plotting/phase/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


def test__plot_imaging_for_phase(
    imaging_data_7x7, mask_7x7, general_config, phase_plotter_path, plot_patch
):
    al.phase_plotters.plot_imaging_for_phase(
        imaging_data=imaging_data_7x7,
        mask=mask_7x7,
        positions=None,
        units="arcsec",
        zoom_around_mask=True,
        extract_array_from_mask=True,
        should_plot_as_subplot=True,
        should_plot_image=True,
        should_plot_noise_map=False,
        should_plot_psf=True,
        should_plot_signal_to_noise_map=False,
        should_plot_absolute_signal_to_noise_map=False,
        should_plot_potential_chi_squared_map=True,
        visualize_path=phase_plotter_path,
        subplot_path=phase_plotter_path,
    )

    assert phase_plotter_path + "imaging_data.png" in plot_patch.paths
    assert phase_plotter_path + "imaging/imaging_image.png" in plot_patch.paths
    assert phase_plotter_path + "imaging/imaging_noise_map.png" not in plot_patch.paths
    assert phase_plotter_path + "imaging/imaging_psf.png" in plot_patch.paths
    assert (
        phase_plotter_path + "imaging/imaging_signal_to_noise_map.png"
        not in plot_patch.paths
    )
    assert (
        phase_plotter_path + "imaging/imaging_absolute_signal_to_noise_map.png"
        not in plot_patch.paths
    )
    assert (
        phase_plotter_path + "imaging/imaging_potential_chi_squared_map.png"
        in plot_patch.paths
    )


def test__plot_ray_tracing_for_phase__dependent_on_input(
    tracer_x2_plane_7x7, sub_grid_7x7, mask_7x7, phase_plotter_path, plot_patch
):
    al.phase_plotters.plot_ray_tracing_for_phase(
        tracer=tracer_x2_plane_7x7,
        grid=sub_grid_7x7,
        during_analysis=False,
        mask=mask_7x7,
        positions=None,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        units="arcsec",
        should_plot_as_subplot=True,
        should_plot_all_at_end_png=False,
        should_plot_all_at_end_fits=False,
        should_plot_image=True,
        should_plot_source_plane=True,
        should_plot_convergence=False,
        should_plot_potential=True,
        should_plot_deflections=False,
        visualize_path=phase_plotter_path,
        subplot_path=phase_plotter_path,
    )

    assert phase_plotter_path + "tracer.png" in plot_patch.paths
    assert (
        phase_plotter_path + "ray_tracing/tracer_profile_image.png" in plot_patch.paths
    )
    assert (
        phase_plotter_path + "ray_tracing/tracer_source_plane.png" in plot_patch.paths
    )
    assert (
        phase_plotter_path + "ray_tracing/tracer_convergence.png"
        not in plot_patch.paths
    )
    assert phase_plotter_path + "ray_tracing/tracer_potential.png" in plot_patch.paths
    assert (
        phase_plotter_path + "ray_tracing/tracer_deflections_y.png"
        not in plot_patch.paths
    )
    assert (
        phase_plotter_path + "ray_tracing/tracer_deflections_x.png"
        not in plot_patch.paths
    )


def test__lens_fit_for_phase__source_and_lens__depedent_on_input(
    lens_imaging_fit_x2_plane_7x7, phase_plotter_path, plot_patch
):
    al.phase_plotters.plot_lens_imaging_fit_for_phase(
        fit=lens_imaging_fit_x2_plane_7x7,
        during_analysis=False,
        should_plot_mask=True,
        positions=None,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        units="arcsec",
        should_plot_image_plane_pix=True,
        should_plot_all_at_end_png=False,
        should_plot_all_at_end_fits=False,
        should_plot_fit_as_subplot=True,
        should_plot_fit_of_planes_as_subplot=False,
        should_plot_inversion_as_subplot=False,
        should_plot_image=True,
        should_plot_noise_map=False,
        should_plot_signal_to_noise_map=False,
        should_plot_model_image=True,
        should_plot_residual_map=False,
        should_plot_normalized_residual_map=True,
        should_plot_chi_squared_map=True,
        should_plot_pixelization_residual_map=False,
        should_plot_pixelization_normalized_residual_map=False,
        should_plot_pixelization_chi_squared_map=False,
        should_plot_pixelization_regularization_weights=False,
        should_plot_subtracted_images_of_planes=True,
        should_plot_model_images_of_planes=False,
        should_plot_plane_images_of_planes=True,
        visualize_path=phase_plotter_path,
        subplot_path=phase_plotter_path,
    )

    assert phase_plotter_path + "lens_fit.png" in plot_patch.paths
    assert phase_plotter_path + "lens_fit/fit_image.png" in plot_patch.paths
    assert phase_plotter_path + "lens_fit/fit_noise_map.png" not in plot_patch.paths
    assert (
        phase_plotter_path + "lens_fit/fit_signal_to_noise_map.png"
        not in plot_patch.paths
    )
    assert phase_plotter_path + "lens_fit/fit_model_image.png" in plot_patch.paths
    assert phase_plotter_path + "lens_fit/fit_residual_map.png" not in plot_patch.paths
    assert (
        phase_plotter_path + "lens_fit/fit_normalized_residual_map.png"
        in plot_patch.paths
    )
    assert phase_plotter_path + "lens_fit/fit_chi_squared_map.png" in plot_patch.paths
    assert (
        phase_plotter_path + "lens_fit/fit_subtracted_image_of_plane_0.png"
        in plot_patch.paths
    )
    assert (
        phase_plotter_path + "lens_fit/fit_subtracted_image_of_plane_1.png"
        in plot_patch.paths
    )
    assert (
        phase_plotter_path + "lens_fit/fit_model_image_of_plane_0.png"
        not in plot_patch.paths
    )
    assert (
        phase_plotter_path + "lens_fit/fit_model_image_of_plane_1.png"
        not in plot_patch.paths
    )
    assert (
        phase_plotter_path + "lens_fit/fit_plane_image_of_plane_0.png"
        in plot_patch.paths
    )
    assert (
        phase_plotter_path + "lens_fit/fit_plane_image_of_plane_1.png"
        in plot_patch.paths
    )


def test__hyper_images_for_phase__source_and_lens__depedent_on_input(
    hyper_model_image_7x7, binned_grid_7x7, mask_7x7, phase_plotter_path, plot_patch
):
    al.phase_plotters.plot_hyper_images_for_phase(
        hyper_model_image_2d=hyper_model_image_7x7,
        hyper_galaxy_image_2d_path_dict=None,
        binned_hyper_galaxy_image_2d_path_dict=None,
        mask=mask_7x7,
        binned_grid=binned_grid_7x7,
        extract_array_from_mask=True,
        zoom_around_mask=True,
        units="arcsec",
        should_plot_hyper_model_image=True,
        should_plot_hyper_galaxy_images=False,
        should_plot_binned_hyper_galaxy_images=False,
        visualize_path=phase_plotter_path,
    )

    assert (
        phase_plotter_path + "hyper_galaxies/hyper_model_image.png" in plot_patch.paths
    )
    assert (
        phase_plotter_path + "hyper_galaxies/hyper_galaxy_images.png"
        not in plot_patch.paths
    )
    assert (
        phase_plotter_path + "hyper_galaxies/hyper_galaxy_cluster_images.png"
        not in plot_patch.paths
    )
