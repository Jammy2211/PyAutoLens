import autolens as al
import os

import pytest


@pytest.fixture(name="phase_plotter_path")
def make_phase_plotter_setup():
    return "{}/../../test_files/plotting/phase/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


def test__plot_imaging_for_phase(
    imaging_7x7, mask_7x7, general_config, phase_plotter_path, plot_patch
):
    al.plot.phase.imaging_of_phase(
        imaging=imaging_7x7,
        mask=mask_7x7,
        positions=None,
        kpc_per_arcsec=2.0,
        unit_label="kpc",
        plot_as_subplot=True,
        plot_image=True,
        plot_noise_map=False,
        plot_psf=True,
        plot_signal_to_noise_map=False,
        plot_absolute_signal_to_noise_map=False,
        plot_potential_chi_squared_map=True,
        visualize_path=phase_plotter_path,
        subplot_path=phase_plotter_path,
    )

    assert phase_plotter_path + "imaging.png" in plot_patch.paths
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


def test__plot_interferometer_for_phase(
    interferometer_7, mask_7x7, general_config, phase_plotter_path, plot_patch
):
    al.plot.phase.interferometer_of_phase(
        interferometer=interferometer_7,
        kpc_per_arcsec=2.0,
        unit_label="kpc",
        plot_as_subplot=True,
        plot_visibilities=False,
        plot_uv_wavelengths=False,
        plot_primary_beam=True,
        visualize_path=phase_plotter_path,
        subplot_path=phase_plotter_path,
    )

    assert phase_plotter_path + "interferometer.png" in plot_patch.paths
    assert (
        phase_plotter_path + "interferometer/interferometer_visibilities.png"
        not in plot_patch.paths
    )
    assert (
        phase_plotter_path + "interferometer/interferometer_primary_beam.png"
        in plot_patch.paths
    )


def test__plot_ray_tracing_for_phase__dependent_on_input(
    tracer_x2_plane_7x7, sub_grid_7x7, mask_7x7, phase_plotter_path, plot_patch
):
    al.plot.phase.ray_tracing_of_phase(
        tracer=tracer_x2_plane_7x7,
        grid=sub_grid_7x7,
        during_analysis=False,
        plot_in_kpc=True,
        mask=mask_7x7,
        include_caustics=True,
        include_critical_curves=True,
        positions=None,
        plot_as_subplot=True,
        plot_all_at_end_png=False,
        plot_all_at_end_fits=False,
        plot_image=True,
        plot_source_plane=True,
        plot_convergence=False,
        plot_potential=True,
        plot_deflections=False,
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


def test__imaging_fit_for_phase__source_and_lens__depedent_on_input(
    masked_imaging_fit_x2_plane_7x7, phase_plotter_path, plot_patch
):
    al.plot.phase.imaging_fit_of_phase(
        fit=masked_imaging_fit_x2_plane_7x7,
        during_analysis=False,
        include_mask=True,
        include_positions=True,
        plot_in_kpc=True,
        include_critical_curves=True,
        include_caustics=True,
        include_image_plane_pix=True,
        plot_all_at_end_png=False,
        plot_all_at_end_fits=False,
        plot_fit_as_subplot=True,
        plot_fit_of_planes_as_subplot=False,
        plot_inversion_as_subplot=False,
        plot_image=True,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_model_image=True,
        plot_residual_map=False,
        plot_normalized_residual_map=True,
        plot_chi_squared_map=True,
        plot_inversion_reconstruction=False,
        plot_inversion_errors=False,
        plot_inversion_residual_map=False,
        plot_inversion_normalized_residual_map=False,
        plot_inversion_chi_squared_map=False,
        plot_inversion_regularization_weights=False,
        plot_inversion_interpolated_reconstruction=False,
        plot_inversion_interpolated_errors=False,
        plot_subtracted_images_of_planes=True,
        plot_model_images_of_planes=False,
        plot_plane_images_of_planes=True,
        visualize_path=phase_plotter_path,
        subplot_path=phase_plotter_path,
    )

    assert phase_plotter_path + "fit.png" in plot_patch.paths
    assert phase_plotter_path + "fit/fit_image.png" in plot_patch.paths
    assert phase_plotter_path + "fit/fit_noise_map.png" not in plot_patch.paths
    assert (
        phase_plotter_path + "fit/fit_signal_to_noise_map.png" not in plot_patch.paths
    )
    assert phase_plotter_path + "fit/fit_model_image.png" in plot_patch.paths
    assert phase_plotter_path + "fit/fit_residual_map.png" not in plot_patch.paths
    assert (
        phase_plotter_path + "fit/fit_normalized_residual_map.png" in plot_patch.paths
    )
    assert phase_plotter_path + "fit/fit_chi_squared_map.png" in plot_patch.paths
    assert (
        phase_plotter_path + "fit/fit_subtracted_image_of_plane_0.png"
        in plot_patch.paths
    )
    assert (
        phase_plotter_path + "fit/fit_subtracted_image_of_plane_1.png"
        in plot_patch.paths
    )
    assert (
        phase_plotter_path + "fit/fit_model_image_of_plane_0.png"
        not in plot_patch.paths
    )
    assert (
        phase_plotter_path + "fit/fit_model_image_of_plane_1.png"
        not in plot_patch.paths
    )
    assert phase_plotter_path + "fit/fit_plane_image_of_plane_0.png" in plot_patch.paths
    assert phase_plotter_path + "fit/fit_plane_image_of_plane_1.png" in plot_patch.paths


def test__interferometer_fit_for_phase__source_and_lens__depedent_on_input(
    masked_interferometer_fit_x2_plane_7x7, phase_plotter_path, plot_patch
):
    al.plot.phase.interferometer_fit_of_phase(
        fit=masked_interferometer_fit_x2_plane_7x7,
        during_analysis=False,
        include_mask=True,
        include_positions=True,
        plot_in_kpc=True,
        include_critical_curves=True,
        include_caustics=True,
        include_image_plane_pix=True,
        plot_all_at_end_png=False,
        plot_all_at_end_fits=False,
        plot_fit_as_subplot=True,
        plot_inversion_as_subplot=False,
        plot_visibilities=True,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_model_visibilities=True,
        plot_residual_map=False,
        plot_normalized_residual_map=True,
        plot_chi_squared_map=True,
        plot_inversion_reconstruction=False,
        plot_inversion_errors=False,
        plot_inversion_residual_map=False,
        plot_inversion_normalized_residual_map=False,
        plot_inversion_chi_squared_map=False,
        plot_inversion_regularization_weights=False,
        plot_inversion_interpolated_reconstruction=False,
        plot_inversion_interpolated_errors=False,
        visualize_path=phase_plotter_path,
        subplot_path=phase_plotter_path,
    )

    assert phase_plotter_path + "fit.png" in plot_patch.paths
    assert phase_plotter_path + "fit/fit_visibilities.png" in plot_patch.paths
    assert phase_plotter_path + "fit/fit_noise_map.png" not in plot_patch.paths
    assert (
        phase_plotter_path + "fit/fit_signal_to_noise_map.png" not in plot_patch.paths
    )
    assert phase_plotter_path + "fit/fit_model_visibilities.png" in plot_patch.paths
    assert (
        phase_plotter_path + "fit/fit_residual_map_vs_uv_distances_real.png"
        not in plot_patch.paths
    )
    assert (
        phase_plotter_path + "fit/fit_normalized_residual_map_vs_uv_distances_real.png"
        in plot_patch.paths
    )
    assert (
        phase_plotter_path + "fit/fit_chi_squared_map_vs_uv_distances_real.png"
        in plot_patch.paths
    )


def test__hyper_images_for_phase__source_and_lens__depedent_on_input(
    hyper_model_image_7x7, mask_7x7, phase_plotter_path, plot_patch
):
    al.plot.phase.plot_hyper_images_for_phase(
        hyper_model_image=hyper_model_image_7x7,
        hyper_galaxy_image_path_dict=None,
        kpc_per_arcsec=2.0,
        unit_label="arcsec",
        mask=mask_7x7,
        plot_hyper_model_image=True,
        plot_hyper_galaxy_images=False,
        visualize_path=phase_plotter_path,
    )

    assert (
        phase_plotter_path + "hyper_galaxies/hyper_model_image.png" in plot_patch.paths
    )
    assert (
        phase_plotter_path + "hyper_galaxies/hyper_galaxy_images.png"
        not in plot_patch.paths
    )
