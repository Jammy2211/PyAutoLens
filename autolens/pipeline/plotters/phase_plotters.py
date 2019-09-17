import autofit as af
from autolens.data.plotters import imaging_plotters
from autolens.lens.plotters import ray_tracing_plotters
from autolens.lens.plotters import lens_imaging_fit_plotters
from autolens.model.inversion.plotters import inversion_plotters
from autolens.pipeline.plotters import hyper_plotters


def plot_imaging_for_phase(
    imaging_data,
    mask,
    positions,
    extract_array_from_mask,
    zoom_around_mask,
    units,
    should_plot_as_subplot,
    should_plot_image,
    should_plot_noise_map,
    should_plot_psf,
    should_plot_signal_to_noise_map,
    should_plot_absolute_signal_to_noise_map,
    should_plot_potential_chi_squared_map,
    visualize_path,
    subplot_path,
):

    output_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=visualize_path, folder_names=["imaging"]
    )

    if should_plot_as_subplot:

        imaging_plotters.plot_imaging_subplot(
            imaging_data=imaging_data,
            mask=mask,
            extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            positions=positions,
            units=units,
            output_path=subplot_path,
            output_format="png",
        )

    imaging_plotters.plot_imaging_individual(
        imaging_data=imaging_data,
        mask=mask,
        extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask,
        positions=positions,
        should_plot_image=should_plot_image,
        should_plot_noise_map=should_plot_noise_map,
        should_plot_psf=should_plot_psf,
        should_plot_signal_to_noise_map=should_plot_signal_to_noise_map,
        should_plot_absolute_signal_to_noise_map=should_plot_absolute_signal_to_noise_map,
        should_plot_potential_chi_squared_map=should_plot_potential_chi_squared_map,
        units=units,
        output_path=output_path,
        output_format="png",
    )


def plot_ray_tracing_for_phase(
    tracer,
    grid,
    during_analysis,
    mask,
    extract_array_from_mask,
    zoom_around_mask,
    positions,
    units,
    should_plot_as_subplot,
    should_plot_all_at_end_png,
    should_plot_all_at_end_fits,
    should_plot_image,
    should_plot_source_plane,
    should_plot_convergence,
    should_plot_potential,
    should_plot_deflections,
    visualize_path,
    subplot_path,
):
    output_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=visualize_path, folder_names=["ray_tracing"]
    )

    if should_plot_as_subplot:

        ray_tracing_plotters.plot_ray_tracing_subplot(
            tracer=tracer,
            grid=grid,
            mask=mask,
            extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            positions=positions,
            units=units,
            output_path=subplot_path,
            output_format="png",
        )

    ray_tracing_plotters.plot_ray_tracing_individual(
        tracer=tracer,
        grid=grid,
        mask=mask,
        extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask,
        positions=positions,
        should_plot_profile_image=should_plot_image,
        should_plot_source_plane=should_plot_source_plane,
        should_plot_convergence=should_plot_convergence,
        should_plot_potential=should_plot_potential,
        should_plot_deflections=should_plot_deflections,
        units=units,
        output_path=output_path,
        output_format="png",
    )

    if not during_analysis:

        if should_plot_all_at_end_png:

            ray_tracing_plotters.plot_ray_tracing_individual(
                tracer=tracer,
                grid=grid,
                mask=mask,
                extract_array_from_mask=extract_array_from_mask,
                zoom_around_mask=zoom_around_mask,
                positions=positions,
                should_plot_profile_image=True,
                should_plot_source_plane=True,
                should_plot_convergence=True,
                should_plot_potential=True,
                should_plot_deflections=True,
                units=units,
                output_path=output_path,
                output_format="png",
            )

        if should_plot_all_at_end_fits:

            fits_path = af.path_util.make_and_return_path_from_path_and_folder_names(
                path=output_path, folder_names=["fits"]
            )

            ray_tracing_plotters.plot_ray_tracing_individual(
                tracer=tracer,
                grid=grid,
                mask=mask,
                extract_array_from_mask=extract_array_from_mask,
                zoom_around_mask=zoom_around_mask,
                positions=positions,
                should_plot_profile_image=True,
                should_plot_source_plane=True,
                should_plot_convergence=True,
                should_plot_potential=True,
                should_plot_deflections=True,
                output_path=fits_path,
                output_format="fits",
            )


def plot_lens_imaging_fit_for_phase(
    fit,
    during_analysis,
    extract_array_from_mask,
    zoom_around_mask,
    positions,
    units,
    should_plot_mask,
    should_plot_image_plane_pix,
    should_plot_all_at_end_png,
    should_plot_all_at_end_fits,
    should_plot_fit_as_subplot,
    should_plot_fit_of_planes_as_subplot,
    should_plot_inversion_as_subplot,
    should_plot_image,
    should_plot_noise_map,
    should_plot_signal_to_noise_map,
    should_plot_model_image,
    should_plot_residual_map,
    should_plot_normalized_residual_map,
    should_plot_chi_squared_map,
    should_plot_pixelization_residual_map,
    should_plot_pixelization_normalized_residual_map,
    should_plot_pixelization_chi_squared_map,
    should_plot_pixelization_regularization_weights,
    should_plot_subtracted_images_of_planes,
    should_plot_model_images_of_planes,
    should_plot_plane_images_of_planes,
    visualize_path,
    subplot_path,
):

    output_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=visualize_path, folder_names=["lens_fit"]
    )

    if should_plot_fit_as_subplot:

        lens_imaging_fit_plotters.plot_fit_subplot(
            fit=fit,
            should_plot_mask=should_plot_mask,
            extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            positions=positions,
            should_plot_image_plane_pix=should_plot_image_plane_pix,
            units=units,
            output_path=subplot_path,
            output_format="png",
        )

    if should_plot_fit_of_planes_as_subplot:

        lens_imaging_fit_plotters.plot_fit_subplot_of_planes(
            fit=fit,
            should_plot_mask=should_plot_mask,
            extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            positions=positions,
            should_plot_image_plane_pix=should_plot_image_plane_pix,
            units=units,
            output_path=subplot_path,
            output_format="png",
        )

    if should_plot_inversion_as_subplot and fit.tracer.has_pixelization:

        inversion_plotters.plot_inversion_subplot(
            inversion=fit.inversion,
            mask=fit.mask,
            positions=positions,
            extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            output_path=subplot_path,
            output_format="png",
        )

    lens_imaging_fit_plotters.plot_fit_individuals(
        fit=fit,
        should_plot_mask=should_plot_mask,
        extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask,
        positions=positions,
        should_plot_image_plane_pix=should_plot_image_plane_pix,
        should_plot_image=should_plot_image,
        should_plot_noise_map=should_plot_noise_map,
        should_plot_signal_to_noise_map=should_plot_signal_to_noise_map,
        should_plot_model_image=should_plot_model_image,
        should_plot_residual_map=should_plot_residual_map,
        should_plot_chi_squared_map=should_plot_chi_squared_map,
        should_plot_normalized_residual_map=should_plot_normalized_residual_map,
        should_plot_pixelization_residual_map=should_plot_pixelization_residual_map,
        should_plot_pixelization_normalized_residual_map=should_plot_pixelization_normalized_residual_map,
        should_plot_pixelization_chi_squared_map=should_plot_pixelization_chi_squared_map,
        should_plot_pixelization_regularization_weight_map=should_plot_pixelization_regularization_weights,
        should_plot_subtracted_images_of_planes=should_plot_subtracted_images_of_planes,
        should_plot_model_images_of_planes=should_plot_model_images_of_planes,
        should_plot_plane_images_of_planes=should_plot_plane_images_of_planes,
        units=units,
        output_path=output_path,
        output_format="png",
    )

    if not during_analysis:

        if should_plot_all_at_end_png:

            lens_imaging_fit_plotters.plot_fit_individuals(
                fit=fit,
                should_plot_mask=should_plot_mask,
                extract_array_from_mask=extract_array_from_mask,
                zoom_around_mask=zoom_around_mask,
                positions=positions,
                should_plot_image_plane_pix=should_plot_image_plane_pix,
                should_plot_image=True,
                should_plot_noise_map=True,
                should_plot_signal_to_noise_map=True,
                should_plot_model_image=True,
                should_plot_residual_map=True,
                should_plot_normalized_residual_map=True,
                should_plot_chi_squared_map=True,
                should_plot_pixelization_residual_map=True,
                should_plot_pixelization_normalized_residual_map=True,
                should_plot_pixelization_chi_squared_map=True,
                should_plot_pixelization_regularization_weight_map=True,
                should_plot_subtracted_images_of_planes=True,
                should_plot_model_images_of_planes=True,
                should_plot_plane_images_of_planes=True,
                units=units,
                output_path=output_path,
                output_format="png",
            )

        if should_plot_all_at_end_fits:

            fits_path = af.path_util.make_and_return_path_from_path_and_folder_names(
                path=output_path, folder_names=["fits"]
            )

            lens_imaging_fit_plotters.plot_fit_individuals(
                fit=fit,
                should_plot_mask=should_plot_mask,
                extract_array_from_mask=extract_array_from_mask,
                zoom_around_mask=zoom_around_mask,
                positions=positions,
                should_plot_image_plane_pix=should_plot_image_plane_pix,
                should_plot_image=True,
                should_plot_noise_map=True,
                should_plot_signal_to_noise_map=True,
                should_plot_model_image=True,
                should_plot_residual_map=True,
                should_plot_normalized_residual_map=True,
                should_plot_chi_squared_map=True,
                should_plot_pixelization_residual_map=True,
                should_plot_pixelization_normalized_residual_map=True,
                should_plot_pixelization_chi_squared_map=True,
                should_plot_pixelization_regularization_weight_map=True,
                should_plot_subtracted_images_of_planes=True,
                should_plot_model_images_of_planes=True,
                should_plot_plane_images_of_planes=True,
                output_path=fits_path,
                output_format="fits",
            )


def plot_hyper_images_for_phase(
    hyper_model_image_2d,
    hyper_galaxy_image_2d_path_dict,
    binned_hyper_galaxy_image_2d_path_dict,
    mask,
    binned_grid,
    extract_array_from_mask,
    zoom_around_mask,
    units,
    should_plot_hyper_model_image,
    should_plot_hyper_galaxy_images,
    should_plot_binned_hyper_galaxy_images,
    visualize_path,
):

    output_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=visualize_path, folder_names=["hyper_galaxies"]
    )

    if should_plot_hyper_model_image:

        hyper_plotters.plot_hyper_model_image(
            hyper_model_image=hyper_model_image_2d,
            mask=mask,
            extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            units=units,
            output_path=output_path,
            output_format="png",
        )

    if should_plot_hyper_galaxy_images:

        hyper_plotters.plot_hyper_galaxy_images_subplot(
            hyper_galaxy_image_path_dict=hyper_galaxy_image_2d_path_dict,
            mask=mask,
            extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            units=units,
            output_path=output_path,
            output_format="png",
        )

    if (
        should_plot_binned_hyper_galaxy_images
        and binned_hyper_galaxy_image_2d_path_dict is not None
    ):

        hyper_plotters.plot_binned_hyper_galaxy_images_subplot(
            hyper_galaxy_cluster_image_path_dict=binned_hyper_galaxy_image_2d_path_dict,
            mask=binned_grid.mask,
            extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask,
            units=units,
            output_path=output_path,
            output_format="png",
        )
