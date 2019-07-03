from autolens.data.plotters import ccd_plotters
from autolens.lens.plotters import ray_tracing_plotters
from autolens.lens.plotters import lens_fit_plotters
from autolens.model.inversion.plotters import inversion_plotters
from autolens.pipeline.plotters import hyper_plotters

def plot_ccd_for_phase(
        ccd_data, mask, positions, extract_array_from_mask, zoom_around_mask, units,
        should_plot_as_subplot,
        should_plot_image,
        should_plot_noise_map,
        should_plot_psf,
        should_plot_signal_to_noise_map,
        should_plot_absolute_signal_to_noise_map,
        should_plot_potential_chi_squared_map,
        visualize_path):

    output_path = visualize_path

    if should_plot_as_subplot:

        ccd_plotters.plot_ccd_subplot(
            ccd_data=ccd_data, mask=mask, extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask, positions=positions,
            units=units,
            output_path=output_path, output_format='png')

    ccd_plotters.plot_ccd_individual(
        ccd_data=ccd_data, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, positions=positions,
        should_plot_image=should_plot_image,
        should_plot_noise_map=should_plot_noise_map,
        should_plot_psf=should_plot_psf,
        should_plot_signal_to_noise_map=should_plot_signal_to_noise_map,
        should_plot_absolute_signal_to_noise_map=should_plot_absolute_signal_to_noise_map,
        should_plot_potential_chi_squared_map=should_plot_potential_chi_squared_map,
        units=units,
        output_path=output_path, output_format='png')


def plot_ray_tracing_for_phase(
        tracer, during_analysis, mask, extract_array_from_mask, zoom_around_mask, positions, units,
        should_plot_as_subplot,
        should_plot_all_at_end_png,
        should_plot_all_at_end_fits,
        should_plot_image_plane_image,
        should_plot_source_plane,
        should_plot_convergence,
        should_plot_potential,
        should_plot_deflections,
        visualize_path):
    output_path = visualize_path

    if should_plot_as_subplot:

        ray_tracing_plotters.plot_ray_tracing_subplot(
            tracer=tracer, mask=mask, extract_array_from_mask=extract_array_from_mask,
            zoom_around_mask=zoom_around_mask, positions=positions,
            units=units,
            output_path=output_path, output_format='png')

    ray_tracing_plotters.plot_ray_tracing_individual(
        tracer=tracer, mask=mask, extract_array_from_mask=extract_array_from_mask,
        zoom_around_mask=zoom_around_mask, positions=positions,
        should_plot_image_plane_image=should_plot_image_plane_image,
        should_plot_source_plane=should_plot_source_plane,
        should_plot_convergence=should_plot_convergence,
        should_plot_potential=should_plot_potential,
        should_plot_deflections=should_plot_deflections,
        units=units,
        output_path=output_path, output_format='png')

    if not during_analysis:

        if should_plot_all_at_end_png:

            ray_tracing_plotters.plot_ray_tracing_individual(
                tracer=tracer, mask=mask, extract_array_from_mask=extract_array_from_mask,
                zoom_around_mask=zoom_around_mask, positions=positions,
                should_plot_image_plane_image=True,
                should_plot_source_plane=True,
                should_plot_convergence=True,
                should_plot_potential=True,
                should_plot_deflections=True,
                units=units,
                output_path=output_path, output_format='png')

        if should_plot_all_at_end_fits:

            ray_tracing_plotters.plot_ray_tracing_individual(
                tracer=tracer, mask=mask, extract_array_from_mask=extract_array_from_mask,
                zoom_around_mask=zoom_around_mask, positions=positions,
                should_plot_image_plane_image=True,
                should_plot_source_plane=True,
                should_plot_convergence=True,
                should_plot_potential=True,
                should_plot_deflections=True,
                output_path=output_path + 'fits/', output_format='fits')


def plot_lens_fit_for_phase(
        fit, during_analysis, extract_array_from_mask, zoom_around_mask, positions, units,
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
        visualize_path):

    output_path = visualize_path

    if should_plot_fit_as_subplot:

        lens_fit_plotters.plot_fit_subplot(
            fit=fit, should_plot_mask=should_plot_mask,
            extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask,
            positions=positions, should_plot_image_plane_pix=should_plot_image_plane_pix,
            units=units,
            output_path=output_path, output_format='png')

    if should_plot_fit_of_planes_as_subplot:

        lens_fit_plotters.plot_fit_subplot_of_planes(
            fit=fit, should_plot_mask=should_plot_mask,
            extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask,
            positions=positions, should_plot_image_plane_pix=should_plot_image_plane_pix,
            units=units,
            output_path=output_path, output_format='png')

    if should_plot_inversion_as_subplot and fit.tracer.has_pixelization:

        inversion_plotters.plot_inversion_subplot(
            inversion=fit.inversion, mask=fit.mask_2d, positions=positions,
            extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask,
            output_path=output_path, output_format='png')

    lens_fit_plotters.plot_fit_individuals(
        fit=fit, should_plot_mask=should_plot_mask,
        extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask,
        positions=positions, should_plot_image_plane_pix=should_plot_image_plane_pix,
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
        output_path=output_path, output_format='png')

    if not during_analysis:

        if should_plot_all_at_end_png:

            lens_fit_plotters.plot_fit_individuals(
                fit=fit, should_plot_mask=should_plot_mask,
                extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask,
                positions=positions, should_plot_image_plane_pix=should_plot_image_plane_pix,
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
                output_path=output_path, output_format='png')

        if should_plot_all_at_end_fits:

            lens_fit_plotters.plot_fit_individuals(
                fit=fit, should_plot_mask=should_plot_mask,
                extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask,
                positions=positions, should_plot_image_plane_pix=should_plot_image_plane_pix,
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
                output_path=output_path + 'fits/', output_format='fits')

def plot_hyper_images_for_phase(
        hyper_model_image_2d, hyper_galaxy_image_2d_path_dict, hyper_galaxy_cluster_image_2d_path_dict,
        mask, cluster, extract_array_from_mask, zoom_around_mask, units,
        should_plot_hyper_model_image,
        should_plot_hyper_galaxy_images,
        should_plot_hyper_galaxy_cluster_images,
        visualize_path):

    output_path = visualize_path

    if should_plot_hyper_model_image:

        hyper_plotters.plot_hyper_model_image(
            hyper_model_image=hyper_model_image_2d, mask=mask,
            extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask,
            units=units,
            output_path=output_path, output_format='png')

    if should_plot_hyper_galaxy_images:

        hyper_plotters.plot_hyper_galaxy_images_subplot(
            hyper_galaxy_image_path_dict=hyper_galaxy_image_2d_path_dict, mask=mask,
            extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask,
            units=units,
            output_path=output_path, output_format='png')

    if should_plot_hyper_galaxy_cluster_images and hyper_galaxy_cluster_image_2d_path_dict is not None:

        hyper_plotters.plot_hyper_galaxy_cluster_images_subplot(
            hyper_galaxy_cluster_image_path_dict=hyper_galaxy_cluster_image_2d_path_dict, mask=cluster.mask,
            extract_array_from_mask=extract_array_from_mask, zoom_around_mask=zoom_around_mask,
            units=units,
            output_path=output_path, output_format='png')