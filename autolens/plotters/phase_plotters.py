import autofit as af
import autoarray as aa
from autolens.plotters import ray_tracing_plotters, hyper_plotters
from autolens.plotters.fit_imaging_plotters import fit_imaging_plotters


def imaging_of_phase(
    imaging,
    mask,
    positions,
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

        aa.plot.imaging.subplot(
            imaging=imaging,
            mask=mask,
            positions=positions,
            units=units,
            output_path=subplot_path,
            output_format="png",
        )

    aa.plot.imaging.individual(
        imaging=imaging,
        mask=mask,
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


def ray_tracing_of_phase(
    tracer,
    grid,
    during_analysis,
    mask,
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

        al.plot.tracer.subplot(
            tracer=tracer,
            grid=grid,
            mask=mask,
            positions=positions,
            units=units,
            output_path=subplot_path,
            output_format="png",
        )

    al.plot.tracer.individual(
        tracer=tracer,
        grid=grid,
        mask=mask,
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

            al.plot.tracer.individual(
                tracer=tracer,
                grid=grid,
                mask=mask,
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

            al.plot.tracer.individual(
                tracer=tracer,
                grid=grid,
                mask=mask,
                positions=positions,
                should_plot_profile_image=True,
                should_plot_source_plane=True,
                should_plot_convergence=True,
                should_plot_potential=True,
                should_plot_deflections=True,
                output_path=fits_path,
                output_format="fits",
            )


def imaging_fit_of_phase(
    fit,
    during_analysis,
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
    should_plot_inversion_residual_map,
    should_plot_inversion_normalized_residual_map,
    should_plot_inversion_chi_squared_map,
    should_plot_inversion_regularization_weights,
    should_plot_subtracted_images_of_planes,
    should_plot_model_images_of_planes,
    should_plot_plane_images_of_planes,
    visualize_path,
    subplot_path,
):

    output_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=visualize_path, folder_names=["fit"]
    )

    if should_plot_fit_as_subplot:

        fit_imaging_plotters.subplot(
            fit=fit,
            should_plot_mask=should_plot_mask,
            positions=positions,
            should_plot_image_plane_pix=should_plot_image_plane_pix,
            units=units,
            output_path=subplot_path,
            output_format="png",
        )

    if should_plot_fit_of_planes_as_subplot:

        fit_imaging_plotters.subplot_of_planes(
            fit=fit,
            should_plot_mask=should_plot_mask,
            positions=positions,
            should_plot_image_plane_pix=should_plot_image_plane_pix,
            units=units,
            output_path=subplot_path,
            output_format="png",
        )

    if should_plot_inversion_as_subplot and fit.tracer.has_pixelization:

        aa.plot.inversion.subplot(
            inversion=fit.inversion,
            mask=fit.mask,
            positions=positions,
            output_path=subplot_path,
            output_format="png",
        )

    fit_imaging_plotters.individuals(
        fit=fit,
        should_plot_mask=should_plot_mask,
        positions=positions,
        should_plot_image_plane_pix=should_plot_image_plane_pix,
        should_plot_image=should_plot_image,
        should_plot_noise_map=should_plot_noise_map,
        should_plot_signal_to_noise_map=should_plot_signal_to_noise_map,
        should_plot_model_image=should_plot_model_image,
        should_plot_residual_map=should_plot_residual_map,
        should_plot_chi_squared_map=should_plot_chi_squared_map,
        should_plot_normalized_residual_map=should_plot_normalized_residual_map,
        should_plot_inversion_residual_map=should_plot_inversion_residual_map,
        should_plot_inversion_normalized_residual_map=should_plot_inversion_normalized_residual_map,
        should_plot_inversion_chi_squared_map=should_plot_inversion_chi_squared_map,
        should_plot_inversion_regularization_weight_map=should_plot_inversion_regularization_weights,
        should_plot_subtracted_images_of_planes=should_plot_subtracted_images_of_planes,
        should_plot_model_images_of_planes=should_plot_model_images_of_planes,
        should_plot_plane_images_of_planes=should_plot_plane_images_of_planes,
        units=units,
        output_path=output_path,
        output_format="png",
    )

    if not during_analysis:

        if should_plot_all_at_end_png:

            fit_imaging_plotters.individuals(
                fit=fit,
                should_plot_mask=should_plot_mask,
                positions=positions,
                should_plot_image_plane_pix=should_plot_image_plane_pix,
                should_plot_image=True,
                should_plot_noise_map=True,
                should_plot_signal_to_noise_map=True,
                should_plot_model_image=True,
                should_plot_residual_map=True,
                should_plot_normalized_residual_map=True,
                should_plot_chi_squared_map=True,
                should_plot_inversion_residual_map=True,
                should_plot_inversion_normalized_residual_map=True,
                should_plot_inversion_chi_squared_map=True,
                should_plot_inversion_regularization_weight_map=True,
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

            fit_imaging_plotters.individuals(
                fit=fit,
                should_plot_mask=should_plot_mask,
                positions=positions,
                should_plot_image_plane_pix=should_plot_image_plane_pix,
                should_plot_image=True,
                should_plot_noise_map=True,
                should_plot_signal_to_noise_map=True,
                should_plot_model_image=True,
                should_plot_residual_map=True,
                should_plot_normalized_residual_map=True,
                should_plot_chi_squared_map=True,
                should_plot_inversion_residual_map=True,
                should_plot_inversion_normalized_residual_map=True,
                should_plot_inversion_chi_squared_map=True,
                should_plot_inversion_regularization_weight_map=True,
                should_plot_subtracted_images_of_planes=True,
                should_plot_model_images_of_planes=True,
                should_plot_plane_images_of_planes=True,
                output_path=fits_path,
                output_format="fits",
            )


def plot_hyper_images_for_phase(
    hyper_model_image,
    hyper_galaxy_image_path_dict,
    mask,
    units,
    should_plot_hyper_model_image,
    should_plot_hyper_galaxy_images,
    visualize_path,
):

    output_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=visualize_path, folder_names=["hyper_galaxies"]
    )

    if should_plot_hyper_model_image:

        hyper_plotters.hyper_model_image(
            hyper_model_image=hyper_model_image,
            mask=mask,
            units=units,
            output_path=output_path,
            output_format="png",
        )

    if should_plot_hyper_galaxy_images:

        hyper_plotters.subplot_of_hyper_galaxy_images(
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            mask=mask,
            units=units,
            output_path=output_path,
            output_format="png",
        )
