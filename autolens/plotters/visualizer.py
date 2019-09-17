import autofit as af
from autolens.pipeline.plotters import phase_plotters


class Visualizer:
    def __init__(self, analysis):
        self.analysis = analysis
        self.should_plot_image_plane_pix = af.conf.instance.visualize.get(
            "figures", "plot_image_plane_adaptive_pixelization_grid", bool
        )

        self.plot_data_as_subplot = af.conf.instance.visualize.get(
            "plots", "plot_data_as_subplot", bool
        )

        self.plot_data_image = af.conf.instance.visualize.get(
            "plots", "plot_data_image", bool
        )

        self.plot_data_noise_map = af.conf.instance.visualize.get(
            "plots", "plot_data_noise_map", bool
        )

        self.plot_data_psf = af.conf.instance.visualize.get(
            "plots", "plot_data_psf", bool
        )

        self.plot_data_signal_to_noise_map = af.conf.instance.visualize.get(
            "plots", "plot_data_signal_to_noise_map", bool
        )

        self.plot_data_absolute_signal_to_noise_map = af.conf.instance.visualize.get(
            "plots", "plot_data_absolute_signal_to_noise_map", bool
        )

        self.plot_data_potential_chi_squared_map = af.conf.instance.visualize.get(
            "plots", "plot_data_potential_chi_squared_map", bool
        )

        self.plot_lens_fit_all_at_end_png = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_all_at_end_png", bool
        )
        self.plot_lens_fit_all_at_end_fits = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_all_at_end_fits", bool
        )

        self.plot_lens_fit_as_subplot = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_as_subplot", bool
        )

        self.plot_lens_fit_of_planes_as_subplot = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_of_planes_as_subplot", bool
        )

        self.plot_lens_fit_inversion_as_subplot = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_inversion_as_subplot", bool
        )

        self.plot_lens_fit_image = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_image", bool
        )

        self.plot_lens_fit_noise_map = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_noise_map", bool
        )

        self.plot_lens_fit_signal_to_noise_map = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_signal_to_noise_map", bool
        )

        self.plot_lens_fit_model_image = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_model_image", bool
        )

        self.plot_lens_fit_residual_map = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_residual_map", bool
        )

        self.plot_lens_fit_normalized_residual_map = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_normalized_residual_map", bool
        )

        self.plot_lens_fit_chi_squared_map = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_chi_squared_map", bool
        )

        self.plot_lens_fit_contribution_maps = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_contribution_maps", bool
        )

        self.plot_lens_fit_pixelization_residual_map = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_pixelization_residual_map", bool
        )

        self.plot_lens_fit_pixelization_normalized_residuals = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_pixelization_normalized_residual_map", bool
        )

        self.plot_lens_fit_pixelization_chi_squared_map = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_pixelization_chi_squared_map", bool
        )

        self.plot_lens_fit_pixelization_regularization_weights = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_pixelization_regularization_weight_map", bool
        )

        self.plot_lens_fit_subtracted_images_of_planes = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_subtracted_images_of_planes", bool
        )

        self.plot_lens_fit_model_images_of_planes = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_model_images_of_planes", bool
        )

        self.plot_lens_fit_plane_images_of_planes = af.conf.instance.visualize.get(
            "plots", "plot_lens_fit_plane_images_of_planes", bool
        )

        self.plot_hyper_model_image = af.conf.instance.visualize.get(
            "plots", "plot_hyper_model_image", bool
        )

        self.plot_hyper_galaxy_images = af.conf.instance.visualize.get(
            "plots", "plot_hyper_galaxy_images", bool
        )

        self.plot_binned_hyper_galaxy_images = af.conf.instance.visualize.get(
            "plots", "plot_binned_hyper_galaxy_images", bool
        )

        self.should_plot_mask = af.conf.instance.visualize.get(
            "figures", "plot_mask_on_images", bool
        )
        self.extract_array_from_mask = af.conf.instance.visualize.get(
            "figures", "extract_images_from_mask", bool
        )
        self.zoom_around_mask = af.conf.instance.visualize.get(
            "figures", "zoom_around_mask_of_images", bool
        )
        self.should_plot_positions = af.conf.instance.visualize.get(
            "figures", "plot_positions_on_images", bool
        )
        self.plot_units = af.conf.instance.visualize.get(
            "figures", "plot_units", str
        ).strip()

        self.plot_ray_tracing_all_at_end_png = af.conf.instance.visualize.get(
            "plots", "plot_ray_tracing_all_at_end_png", bool
        )
        self.plot_ray_tracing_all_at_end_fits = af.conf.instance.visualize.get(
            "plots", "plot_ray_tracing_all_at_end_fits", bool
        )

        self.plot_ray_tracing_as_subplot = af.conf.instance.visualize.get(
            "plots", "plot_ray_tracing_as_subplot", bool
        )
        self.plot_ray_tracing_profile_image = af.conf.instance.visualize.get(
            "plots", "plot_ray_tracing_profile_image", bool
        )
        self.plot_ray_tracing_source_plane = af.conf.instance.visualize.get(
            "plots", "plot_ray_tracing_source_plane_image", bool
        )
        self.plot_ray_tracing_convergence = af.conf.instance.visualize.get(
            "plots", "plot_ray_tracing_convergence", bool
        )
        self.plot_ray_tracing_potential = af.conf.instance.visualize.get(
            "plots", "plot_ray_tracing_potential", bool
        )
        self.plot_ray_tracing_deflections = af.conf.instance.visualize.get(
            "plots", "plot_ray_tracing_deflections", bool
        )
        self.plot_ray_tracing_magnification = af.conf.instance.visualize.get(
            "plots", "plot_ray_tracing_magnification", bool
        )

    def visualize(
            self,
            instance,
            image_path,
            during_analysis
    ):
        subplot_path = af.path_util.make_and_return_path_from_path_and_folder_names(
            path=image_path, folder_names=["subplots"]
        )

        instance = self.analysis.associate_images(instance=instance)

        mask = self.analysis.lens_imaging_data.mask if self.should_plot_mask else None
        positions = self.analysis.lens_imaging_data.positions if self.should_plot_positions else None

        tracer = self.analysis.tracer_for_instance(instance=instance)

        phase_plotters.plot_ray_tracing_for_phase(
            tracer=tracer,
            grid=self.analysis.lens_imaging_data.grid,
            during_analysis=during_analysis,
            mask=mask,
            extract_array_from_mask=self.extract_array_from_mask,
            zoom_around_mask=self.zoom_around_mask,
            positions=positions,
            units=self.plot_units,
            should_plot_as_subplot=self.plot_ray_tracing_as_subplot,
            should_plot_all_at_end_png=self.plot_ray_tracing_all_at_end_png,
            should_plot_all_at_end_fits=self.plot_ray_tracing_all_at_end_fits,
            should_plot_image_plane_image=self.plot_ray_tracing_profile_image,
            should_plot_source_plane=self.plot_ray_tracing_source_plane,
            should_plot_convergence=self.plot_ray_tracing_convergence,
            should_plot_potential=self.plot_ray_tracing_potential,
            should_plot_deflections=self.plot_ray_tracing_deflections,
            visualize_path=image_path,
            subplot_path=subplot_path,
        )

        hyper_image_sky = self.analysis.hyper_image_sky_for_instance(instance=instance)

        hyper_background_noise = self.analysis.hyper_background_noise_for_instance(
            instance=instance
        )

        fit = self.analysis.lens_imaging_fit_for_tracer(
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        phase_plotters.plot_lens_imaging_fit_for_phase(
            fit=fit,
            during_analysis=during_analysis,
            should_plot_mask=self.should_plot_mask,
            extract_array_from_mask=self.extract_array_from_mask,
            zoom_around_mask=self.zoom_around_mask,
            positions=positions,
            should_plot_image_plane_pix=self.should_plot_image_plane_pix,
            should_plot_all_at_end_png=self.plot_lens_fit_all_at_end_png,
            should_plot_all_at_end_fits=self.plot_lens_fit_all_at_end_fits,
            should_plot_fit_as_subplot=self.plot_lens_fit_as_subplot,
            should_plot_fit_of_planes_as_subplot=self.plot_lens_fit_of_planes_as_subplot,
            should_plot_inversion_as_subplot=self.plot_lens_fit_inversion_as_subplot,
            should_plot_image=self.plot_lens_fit_image,
            should_plot_noise_map=self.plot_lens_fit_noise_map,
            should_plot_signal_to_noise_map=self.plot_lens_fit_signal_to_noise_map,
            should_plot_model_image=self.plot_lens_fit_model_image,
            should_plot_residual_map=self.plot_lens_fit_residual_map,
            should_plot_normalized_residual_map=self.plot_lens_fit_normalized_residual_map,
            should_plot_chi_squared_map=self.plot_lens_fit_chi_squared_map,
            should_plot_pixelization_residual_map=self.plot_lens_fit_pixelization_residual_map,
            should_plot_pixelization_normalized_residual_map=self.plot_lens_fit_normalized_residual_map,
            should_plot_pixelization_chi_squared_map=self.plot_lens_fit_pixelization_chi_squared_map,
            should_plot_pixelization_regularization_weights=self.plot_lens_fit_pixelization_regularization_weights,
            should_plot_subtracted_images_of_planes=self.plot_lens_fit_subtracted_images_of_planes,
            should_plot_model_images_of_planes=self.plot_lens_fit_model_images_of_planes,
            should_plot_plane_images_of_planes=self.plot_lens_fit_plane_images_of_planes,
            units=self.plot_units,
            visualize_path=image_path,
            subplot_path=subplot_path,
        )

    def initial_plot(self, lens_imaging_data, image_path, last_results):
        mask = lens_imaging_data.mask if self.should_plot_mask else None
        positions = lens_imaging_data.positions if self.should_plot_positions else None

        subplot_path = af.path_util.make_and_return_path_from_path_and_folder_names(
            path=image_path, folder_names=["subplots"]
        )

        phase_plotters.plot_imaging_for_phase(
            imaging_data=lens_imaging_data.imaging_data,
            mask=mask,
            positions=positions,
            extract_array_from_mask=self.extract_array_from_mask,
            zoom_around_mask=self.zoom_around_mask,
            units=self.plot_units,
            should_plot_as_subplot=self.plot_data_as_subplot,
            should_plot_image=self.plot_data_image,
            should_plot_noise_map=self.plot_data_noise_map,
            should_plot_psf=self.plot_data_psf,
            should_plot_signal_to_noise_map=self.plot_data_signal_to_noise_map,
            should_plot_absolute_signal_to_noise_map=self.plot_data_absolute_signal_to_noise_map,
            should_plot_potential_chi_squared_map=self.plot_data_potential_chi_squared_map,
            visualize_path=image_path,
            subplot_path=subplot_path,
        )

        if last_results is not None:
            if mask is not None:
                phase_plotters.plot_hyper_images_for_phase(
                    hyper_model_image_2d=mask.mapping.scaled_array_2d_from_array_1d(
                        array_1d=last_results.hyper_model_image_1d
                    ),
                    hyper_galaxy_image_2d_path_dict=last_results.hyper_galaxy_image_2d_path_dict,
                    binned_hyper_galaxy_image_2d_path_dict=last_results.binned_hyper_galaxy_image_2d_path_dict_from_binned_grid(
                        binned_grid=lens_imaging_data.grid.binned
                    ),
                    mask=lens_imaging_data.mask,
                    binned_grid=lens_imaging_data.grid.binned,
                    extract_array_from_mask=self.extract_array_from_mask,
                    zoom_around_mask=self.zoom_around_mask,
                    units=self.plot_units,
                    should_plot_hyper_model_image=self.plot_hyper_model_image,
                    should_plot_hyper_galaxy_images=self.plot_hyper_galaxy_images,
                    should_plot_binned_hyper_galaxy_images=self.plot_binned_hyper_galaxy_images,
                    visualize_path=image_path,
                )
