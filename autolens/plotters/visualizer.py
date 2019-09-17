import autofit as af
from autolens.pipeline.plotters import phase_plotters


class Visualizer:
    def __init__(
            self,
            lens_imaging_data,
            image_path
    ):
        self.lens_imaging_data = lens_imaging_data
        self.image_path = image_path
        self.subplot_path = af.path_util.make_and_return_path_from_path_and_folder_names(
            path=image_path, folder_names=["subplots"]
        )

        def setting(section, name):
            return af.conf.instance.visualize.get(
                section, name, bool
            )

        def plot_setting(name):
            return setting("plots", name)

        def figure_setting(name):
            return setting("figures", name)

        self.should_plot_image_plane_pix = figure_setting("plot_image_plane_adaptive_pixelization_grid")
        self.plot_data_as_subplot = plot_setting("plot_data_as_subplot")
        self.plot_data_image = plot_setting("plot_data_image")
        self.plot_data_noise_map = plot_setting("plot_data_noise_map")
        self.plot_data_psf = plot_setting("plot_data_psf")

        self.plot_data_signal_to_noise_map = plot_setting("plot_data_signal_to_noise_map")
        self.plot_data_absolute_signal_to_noise_map = plot_setting("plot_data_absolute_signal_to_noise_map")
        self.plot_data_potential_chi_squared_map = plot_setting("plot_data_potential_chi_squared_map")
        self.plot_lens_fit_all_at_end_png = plot_setting("plot_lens_fit_all_at_end_png")
        self.plot_lens_fit_all_at_end_fits = plot_setting("plot_lens_fit_all_at_end_fits")
        self.plot_lens_fit_as_subplot = plot_setting("plot_lens_fit_as_subplot")
        self.plot_lens_fit_of_planes_as_subplot = plot_setting("plot_lens_fit_of_planes_as_subplot")
        self.plot_lens_fit_inversion_as_subplot = plot_setting("plot_lens_fit_inversion_as_subplot")
        self.plot_lens_fit_image = plot_setting("plot_lens_fit_image")
        self.plot_lens_fit_noise_map = plot_setting("plot_lens_fit_noise_map")
        self.plot_lens_fit_signal_to_noise_map = plot_setting("plot_lens_fit_signal_to_noise_map")
        self.plot_lens_fit_model_image = plot_setting("plot_lens_fit_model_image")
        self.plot_lens_fit_residual_map = plot_setting("plot_lens_fit_residual_map")
        self.plot_lens_fit_normalized_residual_map = plot_setting("plot_lens_fit_normalized_residual_map")
        self.plot_lens_fit_chi_squared_map = plot_setting("plot_lens_fit_chi_squared_map")
        self.plot_lens_fit_contribution_maps = plot_setting("plot_lens_fit_contribution_maps")
        self.plot_lens_fit_pixelization_residual_map = plot_setting("plot_lens_fit_pixelization_residual_map")
        self.plot_lens_fit_pixelization_normalized_residuals = plot_setting(
            "plot_lens_fit_pixelization_normalized_residual_map")
        self.plot_lens_fit_pixelization_chi_squared_map = plot_setting(
            "plot_lens_fit_pixelization_chi_squared_map")
        self.plot_lens_fit_pixelization_regularization_weights = plot_setting(
            "plot_lens_fit_pixelization_regularization_weight_map")
        self.plot_lens_fit_subtracted_images_of_planes = plot_setting(
            "plot_lens_fit_subtracted_images_of_planes")
        self.plot_lens_fit_model_images_of_planes = plot_setting("plot_lens_fit_model_images_of_planes")
        self.plot_lens_fit_plane_images_of_planes = plot_setting("plot_lens_fit_plane_images_of_planes")
        self.plot_hyper_model_image = plot_setting("plot_hyper_model_image")
        self.plot_hyper_galaxy_images = plot_setting("plot_hyper_galaxy_images")
        self.plot_binned_hyper_galaxy_images = plot_setting("plot_binned_hyper_galaxy_images")
        self.should_plot_mask = figure_setting("plot_mask_on_images")
        self.extract_array_from_mask = figure_setting("extract_images_from_mask")
        self.zoom_around_mask = figure_setting("zoom_around_mask_of_images")
        self.should_plot_positions = figure_setting("plot_positions_on_images")
        self.plot_units = af.conf.instance.visualize.get(
            "figures", "plot_units", str
        ).strip()

        self.plot_ray_tracing_all_at_end_png = plot_setting("plot_ray_tracing_all_at_end_png")
        self.plot_ray_tracing_all_at_end_fits = plot_setting("plot_ray_tracing_all_at_end_fits")
        self.plot_ray_tracing_as_subplot = plot_setting("plot_ray_tracing_as_subplot")
        self.plot_ray_tracing_profile_image = plot_setting("plot_ray_tracing_profile_image")
        self.plot_ray_tracing_source_plane = plot_setting("plot_ray_tracing_source_plane_image")
        self.plot_ray_tracing_convergence = plot_setting("plot_ray_tracing_convergence")
        self.plot_ray_tracing_potential = plot_setting("plot_ray_tracing_potential")
        self.plot_ray_tracing_deflections = plot_setting("plot_ray_tracing_deflections")
        self.plot_ray_tracing_magnification = plot_setting("plot_ray_tracing_magnification")

    def plot_ray_tracing(
            self,
            tracer,
            during_analysis,
    ):
        positions = self.lens_imaging_data.positions if self.should_plot_positions else None
        mask = self.lens_imaging_data.mask if self.should_plot_mask else None
        phase_plotters.plot_ray_tracing_for_phase(
            tracer=tracer,
            grid=self.lens_imaging_data.grid,
            during_analysis=during_analysis,
            mask=mask,
            extract_array_from_mask=self.extract_array_from_mask,
            zoom_around_mask=self.zoom_around_mask,
            positions=positions,
            units=self.plot_units,
            should_plot_as_subplot=self.plot_ray_tracing_as_subplot,
            should_plot_all_at_end_png=self.plot_ray_tracing_all_at_end_png,
            should_plot_all_at_end_fits=self.plot_ray_tracing_all_at_end_fits,
            should_plot_image=self.plot_ray_tracing_profile_image,
            should_plot_source_plane=self.plot_ray_tracing_source_plane,
            should_plot_convergence=self.plot_ray_tracing_convergence,
            should_plot_potential=self.plot_ray_tracing_potential,
            should_plot_deflections=self.plot_ray_tracing_deflections,
            visualize_path=self.image_path,
            subplot_path=self.subplot_path,
        )

    def plot_lens_imaging(
            self,
            fit,
            during_analysis
    ):
        positions = self.lens_imaging_data.positions if self.should_plot_positions else None
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
            should_plot_pixelization_regularization_weights=(
                self.plot_lens_fit_pixelization_regularization_weights
            ),
            should_plot_subtracted_images_of_planes=self.plot_lens_fit_subtracted_images_of_planes,
            should_plot_model_images_of_planes=self.plot_lens_fit_model_images_of_planes,
            should_plot_plane_images_of_planes=self.plot_lens_fit_plane_images_of_planes,
            units=self.plot_units,
            visualize_path=self.image_path,
            subplot_path=self.subplot_path,
        )

    def plot_imaging(self):
        mask = self.lens_imaging_data.mask if self.should_plot_mask else None
        positions = self.lens_imaging_data.positions if self.should_plot_positions else None

        phase_plotters.plot_imaging_for_phase(
            imaging_data=self.lens_imaging_data.imaging_data,
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
            visualize_path=self.image_path,
            subplot_path=self.subplot_path,
        )

    def plot_hyper_images(
            self,
            last_results
    ):
        mask = self.lens_imaging_data.mask
        if self.should_plot_mask and mask is not None and last_results is not None:
            phase_plotters.plot_hyper_images_for_phase(
                hyper_model_image_2d=mask.mapping.scaled_array_2d_from_array_1d(
                    array_1d=last_results.hyper_model_image_1d
                ),
                hyper_galaxy_image_2d_path_dict=last_results.hyper_galaxy_image_2d_path_dict,
                binned_hyper_galaxy_image_2d_path_dict=last_results.binned_hyper_galaxy_image_2d_path_dict(
                    binned_grid=self.lens_imaging_data.grid.binned
                ),
                mask=self.lens_imaging_data.mask,
                binned_grid=self.lens_imaging_data.grid.binned,
                extract_array_from_mask=self.extract_array_from_mask,
                zoom_around_mask=self.zoom_around_mask,
                units=self.plot_units,
                should_plot_hyper_model_image=self.plot_hyper_model_image,
                should_plot_hyper_galaxy_images=self.plot_hyper_galaxy_images,
                should_plot_binned_hyper_galaxy_images=self.plot_binned_hyper_galaxy_images,
                visualize_path=self.image_path,
            )
