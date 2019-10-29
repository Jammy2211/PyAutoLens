import os

import autofit as af
from autoastro.plotters import galaxy_fit_plotters
from autolens.plotters import phase_plotters, hyper_plotters


def setting(section, name):
    return af.conf.instance.visualize.get(section, name, bool)


def plot_setting(name):
    return setting("plots", name)


def figure_setting(name):
    return setting("figures", name)


class AbstractVisualizer:
    def __init__(self, image_path):
        self.image_path = image_path or ""
        try:
            os.makedirs(self.image_path)
        except (FileExistsError, FileNotFoundError):
            pass
        self.plot_units = af.conf.instance.visualize.get(
            "figures", "plot_units", str
        ).strip()
        self.should_plot_mask_overlay = figure_setting("plot_mask_on_images")
        self.plot_ray_tracing_all_at_end_png = plot_setting(
            "plot_ray_tracing_all_at_end_png"
        )
        self.plot_ray_tracing_all_at_end_fits = plot_setting(
            "plot_ray_tracing_all_at_end_fits"
        )


class SubPlotVisualizer(AbstractVisualizer):
    def __init__(self, image_path):
        super().__init__(image_path)
        self.subplot_path = f"{image_path}subplots/"
        try:
            os.makedirs(self.subplot_path)
        except FileExistsError:
            pass


class PhaseGalaxyVisualizer(AbstractVisualizer):
    def __init__(self, image_path):
        super().__init__(image_path)
        self.plot_galaxy_fit_all_at_end_png = plot_setting(
            "plot_galaxy_fit_all_at_end_png"
        )
        self.plot_galaxy_fit_all_at_end_fits = plot_setting(
            "plot_galaxy_fit_all_at_end_fits"
        )
        self.plot_galaxy_fit_as_subplot = plot_setting("plot_galaxy_fit_as_subplot")
        self.plot_galaxy_fit_image = plot_setting("plot_galaxy_fit_image")
        self.plot_galaxy_fit_noise_map = plot_setting("plot_galaxy_fit_noise_map")
        self.plot_galaxy_fit_model_image = plot_setting("plot_galaxy_fit_model_image")
        self.plot_galaxy_fit_residual_map = plot_setting("plot_galaxy_fit_residual_map")
        self.plot_galaxy_fit_chi_squared_map = plot_setting(
            "plot_galaxy_fit_chi_squared_map"
        )

    def plot_galaxy_fit_subplot(self, fit, path_suffix=""):
        if self.plot_galaxy_fit_as_subplot:
            galaxy_fit_plotters.subplot(
                fit=fit,
                should_plot_mask_overlay=self.should_plot_mask_overlay,
                units=self.plot_units,
                output_path=f"{self.image_path}/{path_suffix}",
                output_format="png",
            )

    def plot_fit_individuals(
        self, fit, plot_all=False, image_format="png", path_suffix=""
    ):
        if plot_all:
            should_plot_image = True
            should_plot_noise_map = True
            should_plot_model_image = True
            should_plot_residual_map = True
            should_plot_chi_squared_map = True
        else:
            should_plot_image = self.plot_galaxy_fit_image
            should_plot_noise_map = self.plot_galaxy_fit_noise_map
            should_plot_model_image = self.plot_galaxy_fit_model_image
            should_plot_residual_map = self.plot_galaxy_fit_residual_map
            should_plot_chi_squared_map = self.plot_galaxy_fit_chi_squared_map
        galaxy_fit_plotters.individuals(
            fit=fit,
            should_plot_mask_overlay=self.should_plot_mask_overlay,
            should_plot_image=should_plot_image,
            should_plot_noise_map=should_plot_noise_map,
            should_plot_model_image=should_plot_model_image,
            should_plot_residual_map=should_plot_residual_map,
            should_plot_chi_squared_map=should_plot_chi_squared_map,
            units=self.plot_units,
            output_path=f"{self.image_path}/{path_suffix}",
            output_format=image_format,
        )


class PhaseImagingVisualizer(SubPlotVisualizer):
    def __init__(self, masked_imaging, image_path):
        super().__init__(image_path)
        self.masked_imaging = masked_imaging

        self.should_plot_image_plane_pix = figure_setting(
            "plot_image_plane_adaptive_pixelization_grid"
        )
        self.plot_data_as_subplot = plot_setting("plot_data_as_subplot")
        self.plot_data_image = plot_setting("plot_data_image")
        self.plot_data_noise_map = plot_setting("plot_data_noise_map")
        self.plot_data_psf = plot_setting("plot_data_psf")

        self.plot_data_signal_to_noise_map = plot_setting(
            "plot_data_signal_to_noise_map"
        )
        self.plot_data_absolute_signal_to_noise_map = plot_setting(
            "plot_data_absolute_signal_to_noise_map"
        )
        self.plot_data_potential_chi_squared_map = plot_setting(
            "plot_data_potential_chi_squared_map"
        )
        self.plot_fit_all_at_end_png = plot_setting("plot_fit_all_at_end_png")
        self.plot_fit_all_at_end_fits = plot_setting(
            "plot_fit_all_at_end_fits"
        )
        self.plot_fit_as_subplot = plot_setting("plot_fit_as_subplot")
        self.plot_fit_of_planes_as_subplot = plot_setting(
            "plot_fit_of_planes_as_subplot"
        )
        self.plot_fit_inversion_as_subplot = plot_setting(
            "plot_fit_inversion_as_subplot"
        )
        self.plot_fit_image = plot_setting("plot_fit_image")
        self.plot_fit_noise_map = plot_setting("plot_fit_noise_map")
        self.plot_fit_signal_to_noise_map = plot_setting(
            "plot_fit_signal_to_noise_map"
        )
        self.plot_fit_model_image = plot_setting("plot_fit_model_image")
        self.plot_fit_residual_map = plot_setting("plot_fit_residual_map")
        self.plot_fit_normalized_residual_map = plot_setting(
            "plot_fit_normalized_residual_map"
        )
        self.plot_fit_chi_squared_map = plot_setting(
            "plot_fit_chi_squared_map"
        )
        self.plot_fit_contribution_maps = plot_setting(
            "plot_fit_contribution_maps"
        )
        self.plot_fit_inversion_residual_map = plot_setting(
            "plot_fit_inversion_residual_map"
        )
        self.plot_fit_pixelization_normalized_residuals = plot_setting(
            "plot_fit_inversion_normalized_residual_map"
        )
        self.plot_fit_inversion_chi_squared_map = plot_setting(
            "plot_fit_inversion_chi_squared_map"
        )
        self.plot_fit_inversion_regularization_weights = plot_setting(
            "plot_fit_inversion_regularization_weight_map"
        )
        self.plot_fit_subtracted_images_of_planes = plot_setting(
            "plot_fit_subtracted_images_of_planes"
        )
        self.plot_fit_model_images_of_planes = plot_setting(
            "plot_fit_model_images_of_planes"
        )
        self.plot_fit_plane_images_of_planes = plot_setting(
            "plot_fit_plane_images_of_planes"
        )
        self.plot_hyper_model_image = plot_setting("plot_hyper_model_image")
        self.plot_hyper_galaxy_images = plot_setting("plot_hyper_galaxy_images")

        self.should_plot_positions = figure_setting("plot_positions_on_images")

        self.plot_ray_tracing_as_subplot = plot_setting("plot_ray_tracing_as_subplot")
        self.plot_ray_tracing_profile_image = plot_setting(
            "plot_ray_tracing_profile_image"
        )
        self.plot_ray_tracing_source_plane = plot_setting(
            "plot_ray_tracing_source_plane_image"
        )
        self.plot_ray_tracing_convergence = plot_setting("plot_ray_tracing_convergence")
        self.plot_ray_tracing_potential = plot_setting("plot_ray_tracing_potential")
        self.plot_ray_tracing_deflections = plot_setting("plot_ray_tracing_deflections")
        self.plot_ray_tracing_magnification = plot_setting(
            "plot_ray_tracing_magnification"
        )

    def plot_ray_tracing(self, tracer, during_analysis):
        positions = (
            self.masked_imaging.positions if self.should_plot_positions else None
        )
        mask = self.masked_imaging.mask if self.should_plot_mask_overlay else None
        phase_plotters.ray_tracing_of_phase(
            tracer=tracer,
            grid=self.masked_imaging.grid,
            during_analysis=during_analysis,
            mask_overlay=mask,
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

    def plot_masked_imaging(self, fit, during_analysis):
        positions = (
            self.masked_imaging.positions if self.should_plot_positions else None
        )
        phase_plotters.imaging_fit_of_phase(
            fit=fit,
            during_analysis=during_analysis,
            should_plot_mask_overlay=self.should_plot_mask_overlay,
            positions=positions,
            should_plot_image_plane_pix=self.should_plot_image_plane_pix,
            should_plot_all_at_end_png=self.plot_fit_all_at_end_png,
            should_plot_all_at_end_fits=self.plot_fit_all_at_end_fits,
            should_plot_fit_as_subplot=self.plot_fit_as_subplot,
            should_plot_fit_of_planes_as_subplot=self.plot_fit_of_planes_as_subplot,
            should_plot_inversion_as_subplot=self.plot_fit_inversion_as_subplot,
            should_plot_image=self.plot_fit_image,
            should_plot_noise_map=self.plot_fit_noise_map,
            should_plot_signal_to_noise_map=self.plot_fit_signal_to_noise_map,
            should_plot_model_image=self.plot_fit_model_image,
            should_plot_residual_map=self.plot_fit_residual_map,
            should_plot_normalized_residual_map=self.plot_fit_normalized_residual_map,
            should_plot_chi_squared_map=self.plot_fit_chi_squared_map,
            should_plot_inversion_residual_map=self.plot_fit_inversion_residual_map,
            should_plot_inversion_normalized_residual_map=self.plot_fit_normalized_residual_map,
            should_plot_inversion_chi_squared_map=self.plot_fit_inversion_chi_squared_map,
            should_plot_inversion_regularization_weights=(
                self.plot_fit_inversion_regularization_weights
            ),
            should_plot_subtracted_images_of_planes=self.plot_fit_subtracted_images_of_planes,
            should_plot_model_images_of_planes=self.plot_fit_model_images_of_planes,
            should_plot_plane_images_of_planes=self.plot_fit_plane_images_of_planes,
            units=self.plot_units,
            visualize_path=self.image_path,
            subplot_path=self.subplot_path,
        )

    def plot_imaging(self):
        mask = self.masked_imaging.mask if self.should_plot_mask_overlay else None
        positions = (
            self.masked_imaging.positions if self.should_plot_positions else None
        )

        phase_plotters.imaging_of_phase(
            imaging=self.masked_imaging.imaging,
            mask_overlay=mask,
            positions=positions,
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

    def plot_hyper_images(self, last_results):
        mask = self.masked_imaging.mask
        if (
            self.should_plot_mask_overlay
            and mask is not None
            and last_results is not None
        ):
            phase_plotters.plot_hyper_images_for_phase(
                hyper_model_image=last_results.hyper_model_image,
                hyper_galaxy_image_path_dict=last_results.hyper_galaxy_image_path_dict,
                mask_overlay=self.masked_imaging.mask,
                units=self.plot_units,
                should_plot_hyper_model_image=self.plot_hyper_model_image,
                should_plot_hyper_galaxy_images=self.plot_hyper_galaxy_images,
                visualize_path=self.image_path,
            )


class HyperGalaxyVisualizer(SubPlotVisualizer):
    def __init__(self, image_path):
        super().__init__(image_path)
        self.plot_hyper_galaxy_subplot = plot_setting("plot_hyper_galaxy_subplot")

    def hyper_galaxy_subplot(
        self,
        hyper_galaxy_image,
        contribution_map,
        noise_map,
        hyper_noise_map,
        chi_squared_map,
        hyper_chi_squared_map,
    ):
        hyper_plotters.subplot_of_hyper_galaxy(
            hyper_galaxy_image_sub=hyper_galaxy_image,
            contribution_map_sub=contribution_map,
            noise_map_sub=noise_map,
            hyper_noise_map_sub=hyper_noise_map,
            chi_squared_map_sub=chi_squared_map,
            hyper_chi_squared_map_sub=hyper_chi_squared_map,
            output_path=self.subplot_path,
            output_format="png",
        )
