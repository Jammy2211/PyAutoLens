import os

import autofit as af
from autoastro.plotters import fit_galaxy_plotters
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
        self.plot_in_kpc = af.conf.instance.visualize.get(
            "figures", "plot_in_kpc", bool
        )
        self.include_mask = figure_setting("include_mask")
        self.include_critical_curves = figure_setting("include_critical_curves")
        self.include_caustics = figure_setting("include_caustics")
        self.plot_ray_tracing_all_at_end_png = plot_setting(
            "plot_ray_tracing_all_at_end_png"
        )
        self.plot_ray_tracing_all_at_end_fits = plot_setting(
            "plot_ray_tracing_all_at_end_fits"
        )
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
            fit_galaxy_plotters.subplot(
                fit=fit,
                include_mask=self.include_mask,
                output_path=f"{self.image_path}/{path_suffix}",
                output_format="png",
            )

    def plot_fit_individuals(
        self, fit, plot_all=False, image_format="png", path_suffix=""
    ):
        if plot_all:
            plot_image = True
            plot_noise_map = True
            plot_model_image = True
            plot_residual_map = True
            plot_chi_squared_map = True
        else:
            plot_image = self.plot_galaxy_fit_image
            plot_noise_map = self.plot_galaxy_fit_noise_map
            plot_model_image = self.plot_galaxy_fit_model_image
            plot_residual_map = self.plot_galaxy_fit_residual_map
            plot_chi_squared_map = self.plot_galaxy_fit_chi_squared_map
        fit_galaxy_plotters.individuals(
            fit=fit,
            include_mask=self.include_mask,
            plot_image=plot_image,
            plot_noise_map=plot_noise_map,
            plot_model_image=plot_model_image,
            plot_residual_map=plot_residual_map,
            plot_chi_squared_map=plot_chi_squared_map,
            output_path=f"{self.image_path}/{path_suffix}",
            output_format=image_format,
        )


class PhaseDatasetVisualize(SubPlotVisualizer):
    def __init__(self, masked_dataset, image_path):
        super().__init__(image_path)
        self.masked_dataset = masked_dataset

        self.include_image_plane_pix = figure_setting(
            "include_image_plane_pixelization_grid"
        )
        self.include_positions = figure_setting("include_positions")

        self.plot_dataset_as_subplot = plot_setting("plot_dataset_as_subplot")
        self.plot_dataset_data = plot_setting("plot_dataset_data")
        self.plot_dataset_noise_map = plot_setting("plot_dataset_noise_map")
        self.plot_dataset_psf = plot_setting("plot_dataset_psf")

        self.plot_dataset_signal_to_noise_map = plot_setting(
            "plot_dataset_signal_to_noise_map"
        )
        self.plot_dataset_absolute_signal_to_noise_map = plot_setting(
            "plot_dataset_absolute_signal_to_noise_map"
        )
        self.plot_dataset_potential_chi_squared_map = plot_setting(
            "plot_dataset_potential_chi_squared_map"
        )
        self.plot_fit_all_at_end_png = plot_setting("plot_fit_all_at_end_png")
        self.plot_fit_all_at_end_fits = plot_setting("plot_fit_all_at_end_fits")
        self.plot_fit_as_subplot = plot_setting("plot_fit_as_subplot")
        self.plot_fit_of_planes_as_subplot = plot_setting(
            "plot_fit_of_planes_as_subplot"
        )
        self.plot_fit_inversion_as_subplot = plot_setting(
            "plot_fit_inversion_as_subplot"
        )
        self.plot_fit_data = plot_setting("plot_fit_data")
        self.plot_fit_noise_map = plot_setting("plot_fit_noise_map")
        self.plot_fit_signal_to_noise_map = plot_setting("plot_fit_signal_to_noise_map")
        self.plot_fit_model_data = plot_setting("plot_fit_model_data")
        self.plot_fit_residual_map = plot_setting("plot_fit_residual_map")
        self.plot_fit_normalized_residual_map = plot_setting(
            "plot_fit_normalized_residual_map"
        )
        self.plot_fit_chi_squared_map = plot_setting("plot_fit_chi_squared_map")
        self.plot_fit_contribution_maps = plot_setting("plot_fit_contribution_maps")

        self.plot_fit_inversion_reconstruction = plot_setting(
            "plot_fit_inversion_reconstruction"
        )

        self.plot_fit_inversion_errors = plot_setting("plot_fit_inversion_errors")

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
        self.plot_fit_inversion_interpolated_reconstruction = plot_setting(
            "plot_fit_inversion_interpolated_reconstruction"
        )
        self.plot_fit_inversion_interpolated_errors = plot_setting(
            "plot_fit_inversion_interpolated_errors"
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

    def plot_ray_tracing(self, tracer, during_analysis):
        positions = self.masked_dataset.positions if self.include_positions else None
        mask = self.masked_dataset.mask if self.include_mask else None
        phase_plotters.ray_tracing_of_phase(
            tracer=tracer,
            grid=self.masked_dataset.grid,
            during_analysis=during_analysis,
            mask=mask,
            include_critical_curves=self.include_critical_curves,
            include_caustics=self.include_caustics,
            positions=positions,
            plot_in_kpc=self.plot_in_kpc,
            plot_as_subplot=self.plot_ray_tracing_as_subplot,
            plot_all_at_end_png=self.plot_ray_tracing_all_at_end_png,
            plot_all_at_end_fits=self.plot_ray_tracing_all_at_end_fits,
            plot_image=self.plot_ray_tracing_profile_image,
            plot_source_plane=self.plot_ray_tracing_source_plane,
            plot_convergence=self.plot_ray_tracing_convergence,
            plot_potential=self.plot_ray_tracing_potential,
            plot_deflections=self.plot_ray_tracing_deflections,
            visualize_path=self.image_path,
            subplot_path=self.subplot_path,
        )


class PhaseImagingVisualizer(PhaseDatasetVisualize):
    def __init__(self, masked_dataset, image_path):
        super(PhaseImagingVisualizer, self).__init__(
            masked_dataset=masked_dataset, image_path=image_path
        )

        self.plot_dataset_psf = plot_setting("plot_dataset_psf")

        self.plot_hyper_model_image = plot_setting("plot_hyper_model_image")
        self.plot_hyper_galaxy_images = plot_setting("plot_hyper_galaxy_images")

        self.plot_imaging()

    @property
    def masked_imaging(self):
        return self.masked_dataset

    def plot_imaging(self):
        mask = self.masked_dataset.mask if self.include_mask else None
        positions = self.masked_dataset.positions if self.include_positions else None

        phase_plotters.imaging_of_phase(
            imaging=self.masked_dataset.imaging,
            mask=mask,
            positions=positions,
            unit_label="arcsec",
            kpc_per_arcsec=None,
            plot_as_subplot=self.plot_dataset_as_subplot,
            plot_image=self.plot_dataset_data,
            plot_noise_map=self.plot_dataset_noise_map,
            plot_psf=self.plot_dataset_psf,
            plot_signal_to_noise_map=self.plot_dataset_signal_to_noise_map,
            plot_absolute_signal_to_noise_map=self.plot_dataset_absolute_signal_to_noise_map,
            plot_potential_chi_squared_map=self.plot_dataset_potential_chi_squared_map,
            visualize_path=self.image_path,
            subplot_path=self.subplot_path,
        )

    def plot_fit(self, fit, during_analysis):

        phase_plotters.imaging_fit_of_phase(
            fit=fit,
            during_analysis=during_analysis,
            include_mask=self.include_mask,
            include_critical_curves=self.include_critical_curves,
            include_caustics=self.include_caustics,
            include_positions=self.include_positions,
            plot_in_kpc=self.plot_in_kpc,
            include_image_plane_pix=self.include_image_plane_pix,
            plot_all_at_end_png=self.plot_fit_all_at_end_png,
            plot_all_at_end_fits=self.plot_fit_all_at_end_fits,
            plot_fit_as_subplot=self.plot_fit_as_subplot,
            plot_fit_of_planes_as_subplot=self.plot_fit_of_planes_as_subplot,
            plot_inversion_as_subplot=self.plot_fit_inversion_as_subplot,
            plot_image=self.plot_fit_data,
            plot_noise_map=self.plot_fit_noise_map,
            plot_signal_to_noise_map=self.plot_fit_signal_to_noise_map,
            plot_model_image=self.plot_fit_model_data,
            plot_residual_map=self.plot_fit_residual_map,
            plot_normalized_residual_map=self.plot_fit_normalized_residual_map,
            plot_chi_squared_map=self.plot_fit_chi_squared_map,
            plot_inversion_residual_map=self.plot_fit_inversion_residual_map,
            plot_inversion_reconstruction=self.plot_fit_inversion_reconstruction,
            plot_inversion_errors=self.plot_fit_inversion_errors,
            plot_inversion_normalized_residual_map=self.plot_fit_normalized_residual_map,
            plot_inversion_chi_squared_map=self.plot_fit_inversion_chi_squared_map,
            plot_inversion_regularization_weights=(
                self.plot_fit_inversion_regularization_weights
            ),
            plot_inversion_interpolated_reconstruction=self.plot_fit_inversion_interpolated_reconstruction,
            plot_inversion_interpolated_errors=self.plot_fit_inversion_interpolated_errors,
            plot_subtracted_images_of_planes=self.plot_fit_subtracted_images_of_planes,
            plot_model_images_of_planes=self.plot_fit_model_images_of_planes,
            plot_plane_images_of_planes=self.plot_fit_plane_images_of_planes,
            visualize_path=self.image_path,
            subplot_path=self.subplot_path,
        )

    def plot_hyper_images(self, last_results):
        mask = self.masked_dataset.mask
        if self.include_mask and mask is not None and last_results is not None:
            phase_plotters.plot_hyper_images_for_phase(
                hyper_model_image=last_results.hyper_model_image,
                hyper_galaxy_image_path_dict=last_results.hyper_galaxy_image_path_dict,
                mask=self.masked_dataset.mask,
                unit_label="arcsec",
                kpc_per_arcsec=None,
                plot_hyper_model_image=self.plot_hyper_model_image,
                plot_hyper_galaxy_images=self.plot_hyper_galaxy_images,
                visualize_path=self.image_path,
            )


class PhaseInterferometerVisualizer(PhaseDatasetVisualize):
    def __init__(self, masked_dataset, image_path):
        super(PhaseInterferometerVisualizer, self).__init__(
            masked_dataset=masked_dataset, image_path=image_path
        )

        self.plot_dataset_uv_wavelengths = plot_setting("plot_dataset_uv_wavelengths")
        self.plot_dataset_primary_beam = plot_setting("plot_dataset_primary_beam")

        self.plot_hyper_model_image = plot_setting("plot_hyper_model_image")
        self.plot_hyper_galaxy_images = plot_setting("plot_hyper_galaxy_images")

        self.plot_interferometer()

    @property
    def masked_interferometer(self):
        return self.masked_dataset

    def plot_interferometer(self):

        phase_plotters.interferometer_of_phase(
            interferometer=self.masked_interferometer.interferometer,
            unit_label="arcsec",
            kpc_per_arcsec=None,
            plot_as_subplot=self.plot_dataset_as_subplot,
            plot_visibilities=self.plot_dataset_data,
            plot_uv_wavelengths=self.plot_dataset_uv_wavelengths,
            plot_primary_beam=self.plot_dataset_primary_beam,
            visualize_path=self.image_path,
            subplot_path=self.subplot_path,
        )

    def plot_fit(self, fit, during_analysis):

        phase_plotters.interferometer_fit_of_phase(
            fit=fit,
            during_analysis=during_analysis,
            include_positions=self.include_positions,
            plot_in_kpc=self.plot_in_kpc,
            include_mask=self.include_mask,
            include_critical_curves=self.include_critical_curves,
            include_caustics=self.include_caustics,
            include_image_plane_pix=self.include_image_plane_pix,
            plot_all_at_end_png=self.plot_fit_all_at_end_png,
            plot_all_at_end_fits=self.plot_fit_all_at_end_fits,
            plot_fit_as_subplot=self.plot_fit_as_subplot,
            plot_inversion_as_subplot=self.plot_fit_inversion_as_subplot,
            plot_visibilities=self.plot_fit_data,
            plot_noise_map=self.plot_fit_noise_map,
            plot_signal_to_noise_map=self.plot_fit_signal_to_noise_map,
            plot_model_visibilities=self.plot_fit_model_data,
            plot_residual_map=self.plot_fit_residual_map,
            plot_normalized_residual_map=self.plot_fit_normalized_residual_map,
            plot_chi_squared_map=self.plot_fit_chi_squared_map,
            plot_inversion_reconstruction=self.plot_fit_inversion_reconstruction,
            plot_inversion_errors=self.plot_fit_inversion_errors,
            plot_inversion_residual_map=self.plot_fit_inversion_residual_map,
            plot_inversion_normalized_residual_map=self.plot_fit_normalized_residual_map,
            plot_inversion_chi_squared_map=self.plot_fit_inversion_chi_squared_map,
            plot_inversion_regularization_weights=(
                self.plot_fit_inversion_regularization_weights
            ),
            plot_inversion_interpolated_reconstruction=self.plot_fit_inversion_interpolated_reconstruction,
            plot_inversion_interpolated_errors=self.plot_fit_inversion_interpolated_errors,
            visualize_path=self.image_path,
            subplot_path=self.subplot_path,
        )

    def plot_hyper_images(self, last_results):
        mask = self.masked_dataset.mask
        if self.include_mask and mask is not None and last_results is not None:
            phase_plotters.plot_hyper_images_for_phase(
                hyper_model_image=last_results.hyper_model_image,
                hyper_galaxy_image_path_dict=last_results.hyper_galaxy_image_path_dict,
                mask=self.masked_interferometer.mask,
                plot_hyper_model_image=self.plot_hyper_model_image,
                plot_hyper_galaxy_images=self.plot_hyper_galaxy_images,
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
