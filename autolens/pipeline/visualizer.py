import autoarray as aa
import autofit as af
from autoarray.plotters import plotters, mat_objs
from autoastro.plots import lensing_plotters
from autoastro.plots import fit_galaxy_plots
from autolens.plots import ray_tracing_plots, hyper_plots
from autolens.plots.fit_imaging_plots import fit_imaging_plots
from autolens.plots.fit_interferometer_plots import fit_interferometer_plots


def setting(section, name):
    return af.conf.instance.visualize_plots.get(section, name, bool)


def plot_setting(name):
    return setting("plots", name)


class AbstractVisualizer:
    def __init__(self, image_path):

        self.plotter = plotters.Plotter(output=mat_objs.Output(path=image_path, format="png"))
        self.sub_plotter = plotters.SubPlotter(output=mat_objs.Output(path=image_path+"subplots/", format="png"))
        self.include = lensing_plotters.Include()

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

    def plot_galaxy_fit_subplot(self, fit):
        if self.plot_galaxy_fit_as_subplot:
            fit_galaxy_plots.subplot_fit_galaxy(
                fit=fit,
                include=self.include,
                sub_plotter=self.sub_plotter
            )

    def plot_fit_individuals(
        self, fit,
    ):

        fit_galaxy_plots.individuals(
            fit=fit,
            plot_image=self.plot_galaxy_fit_image,
            plot_noise_map=self.plot_galaxy_fit_noise_map,
            plot_model_image=self.plot_galaxy_fit_model_image,
            plot_residual_map=self.plot_galaxy_fit_residual_map,
            plot_chi_squared_map=self.plot_galaxy_fit_chi_squared_map,
            include=self.include,
            plotter=self.plotter
        )


class PhaseDatasetVisualizer(AbstractVisualizer):
    def __init__(self, masked_dataset, image_path):
        super().__init__(image_path)
        self.masked_dataset = masked_dataset

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

    def visualize_ray_tracing(self, tracer, during_analysis):
        positions = self.masked_dataset.positions if self.include.positions else None
        mask = self.masked_dataset.mask if self.include.mask else None

        plotter = self.plotter.plotter_with_new_output(output=mat_objs.Output(path=self.plotter.output.path + "ray_tracing/"))

        if self.plot_ray_tracing_as_subplot:

            ray_tracing_plots.subplot_tracer(
                tracer=tracer,
                grid=self.masked_dataset.grid,
                mask=mask,
                positions=positions,
                include=self.include,
                sub_plotter=self.sub_plotter
            )

        ray_tracing_plots.individual(
            tracer=tracer,
            grid=self.masked_dataset.grid,
            mask=mask,
            positions=positions,
            plot_profile_image=self.plot_ray_tracing_profile_image,
            plot_source_plane=self.plot_ray_tracing_source_plane,
            plot_convergence=self.plot_ray_tracing_convergence,
            plot_potential=self.plot_ray_tracing_potential,
            plot_deflections=self.plot_ray_tracing_deflections,
            plot_magnification=self.plot_ray_tracing_magnification,
            include=self.include,
            plotter=plotter
        )

        if not during_analysis:

            if self.plot_ray_tracing_all_at_end_png:

                ray_tracing_plots.individual(
                    tracer=tracer,
                    grid=self.masked_dataset.grid,
                    mask=mask,
                    positions=positions,
                    plot_profile_image=True,
                    plot_source_plane=True,
                    plot_convergence=True,
                    plot_potential=True,
                    plot_deflections=True,
                    include=self.include,
                    plotter=plotter
                )

            if self.plot_ray_tracing_all_at_end_fits:

                fits_plotter = plotter.plotter_with_new_output(
                    output=mat_objs.Output(path=plotter.output.path + "/fits", format="fits")
                )

                ray_tracing_plots.individual(
                    tracer=tracer,
                    grid=self.masked_dataset.grid,
                    mask=mask,
                    positions=positions,
                    plot_profile_image=True,
                    plot_source_plane=True,
                    plot_convergence=True,
                    plot_potential=True,
                    plot_deflections=True,
                    include=self.include,
                    plotter=fits_plotter
                )


class PhaseImagingVisualizer(PhaseDatasetVisualizer):
    def __init__(self, masked_dataset, image_path):
        super(PhaseImagingVisualizer, self).__init__(
            masked_dataset=masked_dataset, image_path=image_path
        )

        self.plot_dataset_psf = plot_setting("plot_dataset_psf")

        self.plot_hyper_model_image = plot_setting("plot_hyper_model_image")
        self.plot_hyper_galaxy_images = plot_setting("plot_hyper_galaxy_images")

        self.visualize_imaging()

    @property
    def masked_imaging(self):
        return self.masked_dataset

    def visualize_imaging(self):

        plotter = self.plotter.plotter_with_new_output(output=mat_objs.Output(path=self.plotter.output.path + "imaging/"))

        mask = self.masked_dataset.mask if self.include.mask else None
        positions = self.masked_dataset.positions if self.include.positions else None

        if self.plot_dataset_as_subplot:
            aa.plot.imaging.subplot_imaging(
                imaging=self.masked_imaging.imaging,
                mask=mask,
                positions=positions,
                include=self.include,
                sub_plotter=self.sub_plotter
            )

        aa.plot.imaging.individual(
            imaging=self.masked_imaging.imaging,
            mask=mask,
            positions=positions,
            plot_image=self.plot_dataset_data,
            plot_noise_map=self.plot_dataset_noise_map,
            plot_psf=self.plot_dataset_psf,
            plot_signal_to_noise_map=self.plot_dataset_signal_to_noise_map,
            plot_absolute_signal_to_noise_map=self.plot_dataset_absolute_signal_to_noise_map,
            plot_potential_chi_squared_map=self.plot_dataset_potential_chi_squared_map,
            include=self.include,
            plotter=plotter
        )

    def visualize_fit(self, fit, during_analysis):

        plotter = self.plotter.plotter_with_new_output(output=mat_objs.Output(path=self.plotter.output.path + "fit_imaging/"))

        if self.plot_fit_as_subplot:
            fit_imaging_plots.subplot_fit_imaging(
                fit=fit,
                include=self.include,
                sub_plotter=self.sub_plotter
            )

        if self.plot_fit_of_planes_as_subplot:
            fit_imaging_plots.subplot_of_planes(
                fit=fit,
                include=self.include,
                sub_plotter=self.sub_plotter
            )

        if self.plot_fit_inversion_as_subplot and fit.inversion is not None:
            aa.plot.inversion.subplot_inversion(
                inversion=fit.inversion,
                mask=fit.mask,
                include=self.include,
                sub_plotter=self.sub_plotter
            )

        fit_imaging_plots.individuals(
            fit=fit,
            plot_image=self.plot_fit_data,
            plot_noise_map=self.plot_fit_noise_map,
            plot_signal_to_noise_map=self.plot_fit_signal_to_noise_map,
            plot_model_image=self.plot_fit_model_data,
            plot_residual_map=self.plot_fit_residual_map,
            plot_chi_squared_map=self.plot_fit_chi_squared_map,
            plot_normalized_residual_map=self.plot_fit_normalized_residual_map,
            plot_inversion_reconstruction=self.plot_fit_inversion_reconstruction,
            plot_inversion_errors=self.plot_fit_inversion_errors,
            plot_inversion_residual_map=self.plot_fit_inversion_residual_map,
            plot_inversion_normalized_residual_map=self.plot_fit_normalized_residual_map,
            plot_inversion_chi_squared_map=self.plot_fit_inversion_chi_squared_map,
            plot_inversion_regularization_weight_map=self.plot_fit_inversion_regularization_weights,
            plot_inversion_interpolated_reconstruction=self.plot_fit_inversion_interpolated_reconstruction,
            plot_inversion_interpolated_errors=self.plot_fit_inversion_interpolated_errors,
            plot_subtracted_images_of_planes=self.plot_fit_subtracted_images_of_planes,
            plot_model_images_of_planes=self.plot_fit_model_images_of_planes,
            plot_plane_images_of_planes=self.plot_fit_plane_images_of_planes,
            include=self.include,
            plotter=plotter
        )

        if not during_analysis:

            if self.plot_fit_all_at_end_png:
                fit_imaging_plots.individuals(
                    fit=fit,
                    plot_image=True,
                    plot_noise_map=True,
                    plot_signal_to_noise_map=True,
                    plot_model_image=True,
                    plot_residual_map=True,
                    plot_normalized_residual_map=True,
                    plot_chi_squared_map=True,
                    plot_inversion_reconstruction=True,
                    plot_inversion_errors=True,
                    plot_inversion_residual_map=True,
                    plot_inversion_normalized_residual_map=True,
                    plot_inversion_chi_squared_map=True,
                    plot_inversion_regularization_weight_map=True,
                    plot_inversion_interpolated_reconstruction=True,
                    plot_inversion_interpolated_errors=True,
                    plot_subtracted_images_of_planes=True,
                    plot_model_images_of_planes=True,
                    plot_plane_images_of_planes=True,
                    include=self.include,
                    plotter=plotter
                )

            if self.plot_fit_all_at_end_fits:
                fits_plotter = plotter.plotter_with_new_output(
                    output=mat_objs.Output(path=plotter.output.path + "/fits", format="fits")
                )

                fit_imaging_plots.individuals(
                    fit=fit,
                    plot_image=True,
                    plot_noise_map=True,
                    plot_signal_to_noise_map=True,
                    plot_model_image=True,
                    plot_residual_map=True,
                    plot_normalized_residual_map=True,
                    plot_chi_squared_map=True,
                    plot_inversion_reconstruction=True,
                    plot_inversion_errors=True,
                    plot_inversion_residual_map=True,
                    plot_inversion_normalized_residual_map=True,
                    plot_inversion_chi_squared_map=True,
                    plot_inversion_regularization_weight_map=True,
                    plot_inversion_interpolated_reconstruction=True,
                    plot_inversion_interpolated_errors=True,
                    plot_subtracted_images_of_planes=True,
                    plot_model_images_of_planes=True,
                    plot_plane_images_of_planes=True,
                    include=self.include,
                    plotter=fits_plotter
                )

    def visualize_hyper_images(self, last_results):

        mask = self.masked_dataset.mask if self.include.mask else None

        if last_results is not None:
            plotter = self.plotter.plotter_with_new_output(
                output=mat_objs.Output(path=self.plotter.output.path + "hyper/"))

            sub_plotter = self.sub_plotter.plotter_with_new_output(
                output=mat_objs.Output(path=self.plotter.output.path + "hyper/"))

            if self.plot_hyper_model_image:
                hyper_plots.hyper_model_image(
                    hyper_model_image=last_results.hyper_model_image,
                    mask=mask,
                    include=self.include,
                    plotter=plotter
                )

            if self.plot_hyper_galaxy_images:
                hyper_plots.subplot_hyper_galaxy_images(

                    hyper_galaxy_image_path_dict=last_results.hyper_galaxy_image_path_dict,
                    mask=mask,
                    include=self.include,
                    sub_plotter=sub_plotter
                )


class PhaseInterferometerVisualizer(PhaseDatasetVisualizer):
    def __init__(self, masked_dataset, image_path):
        super(PhaseInterferometerVisualizer, self).__init__(
            masked_dataset=masked_dataset, image_path=image_path
        )

        self.plot_dataset_uv_wavelengths = plot_setting("plot_dataset_uv_wavelengths")
        self.plot_dataset_primary_beam = plot_setting("plot_dataset_primary_beam")

        self.plot_hyper_model_image = plot_setting("plot_hyper_model_image")
        self.plot_hyper_galaxy_images = plot_setting("plot_hyper_galaxy_images")

        self.visualize_interferometer()

    @property
    def masked_interferometer(self):
        return self.masked_dataset

    def visualize_interferometer(self):

        plotter = self.plotter.plotter_with_new_output(output=mat_objs.Output(path=self.plotter.output.path + "interferometer/"))

        if self.plot_dataset_as_subplot:
            aa.plot.interferometer.subplot_interferometer(
                interferometer=self.masked_dataset.interferometer,
                include=self.include,
                sub_plotter=self.sub_plotter
            )

        aa.plot.interferometer.individual(
            interferometer=self.masked_dataset.interferometer,
            plot_visibilities=self.plot_dataset_data,
            plot_u_wavelengths=self.plot_dataset_uv_wavelengths,
            plot_v_wavelengths=self.plot_dataset_uv_wavelengths,
            plot_primary_beam=self.plot_dataset_primary_beam,
            include=self.include,
            plotter=plotter
        )

    def visualize_fit(self, fit, during_analysis):

        plotter = self.plotter.plotter_with_new_output(
            output=mat_objs.Output(path=self.plotter.output.path + "fit_interferometer/"))

        if self.plot_fit_as_subplot:
            fit_interferometer_plots.subplot_fit_interferometer(
                fit=fit, include=self.include, sub_plotter=self.sub_plotter,
            )

            fit_interferometer_plots.subplot_fit_real_space(
                fit=fit,
                include=self.include,
                sub_plotter=self.sub_plotter
            )

        # if plot_inversion_as_subplot and fit.inversion is not None:
        #
        #     aa.plot.inversion.subplot(
        #         inversion=fit.inversion,
        #         mask=fit.masked_interferometer.real_space_mask,
        #         positions=positions,
        #         output_path=subplot_path,
        #         format="png",
        #     )

        fit_interferometer_plots.individuals(
            fit=fit,
            plot_visibilities=self.plot_fit_data,
            plot_noise_map=self.plot_fit_noise_map,
            plot_signal_to_noise_map=self.plot_fit_signal_to_noise_map,
            plot_model_visibilities=self.plot_fit_model_data,
            plot_residual_map=self.plot_fit_residual_map,
            plot_chi_squared_map=self.plot_fit_chi_squared_map,
            plot_normalized_residual_map=self.plot_fit_normalized_residual_map,
            plot_inversion_reconstruction=self.plot_fit_inversion_reconstruction,
            plot_inversion_errors=self.plot_fit_inversion_errors,
            # plot_inversion_residual_map=plot_inversion_residual_map,
            # plot_inversion_normalized_residual_map=plot_inversion_normalized_residual_map,
            # plot_inversion_chi_squared_map=plot_inversion_chi_squared_map,
            plot_inversion_regularization_weight_map=self.plot_fit_inversion_regularization_weights,
            plot_inversion_interpolated_reconstruction=self.plot_fit_inversion_interpolated_reconstruction,
            plot_inversion_interpolated_errors=self.plot_fit_inversion_interpolated_errors,
            include=self.include,
            plotter=plotter
        )

        if not during_analysis:

            if self.plot_fit_all_at_end_png:
                fit_interferometer_plots.individuals(
                    fit=fit,
                    plot_visibilities=True,
                    plot_noise_map=True,
                    plot_signal_to_noise_map=True,
                    plot_model_visibilities=True,
                    plot_residual_map=True,
                    plot_normalized_residual_map=True,
                    plot_chi_squared_map=True,
                    plot_inversion_reconstruction=True,
                    plot_inversion_errors=True,
                    # plot_inversion_residual_map=True,
                    # plot_inversion_normalized_residual_map=True,
                    # plot_inversion_chi_squared_map=True,
                    plot_inversion_regularization_weight_map=True,
                    plot_inversion_interpolated_reconstruction=True,
                    plot_inversion_interpolated_errors=True,
                    include=self.include,
                    plotter=plotter
                )

            if self.plot_fit_all_at_end_fits:
                fits_plotter = plotter.plotter_with_new_output(
                    output=mat_objs.Output(path=plotter.output.path + "/fits", format="fits")
                )

                fit_interferometer_plots.individuals(
                    fit=fit,
                    plot_visibilities=True,
                    plot_noise_map=True,
                    plot_signal_to_noise_map=True,
                    plot_model_visibilities=True,
                    plot_residual_map=True,
                    plot_normalized_residual_map=True,
                    plot_chi_squared_map=True,
                    plot_inversion_reconstruction=True,
                    plot_inversion_errors=True,
                    # plot_inversion_residual_map=True,
                    # plot_inversion_normalized_residual_map=True,
                    # plot_inversion_chi_squared_map=True,
                    plot_inversion_regularization_weight_map=True,
                    plot_inversion_interpolated_reconstruction=True,
                    plot_inversion_interpolated_errors=True,
                    include=self.include,
                    plotter=fits_plotter
                )


class HyperGalaxyVisualizer(AbstractVisualizer):
    def __init__(self, image_path):
        super().__init__(image_path)
        self.plot_hyper_galaxy_subplot = plot_setting("plot_hyper_galaxy_subplot")

    def visualize_hyper_galaxy(
        self,
        fit,
        hyper_fit,
        galaxy_image,
        contribution_map_in,
    ):
        hyper_plots.subplot_fit_hyper_galaxy(
            fit=fit, hyper_fit=hyper_fit, galaxy_image=galaxy_image, contribution_map_in=contribution_map_in,
            include=self.include, sub_plotter=self.sub_plotter
        )


