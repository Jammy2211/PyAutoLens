import autoarray as aa
import autofit as af
from autoarray.plot import mat_objs
from autoastro.plot import lensing_plotters
from autoastro.plot import fit_galaxy_plots
from autolens.plot import (
    ray_tracing_plots,
    hyper_plots,
    fit_imaging_plots,
    fit_interferometer_plots,
    inversion_plots,
)
import copy


def setting(section, name):
    return af.conf.instance.visualize_plots.get(section, name, bool)


def plot_setting(section, name):
    return setting(section, name)


class AbstractVisualizer:
    def __init__(self, image_path):

        self.plotter = lensing_plotters.Plotter(
            output=mat_objs.Output(path=image_path, format="png")
        )
        self.sub_plotter = lensing_plotters.SubPlotter(
            output=mat_objs.Output(path=image_path + "subplots/", format="png")
        )
        self.include = lensing_plotters.Include()

        self.plot_ray_tracing_all_at_end_png = plot_setting(
            "ray_tracing", "all_at_end_png"
        )
        self.plot_ray_tracing_all_at_end_fits = plot_setting(
            "ray_tracing", "all_at_end_fits"
        )
        self.plot_subplot_ray_tracing = plot_setting(
            "ray_tracing", "subplot_ray_tracing"
        )
        self.plot_ray_tracing_profile_image = plot_setting(
            "ray_tracing", "profile_image"
        )
        self.plot_ray_tracing_source_plane = plot_setting(
            "ray_tracing", "source_plane_image"
        )
        self.plot_ray_tracing_convergence = plot_setting("ray_tracing", "convergence")
        self.plot_ray_tracing_potential = plot_setting("ray_tracing", "potential")
        self.plot_ray_tracing_deflections = plot_setting("ray_tracing", "deflections")
        self.plot_ray_tracing_magnification = plot_setting(
            "ray_tracing", "magnification"
        )

    def new_visualizer_with_preloaded_critical_curves_and_caustics(
        self, preloaded_critical_curves, preloaded_caustics
    ):

        visualizer = copy.deepcopy(self)

        visualizer.include = visualizer.include.new_include_with_preloaded_critical_curves_and_caustics(
            preloaded_critical_curves=preloaded_critical_curves,
            preloaded_caustics=preloaded_caustics,
        )

        return visualizer


class PhaseGalaxyVisualizer(AbstractVisualizer):
    def __init__(self, image_path):
        super().__init__(image_path)
        self.plot_galaxy_fit_all_at_end_png = plot_setting(
            "galaxy_fit", "all_at_end_png"
        )
        self.plot_galaxy_fit_all_at_end_fits = plot_setting(
            "galaxy_fit", "all_at_end_fits"
        )
        self.plot_subplot_galaxy_fit = plot_setting("galaxy_fit", "subplot_galaxy_fit")
        self.plot_galaxy_fit_image = plot_setting("galaxy_fit", "image")
        self.plot_galaxy_fit_noise_map = plot_setting("galaxy_fit", "noise_map")
        self.plot_galaxy_fit_model_image = plot_setting("galaxy_fit", "model_image")
        self.plot_galaxy_fit_residual_map = plot_setting("galaxy_fit", "residual_map")
        self.plot_galaxy_fit_chi_squared_map = plot_setting(
            "galaxy_fit", "chi_squared_map"
        )

    def plot_galaxy_fit_subplot(self, fit):
        if self.plot_subplot_galaxy_fit:
            fit_galaxy_plots.subplot_fit_galaxy(
                fit=fit, include=self.include, sub_plotter=self.sub_plotter
            )

    def plot_fit_individuals(self, fit):

        fit_galaxy_plots.individuals(
            fit=fit,
            plot_image=self.plot_galaxy_fit_image,
            plot_noise_map=self.plot_galaxy_fit_noise_map,
            plot_model_image=self.plot_galaxy_fit_model_image,
            plot_residual_map=self.plot_galaxy_fit_residual_map,
            plot_chi_squared_map=self.plot_galaxy_fit_chi_squared_map,
            include=self.include,
            plotter=self.plotter,
        )


class PhaseDatasetVisualizer(AbstractVisualizer):
    def __init__(self, masked_dataset, image_path):
        super().__init__(image_path)
        self.masked_dataset = masked_dataset

        self.plot_subplot_dataset = plot_setting("dataset", "subplot_dataset")
        self.plot_dataset_data = plot_setting("dataset", "data")
        self.plot_dataset_noise_map = plot_setting("dataset", "noise_map")
        self.plot_dataset_psf = plot_setting("dataset", "psf")

        self.plot_dataset_signal_to_noise_map = plot_setting(
            "dataset", "signal_to_noise_map"
        )
        self.plot_dataset_absolute_signal_to_noise_map = plot_setting(
            "dataset", "absolute_signal_to_noise_map"
        )
        self.plot_dataset_potential_chi_squared_map = plot_setting(
            "dataset", "potential_chi_squared_map"
        )

        self.plot_fit_all_at_end_png = plot_setting("fit", "all_at_end_png")
        self.plot_fit_all_at_end_fits = plot_setting("fit", "all_at_end_fits")
        self.plot_subplot_fit = plot_setting("fit", "subplot_fit")
        self.plot_subplots_of_all_planes_fits = plot_setting(
            "fit", "subplots_of_plane_fits"
        )

        self.plot_fit_data = plot_setting("fit", "data")
        self.plot_fit_noise_map = plot_setting("fit", "noise_map")
        self.plot_fit_signal_to_noise_map = plot_setting("fit", "signal_to_noise_map")
        self.plot_fit_model_data = plot_setting("fit", "model_data")
        self.plot_fit_residual_map = plot_setting("fit", "residual_map")
        self.plot_fit_normalized_residual_map = plot_setting(
            "fit", "normalized_residual_map"
        )
        self.plot_fit_chi_squared_map = plot_setting("fit", "chi_squared_map")
        self.plot_fit_subtracted_images_of_planes = plot_setting(
            "fit", "subtracted_images_of_planes"
        )
        self.plot_fit_model_images_of_planes = plot_setting(
            "fit", "model_images_of_planes"
        )
        self.plot_fit_plane_images_of_planes = plot_setting(
            "fit", "plane_images_of_planes"
        )

        self.plot_subplot_inversion = plot_setting("inversion", "subplot_inversion")
        self.plot_inversion_reconstructed_image = plot_setting(
            "inversion", "reconstructed_image"
        )

        self.plot_inversion_reconstruction = plot_setting("inversion", "reconstruction")

        self.plot_inversion_errors = plot_setting("inversion", "errors")

        self.plot_inversion_residual_map = plot_setting("inversion", "residual_map")
        self.plot_inversion_normalized_residual_map = plot_setting(
            "inversion", "normalized_residual_map"
        )
        self.plot_inversion_chi_squared_map = plot_setting(
            "inversion", "chi_squared_map"
        )
        self.plot_inversion_regularization_weights = plot_setting(
            "inversion", "regularization_weight_map"
        )
        self.plot_inversion_interpolated_reconstruction = plot_setting(
            "inversion", "interpolated_reconstruction"
        )
        self.plot_inversion_interpolated_errors = plot_setting(
            "inversion", "interpolated_errors"
        )

        self.plot_hyper_model_image = plot_setting("hyper_galaxy", "model_image")
        self.plot_hyper_galaxy_images = plot_setting("hyper_galaxy", "images")

    def visualize_ray_tracing(self, tracer, during_analysis):

        plotter = self.plotter.plotter_with_new_output(
            path=self.plotter.output.path + "ray_tracing/"
        )

        if self.plot_subplot_ray_tracing:

            ray_tracing_plots.subplot_tracer(
                tracer=tracer,
                grid=self.masked_dataset.grid,
                positions=self.include.positions_from_masked_dataset(
                    masked_dataset=self.masked_dataset
                ),
                include=self.include,
                sub_plotter=self.sub_plotter,
            )

        ray_tracing_plots.individual(
            tracer=tracer,
            grid=self.masked_dataset.grid,
            positions=self.include.positions_from_masked_dataset(
                masked_dataset=self.masked_dataset
            ),
            plot_profile_image=self.plot_ray_tracing_profile_image,
            plot_source_plane=self.plot_ray_tracing_source_plane,
            plot_convergence=self.plot_ray_tracing_convergence,
            plot_potential=self.plot_ray_tracing_potential,
            plot_deflections=self.plot_ray_tracing_deflections,
            plot_magnification=self.plot_ray_tracing_magnification,
            include=self.include,
            plotter=plotter,
        )

        if not during_analysis:

            if self.plot_ray_tracing_all_at_end_png:

                ray_tracing_plots.individual(
                    tracer=tracer,
                    grid=self.masked_dataset.grid,
                    positions=self.include.positions_from_masked_dataset(
                        masked_dataset=self.masked_dataset
                    ),
                    plot_profile_image=True,
                    plot_source_plane=True,
                    plot_convergence=True,
                    plot_potential=True,
                    plot_deflections=True,
                    include=self.include,
                    plotter=plotter,
                )

            if self.plot_ray_tracing_all_at_end_fits:

                self.visualize_ray_tracing_in_fits(tracer=tracer)

    def visualize_ray_tracing_in_fits(self, tracer):

        fits_plotter = self.plotter.plotter_with_new_output(
            path=self.plotter.output.path + "/ray_tracing/fits/", format="fits"
        )

        ray_tracing_plots.individual(
            tracer=tracer,
            grid=self.masked_dataset.grid,
            plot_profile_image=True,
            plot_source_plane=True,
            plot_convergence=True,
            plot_potential=True,
            plot_deflections=True,
            include=self.include,
            plotter=fits_plotter,
        )

    def visualize_hyper_images(self, hyper_galaxy_image_path_dict, hyper_model_image):

        plotter = self.plotter.plotter_with_new_output(
            path=self.plotter.output.path + "hyper/"
        )

        sub_plotter = self.sub_plotter.plotter_with_new_output(
            path=self.plotter.output.path + "hyper/"
        )

        if self.plot_hyper_model_image:
            hyper_plots.hyper_model_image(
                hyper_model_image=hyper_model_image,
                mask=self.include.mask_from_masked_dataset(
                    masked_dataset=self.masked_dataset
                ),
                include=self.include,
                plotter=plotter,
            )

        if self.plot_hyper_galaxy_images:
            hyper_plots.subplot_hyper_galaxy_images(
                hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
                mask=self.include.mask_from_masked_dataset(
                    masked_dataset=self.masked_dataset
                ),
                include=self.include,
                sub_plotter=sub_plotter,
            )


class PhaseImagingVisualizer(PhaseDatasetVisualizer):
    def __init__(self, masked_dataset, image_path, results=None):
        super(PhaseImagingVisualizer, self).__init__(
            masked_dataset=masked_dataset, image_path=image_path
        )

        self.plot_dataset_psf = plot_setting("dataset", "psf")

        self.plot_hyper_model_image = plot_setting("hyper_galaxy", "model_image")
        self.plot_hyper_galaxy_images = plot_setting("hyper_galaxy", "images")

        self.visualize_imaging()

    @property
    def masked_imaging(self):
        return self.masked_dataset

    def visualize_imaging(self):

        plotter = self.plotter.plotter_with_new_output(
            path=self.plotter.output.path + "imaging/"
        )

        if self.plot_subplot_dataset:
            aa.plot.Imaging.subplot_imaging(
                imaging=self.masked_imaging.imaging,
                mask=self.include.mask_from_masked_dataset(
                    masked_dataset=self.masked_dataset
                ),
                positions=self.include.positions_from_masked_dataset(
                    masked_dataset=self.masked_dataset
                ),
                include=self.include,
                sub_plotter=self.sub_plotter,
            )

        aa.plot.Imaging.individual(
            imaging=self.masked_imaging.imaging,
            mask=self.include.mask_from_masked_dataset(
                masked_dataset=self.masked_dataset
            ),
            positions=self.include.positions_from_masked_dataset(
                masked_dataset=self.masked_dataset
            ),
            plot_image=self.plot_dataset_data,
            plot_noise_map=self.plot_dataset_noise_map,
            plot_psf=self.plot_dataset_psf,
            plot_signal_to_noise_map=self.plot_dataset_signal_to_noise_map,
            plot_absolute_signal_to_noise_map=self.plot_dataset_absolute_signal_to_noise_map,
            plot_potential_chi_squared_map=self.plot_dataset_potential_chi_squared_map,
            include=self.include,
            plotter=plotter,
        )

    def visualize_fit(self, fit, during_analysis):

        plotter = self.plotter.plotter_with_new_output(
            path=self.plotter.output.path + "fit_imaging/"
        )

        if self.plot_subplot_fit:
            fit_imaging_plots.subplot_fit_imaging(
                fit=fit, include=self.include, sub_plotter=self.sub_plotter
            )

        if self.plot_subplots_of_all_planes_fits:
            fit_imaging_plots.subplots_of_all_planes(
                fit=fit, include=self.include, sub_plotter=self.sub_plotter
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
            plot_subtracted_images_of_planes=self.plot_fit_subtracted_images_of_planes,
            plot_model_images_of_planes=self.plot_fit_model_images_of_planes,
            plot_plane_images_of_planes=self.plot_fit_plane_images_of_planes,
            include=self.include,
            plotter=plotter,
        )

        if fit.inversion is not None:

            if self.plot_subplot_inversion:
                inversion_plots.subplot_inversion(
                    inversion=fit.inversion,
                    image_positions=self.include.positions_from_fit(fit=fit),
                    source_positions=self.include.positions_of_plane_from_fit_and_plane_index(
                        fit=fit, plane_index=-1
                    ),
                    grid=self.include.inversion_image_pixelization_grid_from_fit(
                        fit=fit
                    ),
                    light_profile_centres=self.include.light_profile_centres_of_galaxies_from_obj(
                        obj=fit.tracer.image_plane
                    ),
                    mass_profile_centres=self.include.mass_profile_centres_of_galaxies_from_obj(
                        obj=fit.tracer.image_plane
                    ),
                    critical_curves=self.include.critical_curves_from_obj(
                        obj=fit.tracer
                    ),
                    caustics=self.include.caustics_from_obj(obj=fit.tracer),
                    include=self.include,
                    sub_plotter=self.sub_plotter,
                )

            plotter = self.plotter.plotter_with_new_output(
                path=self.plotter.output.path + "inversion/"
            )

            inversion_plots.individuals(
                inversion=fit.inversion,
                image_positions=self.include.positions_from_fit(fit=fit),
                source_positions=self.include.positions_of_plane_from_fit_and_plane_index(
                    fit=fit, plane_index=-1
                ),
                grid=self.include.inversion_image_pixelization_grid_from_fit(fit=fit),
                light_profile_centres=self.include.light_profile_centres_of_galaxies_from_obj(
                    obj=fit.tracer.image_plane
                ),
                mass_profile_centres=self.include.mass_profile_centres_of_galaxies_from_obj(
                    obj=fit.tracer.image_plane
                ),
                critical_curves=self.include.critical_curves_from_obj(obj=fit.tracer),
                caustics=self.include.caustics_from_obj(obj=fit.tracer),
                plot_reconstructed_image=self.plot_inversion_reconstruction,
                plot_reconstruction=self.plot_inversion_reconstruction,
                plot_errors=self.plot_inversion_errors,
                plot_residual_map=self.plot_inversion_residual_map,
                plot_normalized_residual_map=self.plot_inversion_normalized_residual_map,
                plot_chi_squared_map=self.plot_inversion_chi_squared_map,
                plot_regularization_weight_map=self.plot_inversion_regularization_weights,
                plot_interpolated_reconstruction=self.plot_inversion_interpolated_reconstruction,
                plot_interpolated_errors=self.plot_inversion_interpolated_errors,
                include=self.include,
                plotter=plotter,
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
                    plot_subtracted_images_of_planes=True,
                    plot_model_images_of_planes=True,
                    plot_plane_images_of_planes=True,
                    include=self.include,
                    plotter=plotter,
                )

                if fit.inversion is not None:
                    inversion_plots.individuals(
                        inversion=fit.inversion,
                        image_positions=self.include.positions_from_fit(fit=fit),
                        source_positions=self.include.positions_of_plane_from_fit_and_plane_index(
                            fit=fit, plane_index=-1
                        ),
                        grid=self.include.inversion_image_pixelization_grid_from_fit(
                            fit=fit
                        ),
                        light_profile_centres=self.include.light_profile_centres_of_galaxies_from_obj(
                            obj=fit.tracer.image_plane
                        ),
                        mass_profile_centres=self.include.mass_profile_centres_of_galaxies_from_obj(
                            obj=fit.tracer.image_plane
                        ),
                        critical_curves=self.include.critical_curves_from_obj(
                            obj=fit.tracer
                        ),
                        caustics=self.include.caustics_from_obj(obj=fit.tracer),
                        plot_reconstructed_image=True,
                        plot_reconstruction=True,
                        plot_errors=True,
                        plot_residual_map=True,
                        plot_normalized_residual_map=True,
                        plot_chi_squared_map=True,
                        plot_regularization_weight_map=True,
                        plot_interpolated_reconstruction=True,
                        plot_interpolated_errors=True,
                        include=self.include,
                        plotter=plotter,
                    )

            if self.plot_fit_all_at_end_fits:

                self.visualize_fit_in_fits(fit=fit)

    def visualize_fit_in_fits(self, fit):

        fits_plotter = self.plotter.plotter_with_new_output(
            path=self.plotter.output.path + "fit_imaging/fits/", format="fits"
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
            plot_subtracted_images_of_planes=True,
            plot_model_images_of_planes=True,
            plot_plane_images_of_planes=True,
            include=self.include,
            plotter=fits_plotter,
        )

        if fit.inversion is not None:

            fits_plotter = self.plotter.plotter_with_new_output(
                path=self.plotter.output.path + "inversion/fits/", format="fits"
            )

            inversion_plots.individuals(
                inversion=fit.inversion,
                plot_reconstructed_image=True,
                plot_interpolated_reconstruction=True,
                plot_interpolated_errors=True,
                include=self.include,
                plotter=fits_plotter,
            )


class PhaseInterferometerVisualizer(PhaseDatasetVisualizer):
    def __init__(self, masked_dataset, image_path):
        super(PhaseInterferometerVisualizer, self).__init__(
            masked_dataset=masked_dataset, image_path=image_path
        )

        self.plot_dataset_uv_wavelengths = plot_setting("dataset", "uv_wavelengths")
        self.plot_dataset_primary_beam = plot_setting("dataset", "primary_beam")

        self.plot_hyper_model_image = plot_setting("hyper_galaxy", "model_image")
        self.plot_hyper_galaxy_images = plot_setting("hyper_galaxy", "images")

        self.visualize_interferometer()

    @property
    def masked_interferometer(self):
        return self.masked_dataset

    def visualize_interferometer(self):

        plotter = self.plotter.plotter_with_new_output(
            path=self.plotter.output.path + "interferometer/"
        )

        if self.plot_subplot_dataset:
            aa.plot.Interferometer.subplot_interferometer(
                interferometer=self.masked_dataset.interferometer,
                include=self.include,
                sub_plotter=self.sub_plotter,
            )

        aa.plot.Interferometer.individual(
            interferometer=self.masked_dataset.interferometer,
            plot_visibilities=self.plot_dataset_data,
            plot_u_wavelengths=self.plot_dataset_uv_wavelengths,
            plot_v_wavelengths=self.plot_dataset_uv_wavelengths,
            plot_primary_beam=self.plot_dataset_primary_beam,
            include=self.include,
            plotter=plotter,
        )

    def visualize_fit(self, fit, during_analysis):

        plotter = self.plotter.plotter_with_new_output(
            path=self.plotter.output.path + "fit_interferometer/"
        )

        if self.plot_subplot_fit:
            fit_interferometer_plots.subplot_fit_interferometer(
                fit=fit, include=self.include, sub_plotter=self.sub_plotter
            )

            fit_interferometer_plots.subplot_fit_real_space(
                fit=fit, include=self.include, sub_plotter=self.sub_plotter
            )

        fit_interferometer_plots.individuals(
            fit=fit,
            plot_visibilities=self.plot_fit_data,
            plot_noise_map=self.plot_fit_noise_map,
            plot_signal_to_noise_map=self.plot_fit_signal_to_noise_map,
            plot_model_visibilities=self.plot_fit_model_data,
            plot_residual_map=self.plot_fit_residual_map,
            plot_chi_squared_map=self.plot_fit_chi_squared_map,
            plot_normalized_residual_map=self.plot_fit_normalized_residual_map,
            include=self.include,
            plotter=plotter,
        )

        if fit.inversion is not None:

            plotter = self.plotter.plotter_with_new_output(
                path=self.plotter.output.path + "inversion/"
            )

            # if self.plot_fit_inversion_as_subplot:
            #     inversion_plots.subplot_inversion(
            #         inversion=fit.inversion,
            #         image_positions=self.include.positions_from_fit(fit=fit),
            #         source_positions=self.include.positions_of_plane_from_fit_and_plane_index(
            #             fit=fit, plane_index=-1
            #         ),
            #         grid=self.include.inversion_image_pixelization_grid_from_fit(fit=fit),
            #         light_profile_centres=self.include.light_profile_centres_of_galaxies_from_obj(
            #             obj=fit.tracer.image_plane
            #         ),
            #         mass_profile_centres=self.include.mass_profile_centres_of_galaxies_from_obj(
            #             obj=fit.tracer.image_plane
            #         ),
            #         critical_curves=self.include.critical_curves_from_obj(obj=fit.tracer),
            #         caustics=self.include.caustics_from_obj(obj=fit.tracer),
            #         include=self.include,
            #         sub_plotter=self.sub_plotter,
            #     )

            inversion_plots.individuals(
                inversion=fit.inversion,
                image_positions=self.include.positions_from_fit(fit=fit),
                source_positions=self.include.positions_of_plane_from_fit_and_plane_index(
                    fit=fit, plane_index=-1
                ),
                grid=self.include.inversion_image_pixelization_grid_from_fit(fit=fit),
                light_profile_centres=self.include.light_profile_centres_of_galaxies_from_obj(
                    obj=fit.tracer.image_plane
                ),
                mass_profile_centres=self.include.mass_profile_centres_of_galaxies_from_obj(
                    obj=fit.tracer.image_plane
                ),
                critical_curves=self.include.critical_curves_from_obj(obj=fit.tracer),
                caustics=self.include.caustics_from_obj(obj=fit.tracer),
                plot_reconstructed_image=self.plot_inversion_reconstruction,
                plot_reconstruction=self.plot_inversion_reconstruction,
                plot_errors=self.plot_inversion_errors,
                #   plot_residual_map=self.plot_fit_inversion_residual_map,
                #   plot_normalized_residual_map=self.plot_fit_inversion_normalized_residual_map,
                #   plot_chi_squared_map=self.plot_fit_inversion_chi_squared_map,
                plot_regularization_weight_map=self.plot_inversion_regularization_weights,
                plot_interpolated_reconstruction=self.plot_inversion_interpolated_reconstruction,
                plot_interpolated_errors=self.plot_inversion_interpolated_errors,
                include=self.include,
                plotter=plotter,
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
                    include=self.include,
                    plotter=plotter,
                )

                if fit.inversion is not None:
                    inversion_plots.individuals(
                        inversion=fit.inversion,
                        image_positions=self.include.positions_from_fit(fit=fit),
                        source_positions=self.include.positions_of_plane_from_fit_and_plane_index(
                            fit=fit, plane_index=-1
                        ),
                        grid=self.include.inversion_image_pixelization_grid_from_fit(
                            fit=fit
                        ),
                        light_profile_centres=self.include.light_profile_centres_of_galaxies_from_obj(
                            obj=fit.tracer.image_plane
                        ),
                        mass_profile_centres=self.include.mass_profile_centres_of_galaxies_from_obj(
                            obj=fit.tracer.image_plane
                        ),
                        critical_curves=self.include.critical_curves_from_obj(
                            obj=fit.tracer
                        ),
                        caustics=self.include.caustics_from_obj(obj=fit.tracer),
                        plot_reconstructed_image=True,
                        plot_reconstruction=True,
                        plot_errors=True,
                        #     plot_residual_map=True,
                        #     plot_normalized_residual_map=True,
                        #     plot_chi_squared_map=True,
                        plot_regularization_weight_map=True,
                        plot_interpolated_reconstruction=True,
                        plot_interpolated_errors=True,
                        include=self.include,
                        plotter=plotter,
                    )

            if self.plot_fit_all_at_end_fits:
                fits_plotter = plotter.plotter_with_new_output(
                    path=plotter.output.path + "/fits/", format="fits"
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
                    include=self.include,
                    plotter=fits_plotter,
                )

                if fit.inversion is not None:
                    inversion_plots.individuals(
                        inversion=fit.inversion,
                        image_positions=self.include.positions_from_fit(fit=fit),
                        source_positions=self.include.positions_of_plane_from_fit_and_plane_index(
                            fit=fit, plane_index=-1
                        ),
                        grid=self.include.inversion_image_pixelization_grid_from_fit(
                            fit=fit
                        ),
                        light_profile_centres=self.include.light_profile_centres_of_galaxies_from_obj(
                            obj=fit.tracer.image_plane
                        ),
                        mass_profile_centres=self.include.mass_profile_centres_of_galaxies_from_obj(
                            obj=fit.tracer.image_plane
                        ),
                        critical_curves=self.include.critical_curves_from_obj(
                            obj=fit.tracer
                        ),
                        caustics=self.include.caustics_from_obj(obj=fit.tracer),
                        plot_reconstructed_image=True,
                        plot_interpolated_reconstruction=True,
                        plot_interpolated_errors=True,
                        include=self.include,
                        plotter=plotter,
                    )


class HyperGalaxyVisualizer(AbstractVisualizer):
    def __init__(self, image_path):
        super().__init__(image_path)
        self.plot_hyper_galaxy_subplot = plot_setting(
            "hyper_galaxy", "subplot_hyper_galaxy"
        )

    def visualize_hyper_galaxy(self, fit, hyper_fit, galaxy_image, contribution_map_in):
        hyper_plots.subplot_fit_hyper_galaxy(
            fit=fit,
            hyper_fit=hyper_fit,
            galaxy_image=galaxy_image,
            contribution_map_in=contribution_map_in,
            include=self.include,
            sub_plotter=self.sub_plotter,
        )
