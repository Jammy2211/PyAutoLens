import autofit as af
from autolens.lens import lens_fit, ray_tracing
from autolens.model.galaxy import galaxy as g
from autolens.plotters import visualizer


class Analysis(af.Analysis):
    def __init__(self, lens_imaging_data, cosmology, image_path=None, results=None):
        self.cosmology = cosmology
        self.visualizer = visualizer.PhaseImagingVisualizer(
            lens_imaging_data, image_path
        )

        self.lens_data = lens_imaging_data

        if results is not None and results.last is not None:
            last_results = results.last

            self.visualizer.plot_hyper_images(last_results)

            self.hyper_galaxy_image_1d_path_dict = (
                last_results.hyper_galaxy_image_1d_path_dict
            )

            self.hyper_model_image_1d = last_results.hyper_model_image_1d

            self.binned_hyper_galaxy_image_1d_path_dict = last_results.binned_hyper_galaxy_image_1d_path_dict(
                binned_grid=lens_imaging_data.grid.binned
            )

            self.visualizer.plot_hyper_images(last_results)

    def fit(self, instance):
        """
        Determine the fit of a lens galaxy and source galaxy to the lens_data in this lens.

        Parameters
        ----------
        instance
            A model instance with attributes

        Returns
        -------
        fit : Fit
            A fractional value indicating how well this model fit and the model lens_data itself
        """

        self.associate_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)

        self.lens_data.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer
        )
        self.lens_data.check_inversion_pixels_are_below_limit_via_tracer(
            tracer=tracer
        )

        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        fit = self.lens_imaging_fit_for_tracer(
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        return fit.figure_of_merit

    def associate_images(self, instance: af.ModelInstance) -> af.ModelInstance:
        """
        Takes images from the last result, if there is one, and associates them with galaxies in this phase
        where full-path galaxy names match.

        If the galaxy collection has a different name then an association is not made.

        e.g.
        galaxies.lens will match with:
            galaxies.lens
        but not with:
            galaxies.lens
            galaxies.source

        Parameters
        ----------
        instance
            A model instance with 0 or more galaxies in its tree

        Returns
        -------
        instance
           The input instance with images associated with galaxies where possible.
        """
        if hasattr(self, "hyper_galaxy_image_1d_path_dict"):
            for galaxy_path, galaxy in instance.path_instance_tuples_for_class(
                    g.Galaxy
            ):
                if galaxy_path in self.hyper_galaxy_image_1d_path_dict:
                    galaxy.hyper_model_image_1d = self.hyper_model_image_1d
                    galaxy.hyper_galaxy_image_1d = self.hyper_galaxy_image_1d_path_dict[
                        galaxy_path
                    ]
                    if (
                            hasattr(self, "binned_hyper_galaxy_image_1d_path_dict")
                            and self.binned_hyper_galaxy_image_1d_path_dict is not None
                    ):
                        galaxy.binned_hyper_galaxy_image_1d = self.binned_hyper_galaxy_image_1d_path_dict[
                            galaxy_path
                        ]
        return instance

    def hyper_image_sky_for_instance(self, instance):

        if hasattr(instance, "hyper_image_sky"):
            return instance.hyper_image_sky
        else:
            return None

    def hyper_background_noise_for_instance(self, instance):

        if hasattr(instance, "hyper_background_noise"):
            return instance.hyper_background_noise
        else:
            return None

    def lens_imaging_fit_for_tracer(
            self, tracer, hyper_image_sky, hyper_background_noise
    ):

        return lens_fit.LensImagingFit.from_lens_data_and_tracer(
            lens_data=self.lens_data,
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

    def visualize(self, instance, during_analysis):
        instance = self.associate_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)
        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)
        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        fit = self.lens_imaging_fit_for_tracer(
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )
        self.visualizer.plot_ray_tracing(fit.tracer, during_analysis)
        self.visualizer.plot_lens_imaging(fit, during_analysis)

    def tracer_for_instance(self, instance):
        return ray_tracing.Tracer.from_galaxies(
            galaxies=instance.galaxies, cosmology=self.cosmology
        )
