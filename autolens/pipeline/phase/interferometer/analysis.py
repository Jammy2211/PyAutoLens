import autofit as af
from autoarray.exc import InversionException
from autoastro.galaxy import galaxy as g
from autofit.exc import FitException
from autolens.fit import fit
from autolens.pipeline import visualizer
from autolens.pipeline.phase.dataset import analysis as analysis_data


class Analysis(analysis_data.Analysis):
    def __init__(self, masked_interferometer, cosmology, image_path=None, results=None):

        super(Analysis, self).__init__(cosmology=cosmology, results=results)

        self.visualizer = visualizer.PhaseInterferometerVisualizer(
            masked_dataset=masked_interferometer, image_path=image_path
        )

        self.visualizer.visualize_hyper_images(
            hyper_galaxy_image_path_dict=self.hyper_galaxy_image_path_dict,
            hyper_model_image=self.hyper_model_image,
        )

        self.masked_dataset = masked_interferometer

        result = analysis_data.last_result_with_use_as_hyper_dataset(results=results)

        if result is not None:

            self.hyper_galaxy_visibilities_path_dict = (
                result.hyper_galaxy_visibilities_path_dict
            )

            self.hyper_model_visibilities = result.hyper_model_visibilities

        else:

            self.hyper_galaxy_visibilities_path_dict = None
            self.hyper_model_visibilities = None

    @property
    def masked_interferometer(self):
        return self.masked_dataset

    def fit(self, instance):
        """
        Determine the fit of a lens galaxy and source galaxy to the masked_interferometer in this lens.

        Parameters
        ----------
        instance
            A model instance with attributes

        Returns
        -------
        fit : Fit
            A fractional value indicating how well this model fit and the model masked_interferometer itself
        """

        self.associate_hyper_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)

        self.masked_dataset.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer
        )
        self.masked_dataset.check_inversion_pixels_are_below_limit_via_tracer(
            tracer=tracer
        )

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        try:
            fit = self.masked_interferometer_fit_for_tracer(
                tracer=tracer, hyper_background_noise=hyper_background_noise
            )

            return fit.figure_of_merit
        except InversionException as e:
            raise FitException from e

    def associate_hyper_visibilities(
        self, instance: af.ModelInstance
    ) -> af.ModelInstance:
        """
        Takes visibilities from the last result, if there is one, and associates them with galaxies in this phase
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
           The input instance with visibilities associated with galaxies where possible.
        """
        if self.hyper_galaxy_visibilities_path_dict is not None:
            for galaxy_path, galaxy in instance.path_instance_tuples_for_class(
                g.Galaxy
            ):
                if galaxy_path in self.hyper_galaxy_visibilities_path_dict:
                    galaxy.hyper_model_visibilities = self.hyper_model_visibilities
                    galaxy.hyper_galaxy_visibilities = self.hyper_galaxy_visibilities_path_dict[
                        galaxy_path
                    ]

        return instance

    def masked_interferometer_fit_for_tracer(self, tracer, hyper_background_noise):

        return fit.InterferometerFit(
            masked_interferometer=self.masked_dataset,
            tracer=tracer,
            hyper_background_noise=hyper_background_noise,
        )

    def visualize(self, instance, during_analysis):

        self.associate_hyper_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)
        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        fit = self.masked_interferometer_fit_for_tracer(
            tracer=tracer, hyper_background_noise=hyper_background_noise
        )

        visualizer = self.visualizer.new_visualizer_with_preloaded_critical_curves_and_caustics(
            preloaded_critical_curves=tracer.critical_curves,
            preloaded_caustics=tracer.caustics,
        )

        visualizer.visualize_ray_tracing(
            tracer=fit.tracer, during_analysis=during_analysis
        )
        visualizer.visualize_fit(fit=fit, during_analysis=during_analysis)
