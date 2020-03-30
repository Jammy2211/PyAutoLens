from autoarray.exc import InversionException, GridException
from autofit.exc import FitException
from autolens.fit import fit
from autolens.pipeline import visualizer
from autolens.pipeline.phase.dataset import analysis as analysis_dataset


class Analysis(analysis_dataset.Analysis):
    def __init__(self, masked_imaging, cosmology, image_path=None, results=None):

        super(Analysis, self).__init__(cosmology=cosmology, results=results)

        self.visualizer = visualizer.PhaseImagingVisualizer(
            masked_dataset=masked_imaging, image_path=image_path, results=results
        )

        self.visualizer.visualize_hyper_images(
            hyper_galaxy_image_path_dict=self.hyper_galaxy_image_path_dict,
            hyper_model_image=self.hyper_model_image,
        )

        self.masked_dataset = masked_imaging

    @property
    def masked_imaging(self):
        return self.masked_dataset

    def fit(self, instance):
        """
        Determine the fit of a lens galaxy and source galaxy to the masked_imaging in this lens.

        Parameters
        ----------
        instance
            A model instance with attributes

        Returns
        -------
        fit : Fit
            A fractional value indicating how well this model fit and the model masked_imaging itself
        """

        self.associate_hyper_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)

        self.masked_dataset.check_positions_trace_within_threshold_via_tracer(
            tracer=tracer
        )

        self.masked_dataset.check_inversion_pixels_are_below_limit_via_tracer(
            tracer=tracer
        )

        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        try:
            fit = self.masked_imaging_fit_for_tracer(
                tracer=tracer,
                hyper_image_sky=hyper_image_sky,
                hyper_background_noise=hyper_background_noise,
            )

            return fit.figure_of_merit
        except InversionException or GridException as e:
            raise FitException from e

    def masked_imaging_fit_for_tracer(
        self, tracer, hyper_image_sky, hyper_background_noise
    ):

        return fit.FitImaging(
            masked_imaging=self.masked_dataset,
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

    def visualize(self, instance, during_analysis):
        instance = self.associate_hyper_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)
        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)
        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        fit = self.masked_imaging_fit_for_tracer(
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        if tracer.has_mass_profile:

            visualizer = self.visualizer.new_visualizer_with_preloaded_critical_curves_and_caustics(
                preloaded_critical_curves=tracer.critical_curves,
                preloaded_caustics=tracer.caustics,
            )

        else:

            visualizer = self.visualizer

        visualizer.visualize_ray_tracing(
            tracer=fit.tracer, during_analysis=during_analysis
        )
        visualizer.visualize_fit(fit=fit, during_analysis=during_analysis)
