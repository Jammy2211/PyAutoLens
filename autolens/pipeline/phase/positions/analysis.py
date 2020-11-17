from autoconf import conf
from autoarray.inversion import pixelizations as pix
from autoarray.exc import InversionException, GridException
from autofit.exc import FitException
from autogalaxy.pipeline.phase.dataset import analysis as ag_analysis
from autolens.fit import fit
from autolens.pipeline import visualizer
from autolens.pipeline.phase.dataset import analysis as analysis_dataset

import copy


class Analysis(ag_analysis.Analysis, analysis_dataset.Analysis):
    def __init__(self, positions, solver, imaging, cosmology, results=None):

        super().__init__(cosmology=cosmology, results=results)

        self.solver = solver

        self.visualizer = visualizer.PhasePositionsVisualizer(
            positions=positions, imaging=imaging, results=results
        )

        self.imaging = imaging

    def log_likelihood_function(self, instance):
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

        tracer = self.tracer_for_instance(instance=instance)

        try:
            fit = self.positions_fit_for_tracer(tracer=tracer)

            return fit.figure_of_merit
        except (GridException) as e:
            raise FitException from e

    def positions_fit_for_tracer(self, tracer, hyper_image_sky, hyper_background_noise):

        return fit.FitImaging(
            masked_imaging=self.masked_dataset,
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

    def visualize(self, paths, instance, during_analysis):

        instance = self.associate_hyper_images(instance=instance)
        tracer = self.tracer_for_instance(instance=instance)
        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)
        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        fit = self.positions_fit_for_tracer(
            tracer=tracer,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        if tracer.has_mass_profile:

            try:

                visualizer = self.visualizer.new_visualizer_with_preloaded_critical_curves_and_caustics(
                    preloaded_critical_curves=tracer.critical_curves,
                    preloaded_caustics=tracer.caustics,
                )

            except (Exception, IndexError, ValueError):

                visualizer = self.visualizer

        else:

            visualizer = self.visualizer

        try:
            visualizer.visualize_ray_tracing(
                tracer=fit.tracer, during_analysis=during_analysis
            )
            visualizer.visualize_fit(fit=fit, during_analysis=during_analysis)
        except Exception:
            pass
